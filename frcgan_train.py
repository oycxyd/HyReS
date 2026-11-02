"""Notebook-friendly wrapper around the BasicSR / Real-ESRGAN training pipeline for FRCGAN.

Usage
-----
    from frcgan_train import train_frcgan

    run = train_frcgan(
        name='frcgan_liver_frc0.1',
        dataroot_gt='data/liver/HR', dataroot_lq='data/liver/LR',
        meta_info='data/liver/meta_info.txt',
        pretrain_g='experiments/pretrained_models/RealESRGAN_x1plus.pth',
        frc_weight=0.1, total_iter=50_000,
    )
    run['latest_g']    # path to the newest generator checkpoint

Every call writes the fully resolved option file to ``options/generated/<name>.yml``
(with a provenance header: time, git commit, arguments), so each published result can be
reproduced with  ``python realesrgan/train.py -opt options/generated/<name>.yml``.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_OPTIONS = REPO_ROOT / 'options' / 'finetune_realesrgan_x4plus_pairdata.yml'


# ----------------------------------------------------------------------------------------
# option handling
# ----------------------------------------------------------------------------------------
def _ordered_loader():
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                           lambda loader, node: OrderedDict(loader.construct_pairs(node)))
    return Loader


def load_options(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=_ordered_loader())


def _plain(obj):
    """OrderedDict / Path / tuple -> plain YAML-safe types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def set_option(opt, key, value):
    """Set a nested option with a 'a:b:c' key (creates missing levels, unlike --force_yml)."""
    keys = key.split(':')
    node = opt
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = OrderedDict()
        node = node[k]
    node[keys[-1]] = value


def build_options(name, dataroot_gt, dataroot_lq, meta_info, base_options=DEFAULT_BASE_OPTIONS,
                  pretrain_g=None, pretrain_d=None,
                  frc_weight=0.1, l1_weight=1.0, frc_window=False,
                  perceptual_weight=None, gan_weight=None, gt_usm=None,
                  total_iter=None, lr=None, lr_milestones=None, batch_size=None, gt_size=None, num_workers=None,
                  val_gt=None, val_lq=None, val_freq=5000, val_save_img=False,
                  checkpoint_freq=None, seed=0, use_tb_logger=None, overrides=None):
    """Return the resolved BasicSR option dict. ``None`` keeps the value from ``base_options``.

    FRCGAN specific:
        frc_weight   weight of the FRC term; 0 -> plain L1 pixel loss (baseline).
        l1_weight    weight of the L1 term inside the pixel loss.
        frc_window   Hann window before the FFT in the FRC loss.
        gt_usm       unsharp-mask the GT for the L1/perceptual targets (Real-ESRGAN default True;
                     usually False for quantitative imaging data).
    overrides: {'a:b:c': value} for anything else, e.g. {'network_g:num_block': 6}.
    """
    opt = copy.deepcopy(load_options(base_options))
    opt['name'] = name
    if seed is not None:
        opt['manual_seed'] = seed

    train_ds = opt['datasets']['train']
    train_ds['dataroot_gt'] = str(dataroot_gt)
    train_ds['dataroot_lq'] = str(dataroot_lq)
    train_ds['meta_info'] = str(meta_info)
    for key, val in (('batch_size_per_gpu', batch_size), ('gt_size', gt_size),
                     ('num_worker_per_gpu', num_workers)):
        if val is not None:
            train_ds[key] = val

    path = opt.setdefault('path', OrderedDict())
    if pretrain_g is not None:
        path['pretrain_network_g'] = str(pretrain_g)
    if pretrain_d is not None:
        path['pretrain_network_d'] = str(pretrain_d)

    train = opt['train']
    if frc_weight and frc_weight > 0:
        train['pixel_opt'] = OrderedDict(type='L1FRCLoss', loss_weight=1.0, l1_weight=float(l1_weight),
                                         frc_weight=float(frc_weight), window=bool(frc_window),
                                         reduction='mean')
    else:
        train['pixel_opt'] = OrderedDict(type='L1Loss', loss_weight=float(l1_weight), reduction='mean')
    if perceptual_weight is not None:
        if perceptual_weight > 0:
            train['perceptual_opt']['perceptual_weight'] = float(perceptual_weight)
        else:
            train.pop('perceptual_opt', None)
    if gan_weight is not None:
        train['gan_opt']['loss_weight'] = float(gan_weight)
    if gt_usm is not None:
        opt['l1_gt_usm'] = opt['percep_gt_usm'] = bool(gt_usm)
    if total_iter is not None:
        train['total_iter'] = int(total_iter)
    if lr_milestones is not None:  # e.g. [30_000, 40_000]; [] = constant lr
        train.setdefault('scheduler', OrderedDict(type='MultiStepLR', gamma=0.5))
        train['scheduler']['milestones'] = [int(m) for m in lr_milestones]
    if lr is not None:
        for k in ('optim_g', 'optim_d'):
            if k in train:
                train[k]['lr'] = float(lr)

    if val_gt is not None and val_lq is not None:
        opt['datasets']['val'] = OrderedDict(name='validation', type='PairedImageDataset',
                                             dataroot_gt=str(val_gt), dataroot_lq=str(val_lq),
                                             io_backend=OrderedDict(type='disk'))
        opt['val'] = OrderedDict(val_freq=int(val_freq), save_img=bool(val_save_img), metrics=OrderedDict(
            psnr=OrderedDict(type='calculate_psnr', crop_border=0, test_y_channel=False),
            frc_auc=OrderedDict(type='calculate_frc_auc', window=True)))

    logger = opt.setdefault('logger', OrderedDict())
    if checkpoint_freq is not None:
        logger['save_checkpoint_freq'] = int(checkpoint_freq)
    if use_tb_logger is not None:
        logger['use_tb_logger'] = bool(use_tb_logger)

    for key, val in (overrides or {}).items():
        set_option(opt, key, val)
    return opt


def _git_commit(root):
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '') if out.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def write_options(opt, path, provenance=None, root=REPO_ROOT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f'# generated by frcgan_train.py on {time.strftime("%Y-%m-%d %H:%M:%S")}',
              f'# git commit: {_git_commit(root)}']
    if provenance:
        header.append('# arguments: ' + json.dumps(_plain(provenance), default=str))
    with open(path, 'w') as f:
        f.write('\n'.join(header) + '\n\n')
        yaml.safe_dump(_plain(opt), f, sort_keys=False, default_flow_style=None)
    return path


# ----------------------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------------------
def latest_checkpoint(models_dir, net='g'):
    files = glob.glob(os.path.join(str(models_dir), f'net_{net}_*.pth'))
    latest = os.path.join(str(models_dir), f'net_{net}_latest.pth')
    if os.path.isfile(latest):
        return latest
    numbered = [(int(m.group(1)), f) for f in files if (m := re.search(rf'net_{net}_(\d+)\.pth$', f))]
    return max(numbered)[1] if numbered else None


def _reset_basicsr_logger():
    """Allow a second in-process run to log to its own file."""
    import logging
    try:
        from basicsr.utils import logger as bl
        bl.initialized_logger.clear()
    except Exception:
        pass
    lg = logging.getLogger('basicsr')
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def _run_subprocess(cmd, cwd, log_every=1):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    try:
        for i, line in enumerate(proc.stdout):
            if i % log_every == 0:
                print(line, end='', flush=True)
        return proc.wait()
    except KeyboardInterrupt:  # stop button in Jupyter -> stop training cleanly
        proc.terminate()
        proc.wait()
        print('\nTraining interrupted - resume with auto_resume=True.')
        return -1


def _run_inprocess(opt_file, root, flags):
    import realesrgan.archs  # noqa: F401  (register Real-ESRGAN archs/data/models)
    import realesrgan.data  # noqa: F401
    import realesrgan.models  # noqa: F401
    from basicsr.train import train_pipeline

    _reset_basicsr_logger()
    argv = sys.argv
    sys.argv = ['realesrgan/train.py', '-opt', str(opt_file), *flags]
    cwd = os.getcwd()
    try:
        os.chdir(root)  # relative paths in the yml resolve as on the command line
        train_pipeline(str(root))
        return 0
    finally:
        sys.argv = argv
        os.chdir(cwd)


def train_frcgan(name, dataroot_gt, dataroot_lq, meta_info, *, mode='subprocess',
                 auto_resume=False, debug=False, dry_run=False, root=REPO_ROOT,
                 options_dir=None, log_every=1, **option_kwargs):
    """Build options, save them, and run Real-ESRGAN training.

    mode:
        'subprocess' (default) - runs ``python realesrgan/train.py`` and streams the log.
            GPU memory is freed afterwards, and the Jupyter stop button works.
        'inprocess' - calls basicsr's ``train_pipeline`` in this kernel (easier to debug,
            but restart the kernel before a new run to free GPU memory).
    dry_run: only write and return the option file.
    option_kwargs: passed to ``build_options`` (frc_weight, total_iter, lr, val_gt, ...).
    Returns dict with option file, experiment/model dirs, latest generator checkpoint, return code.
    """
    root = Path(root)
    opt = build_options(name, dataroot_gt, dataroot_lq, meta_info, **option_kwargs)
    options_dir = Path(options_dir) if options_dir else root / 'options' / 'generated'
    provenance = dict(name=name, dataroot_gt=dataroot_gt, dataroot_lq=dataroot_lq,
                      meta_info=meta_info, **option_kwargs)
    opt_file = write_options(opt, options_dir / f'{name}.yml', provenance, root)

    exp_name = ('debug_' + name) if debug and not name.startswith('debug') else name
    exp_root = Path(opt.get('path', {}).get('experiments_root') or root / 'experiments') / exp_name
    result = dict(options=str(opt_file), experiment=str(exp_root), models=str(exp_root / 'models'),
                  latest_g=None, returncode=None)
    if dry_run:
        return result

    flags = (['--auto_resume'] if auto_resume else []) + (['--debug'] if debug else [])
    if mode == 'subprocess':
        cmd = [sys.executable, str(root / 'realesrgan' / 'train.py'), '-opt', str(opt_file), *flags]
        rc = _run_subprocess(cmd, cwd=root, log_every=log_every)
    elif mode == 'inprocess':
        rc = _run_inprocess(opt_file, root, flags)
    else:
        raise ValueError("mode must be 'subprocess' or 'inprocess'")

    result['returncode'] = rc
    result['latest_g'] = latest_checkpoint(exp_root / 'models', 'g')
    if rc not in (0, -1):
        raise RuntimeError(f'training failed (exit code {rc}); see log in {exp_root}')
    return result
