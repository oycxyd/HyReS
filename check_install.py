"""Verify a HyReS / FRC-GAN installation.  Run from the repository root:

    python check_install.py

Checks every package the four notebooks import, the bundled basicsr/realesrgan/miplib,
the FRC loss registration and the CUDA build of PyTorch.
"""
import importlib
import sys

PACKAGES = [  # (import name, purpose)
    ('numpy', 'all'), ('scipy', 'all'), ('skimage', 'all'), ('cv2', 'all'),
    ('PIL', 'all'), ('imageio', 'NB4'), ('tifffile', 'all'), ('imagecodecs', 'LZW TIFF'),
    ('matplotlib', 'all'), ('pandas', 'NB4'), ('natsort', 'all'), ('yaml', 'all'),
    ('tqdm', 'all'), ('h5py', 'HDF5'), ('pyimzml', 'imzML'),
    ('addict', 'basicsr'), ('lmdb', 'basicsr'), ('yapf', 'basicsr'), ('tensorboard', 'basicsr'),
    ('brisque', 'crisque'), ('libsvm.svmutil', 'crisque (libsvm)'), ('pypiqe', 'crisque'),
    ('imutils', 'montage'), ('SimpleITK', 'miplib'), ('pims', 'miplib'), ('psf', 'miplib'),
    ('ipywidgets', 'notebooks'),
]

failed = []


def check(label, fn):
    try:
        out = fn()
        print(f'  OK    {label}' + (f'  ({out})' if out else ''))
    except Exception as exc:  # noqa: BLE001
        failed.append(label)
        print(f'  FAIL  {label}: {type(exc).__name__}: {exc}')


print(f'Python {sys.version.split()[0]}')
print('Packages')
for name, why in PACKAGES:
    check(f'{name:12s} [{why}]',
          lambda n=name: getattr(importlib.import_module(n), '__version__', ''))

print('PyTorch')


def _torch():
    import torch
    import torchvision
    info = f'torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA build {torch.version.cuda}'
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        arch = f'sm_{cap[0]}{cap[1]}'
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError(f'{info}: GPU {arch} not supported by this build')
        return f'{info}, GPU {torch.cuda.get_device_name(0)} ({arch})'
    print('  WARN  CUDA not available: training/inference will run on CPU '
          '(reinstall torch from the PyTorch CUDA index, see INSTALL.md)')
    return info


check('torch / torchvision', _torch)

print('Bundled modules')


def _basicsr():
    import basicsr
    from basicsr.utils.registry import LOSS_REGISTRY, METRIC_REGISTRY
    import basicsr.losses  # noqa: F401  (triggers registration)
    import basicsr.metrics  # noqa: F401
    for k in ('FRCLoss', 'L1FRCLoss'):
        if k not in LOSS_REGISTRY:
            raise RuntimeError(f'{k} not registered: is frc_loss.py in basicsr/basicsr/losses/?')
    if 'calculate_frc_auc' not in METRIC_REGISTRY:
        raise RuntimeError('calculate_frc_auc metric not registered')
    return f'basicsr {basicsr.__version__} from {basicsr.__file__}'


def _realesrgan():
    import realesrgan
    from realesrgan import RealESRGANer  # noqa: F401
    if 'site-packages' in realesrgan.__file__:
        raise RuntimeError(f'PyPI realesrgan found at {realesrgan.__file__}; '
                           'pip uninstall realesrgan and use the bundled folder')
    return realesrgan.__file__


def _miplib():
    from miplib.processing import windowing  # fails on old 'from scipy.signal import tukey'
    from miplib.analysis.resolution import fourier_ring_correlation  # noqa: F401
    from miplib.ui.cli import miplib_entry_point_options as options
    options.get_frc_script_options(['None'])
    return windowing.__file__


def _hyres():
    import hyres_utils  # noqa: F401
    import hyres_evals  # noqa: F401
    import inference_frcgan  # noqa: F401
    import frcgan_train  # noqa: F401


check('basicsr + FRC loss', _basicsr)
check('realesrgan (bundled)', _realesrgan)
check('miplib (bundled)', _miplib)
check('hyres_utils / hyres_evals / inference_frcgan / frcgan_train', _hyres)

print()
if failed:
    print(f'{len(failed)} check(s) failed: ' + ', '.join(f.split()[0] for f in failed))
    sys.exit(1)
print('All checks passed.')
