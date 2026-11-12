# HyReS
Hyperspectral Image Restoration and Super-resolution with Physics-Aware Deep Learning for Biomedical Applications

latest arxiv: https://arxiv.org/html/2503.02908v2

<img width="1400" height="1044" alt="grafik" src="https://github.com/user-attachments/assets/c0047ce4-c358-4da0-9f3f-2bac28694007" />


# Installing HyReS / FRCGAN

This guide sets up one environment that runs all four notebooks:

| Notebook | Purpose | Uses |
|---|---|---|
| `1.FRCGAN_data_generation.ipynb` | HDF5 / imzML datacube → TIFF ion-image stack, entropy ranking | `hyres_utils`, `hyres_evals` |
| `2.FRCGAN_finetune_clean.ipynb` | Fine-tune Real-ESRGAN with the FRC loss | `frcgan_train`, `basicsr`, `realesrgan` |
| `3.FRCGAN_inference.ipynb` | Restore TIFF stacks, rebuild HDF5 / imzML datacube | `inference_frcgan`, `hyres_utils` |
| `4.FRC_res_calc_multiformat.ipynb` | FRC resolution of folders / HDF5 / imzML, CRISQUE | `miplib`, `hyres_evals` |

Tested with Python 3.12 (Windows 11, RTX 5080, CUDA 13.2 wheels) and Python 3.14 (Linux, CPU).
Supported: Python 3.11–3.14.

## Repository layout

Run every notebook and script from the repository root. The three bundled packages are
imported from there and must not be replaced by their PyPI versions.

```
HyReS/
├── basicsr/                  bundled, modified/patched BasicSR (installed editable, --no-deps)
│   └── basicsr/losses/frc_loss.py
├── realesrgan/               bundled modified Real-ESRGAN (imported from the repo root, not installed)
├── miplib/                   bundled modified 2021 miplib (imported from the repo root, not installed)
├── options/finetune_realesrgan_x1plus_pairdata.yml
├── scripts/generate_meta_info_pairdata.py
├── experiments/pretrained_models/   *.pth weights (e.g. net_g_405000_fullbrain.pth)
├── hyres_utils.py  hyres_evals.py  inference_frcgan.py  frcgan_train.py
├── check_install.py
├── requirements.txt
└── 1.…ipynb  2.…ipynb  3.…ipynb  4.…ipynb
```

## Installation

### 1. Create the environment

```bash
conda create -n hyres python=3.12 -y
conda activate hyres
cd path/to/HyReS            # repository root
```

Use only `pip` inside this environment. Do not `conda install` torch, numpy or
opencv as well, or two copies will shadow each other.

### 2. Install PyTorch from the PyTorch index

Install PyTorch from the PyTorch index first. On Windows, the PyPI wheels are CPU-only,
which causes `AssertionError: Torch not compiled with CUDA enabled`.

```bash
# NVIDIA GPU (RTX 50-series / Blackwell needs CUDA >= 12.8 wheels; cu132 used here)
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132

# CPU only
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
```

Pick the CUDA tag your driver supports (check the "CUDA Version" shown by `nvidia-smi`).
Wheels for a newer CUDA than the driver supports will not see the GPU.

### 3. Install all other dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` already covers HDF5 and imzML I/O, BasicSR's runtime dependencies,
CRISQUE (BRISQUE + PIQE), miplib's dependencies and JupyterLab. Install it after torch,
so pip never replaces the CUDA build.

### 4. Install the bundled BasicSR

```bash
pip install --no-deps --use-pep517 --no-build-isolation -e ./basicsr
```

- `--no-deps`: prevents BasicSR from pulling `tb-nightly` or a different torch.
- `--no-build-isolation`: `setup.py` imports torch, and an isolated build env has no torch.

Do not `pip install realesrgan`. The PyPI package requires PyPI `basicsr`, `gfpgan` and
`facexlib`, which conflict with the bundled copy. If you installed it earlier, run
`pip uninstall realesrgan`.

### 5. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name hyres --display-name "Python (hyres)"
jupyter lab
```

In each notebook, select the Python (hyres) kernel. A different kernel is the most common
reason for "module not found" or "CUDA not available" errors in notebooks.

### 6. Verify

```bash
python check_install.py
```

It imports every package the notebooks use. It also checks that:

- `FRCLoss`, `L1FRCLoss` and `calculate_frc_auc` are registered in BasicSR;
- `realesrgan` and `miplib` come from the repository, not site-packages;
- the GPU's architecture (e.g. `sm_120` for the RTX 5080) is in the torch build.

The run should end with "All checks passed."

## Patches in the bundled packages

These changes are already applied in the repository and are listed here for reproducibility.
Re-apply them if you ever refresh a bundled package from upstream.

| Package | File | Change | Reason |
|---|---|---|---|
| basicsr | `basicsr/data/degradations.py` | `torchvision.transforms.functional_tensor` → `torchvision.transforms.functional` | removed in torchvision ≥ 0.17 |
| basicsr | `setup.py` (`get_version`) | `exec` into an explicit namespace dict | Python ≥ 3.13 `locals()` semantics (`KeyError: '__version__'`) |
| basicsr | `requirements.txt` | `tb-nightly` → `tensorboard` | keeps `pip check` clean |
| basicsr | `basicsr/losses/frc_loss.py` (new) | differentiable `FRCLoss`, `L1FRCLoss`, `calculate_frc_auc` metric | FRC-GAN loss; auto-registered |
| miplib | `miplib/analysis/resolution/analysis.py` | fitting initialisation & GOF check modified | initial release incomplete & does not have GOF |
| miplib | `miplib/processing/windowing.py` | `from scipy.signal.windows import tukey` | removed from `scipy.signal` in SciPy ≥ 1.13 |

The miplib on GitHub/PyPI was rewritten in 2026 with a different API. Notebook 4 targets the
bundled 2021 API (`FRC(image1, image2, iterator)`, `FourierCorrelationDataCollection`,
`args.d_bin`), so keep the bundled copy.

## Data and model paths

- Notebook 2: set `DATA_ROOT` (default `/content/drive/MyDrive` from Colab) to your
  training data folder.
- Notebook 3: weights load from `experiments/pretrained_models/<model_name>.pth`. Add
  custom fine-tuned model names to `X4_RRDB_MODELS` in `inference_frcgan.py`, or pass
  `model_dir=`. Pass `file_format='imzml'` to write the restored datacube as imzML instead
  of HDF5.
- `minmax.yaml` / `features.yaml` are now written as plain YAML lists. Files written
  by earlier versions (pickled NumPy arrays) are still read correctly.

## Troubleshooting

| Known Error | Fix |
|---|---|
| `Torch not compiled with CUDA enabled` | CPU wheel installed: `pip uninstall -y torch torchvision`, `pip cache remove "torch*"`, reinstall step 2; check the notebook kernel |
| `no kernel image is available for execution on the device` | torch build too old for the GPU: RTX 50-series needs cu128 or newer |
| `KeyError: '__version__'` while installing basicsr | `setup.py` patch missing (see table) |
| `ModuleNotFoundError: No module named 'torch'` while installing basicsr | add `--no-build-isolation` |
| `ResolutionImpossible ... basicsr` | you tried `pip install realesrgan`; don't, use the bundled folder |
| `No module named 'torchvision.transforms.functional_tensor'` | `degradations.py` patch missing |
| `cannot import name 'tukey' from 'scipy.signal'` | miplib `windowing.py` patch missing |
| `<COMPRESSION.LZW: 5> requires the 'imagecodecs' package` | `pip install -r requirements.txt` (adds `imagecodecs`) |
| `TypeError: only 0-dimensional arrays can be converted` in BRISQUE | use `hyres_evals.crisque` / `brisque_score` (patched for NumPy ≥ 2.4) |
| `ModuleNotFoundError: hyres_eval` | the module is `hyres_evals` (with an s) |


# Generating training images

See `1.FRCGAN_data_generation.ipynb`

# Training your own FRCGAN

See `2.FRCGAN_finetune_clean.ipynb`

# Inference by pre-trained FRCGAN weights

See `3. FRCGAN_inference.ipynb`

# Evaluation & visualisation of restored images

See `4.FRC_res_calc_multiformat.ipynb`

