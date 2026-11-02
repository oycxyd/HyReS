"""Differentiable Fourier ring correlation (FRC) loss for BasicSR / Real-ESRGAN.

FRC between prediction and ground truth for ring k:

    FRC_k = Re( sum_{ring k} F_pred * conj(F_gt) ) / sqrt( sum |F_pred|^2 * sum |F_gt|^2 )

The loss is 1 - AUC of the FRC curve over normalised spatial frequency (0 = perfect).
Rings follow the original implementation: ring k holds frequencies with k < |f| <= k+1
(DC excluded), up to the Nyquist radius min(H, W) // 2.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

try:  # registry is optional, so the module can also be used standalone
    from basicsr.utils.registry import LOSS_REGISTRY, METRIC_REGISTRY
except ImportError:  # pragma: no cover
    LOSS_REGISTRY = METRIC_REGISTRY = None

_REDUCTIONS = ('none', 'mean', 'sum')


@lru_cache(maxsize=16)
def _ring_index(h, w, device_str):
    """Ring index per FFT pixel (unshifted layout); -1 for DC / outside Nyquist circle."""
    device = torch.device(device_str)
    fy = torch.fft.fftfreq(h, d=1.0 / h, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(w, d=1.0 / w, device=device, dtype=torch.float64)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    idx = torch.ceil(r).long() - 1            # k < r <= k+1  ->  k
    n_rings = min(h, w) // 2
    idx[(idx < 0) | (idx >= n_rings)] = -1
    return idx.flatten(), n_rings


def _hann2d(h, w, device, dtype):
    return torch.outer(torch.hann_window(h, periodic=False, device=device, dtype=dtype),
                       torch.hann_window(w, periodic=False, device=device, dtype=dtype))


def frc_curve(pred, target, eps=1e-8, window=False):
    """FRC curve per sample.

    Args:
        pred, target: tensors (B, C, H, W) or (H, W) / (C, H, W). Channels are pooled.
        eps: stabiliser for the denominator (keeps gradients finite).
        window: apply a 2D Hann window before the FFT (reduces edge leakage).
    Returns:
        frc: (B, n_rings) correlation per ring.
        freq: (n_rings,) spatial frequency normalised to [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f'shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}')
    while pred.dim() < 4:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    pred, target = pred.float(), target.float()        # FFT in fp32 (AMP-safe)
    b, c, h, w = pred.shape

    if window:
        win = _hann2d(h, w, pred.device, pred.dtype)
        pred, target = pred * win, target * win

    f1 = torch.fft.fft2(pred)
    f2 = torch.fft.fft2(target)
    cross = (f1 * f2.conj()).real.sum(1).reshape(b, -1)   # pool channels
    p1 = (f1.abs() ** 2).sum(1).reshape(b, -1)
    p2 = (f2.abs() ** 2).sum(1).reshape(b, -1)

    idx, n_rings = _ring_index(h, w, str(pred.device))
    valid = idx >= 0
    ring = idx[valid]

    def ring_sum(x):
        return x.new_zeros(b, n_rings).index_add_(1, ring, x[:, valid])

    c1, c2, c3 = ring_sum(cross), ring_sum(p1), ring_sum(p2)
    frc = c1 / torch.sqrt(c2 * c3 + eps)
    freq = torch.arange(n_rings, device=pred.device, dtype=frc.dtype)
    freq = freq / freq.max().clamp_min(1)
    return frc, freq


def frc_loss(pred, target, reduction='mean', eps=1e-8, window=False):
    """1 - area under the FRC curve (trapezoidal), per sample then reduced."""
    frc, freq = frc_curve(pred, target, eps=eps, window=window)
    auc = torch.trapezoid(frc, freq, dim=1)          # (B,), 1.0 for identical images
    loss = 1.0 - auc
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


class FRCLoss(nn.Module):
    """FRC loss module (BasicSR-style signature)."""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-8, window=False):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_REDUCTIONS}')
        self.loss_weight, self.reduction, self.eps, self.window = loss_weight, reduction, eps, window

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * frc_loss(pred, target, self.reduction, self.eps, self.window)


class L1FRCLoss(nn.Module):
    """l1_weight * L1 + frc_weight * FRC - drop-in replacement for ``pixel_opt``."""

    def __init__(self, loss_weight=1.0, l1_weight=1.0, frc_weight=0.1, reduction='mean',
                 eps=1e-8, window=False):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_REDUCTIONS}')
        self.loss_weight, self.l1_weight, self.frc_weight = loss_weight, l1_weight, frc_weight
        self.reduction, self.eps, self.window = reduction, eps, window

    def forward(self, pred, target, weight=None, **kwargs):
        red = 'mean' if self.reduction == 'none' else self.reduction
        l1 = F.l1_loss(pred, target, reduction=red)
        frc = frc_loss(pred, target, red, self.eps, self.window)
        return self.loss_weight * (self.l1_weight * l1 + self.frc_weight * frc)


def calculate_frc_auc(img, img2, crop_border=0, window=True, **kwargs):
    """Validation metric: FRC-AUC (0..1, higher is better) between two uint8/float HWC images.

    Signature matches BasicSR metrics, so it can be used in the yml ``val: metrics:`` block
    as ``type: calculate_frc_auc``.
    """
    a = torch.as_tensor(img, dtype=torch.float32)
    b = torch.as_tensor(img2, dtype=torch.float32)
    if crop_border:
        a = a[crop_border:-crop_border, crop_border:-crop_border, ...]
        b = b[crop_border:-crop_border, crop_border:-crop_border, ...]
    if a.dim() == 3:  # HWC -> CHW
        a, b = a.permute(2, 0, 1), b.permute(2, 0, 1)
    with torch.no_grad():
        frc, freq = frc_curve(a, b, window=window)
        return float(torch.trapezoid(frc[0], freq))


if LOSS_REGISTRY is not None:
    LOSS_REGISTRY.register()(FRCLoss)
    LOSS_REGISTRY.register()(L1FRCLoss)
    METRIC_REGISTRY.register()(calculate_frc_auc)
