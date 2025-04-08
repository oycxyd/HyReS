import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


import numpy as np
from miplib.data.containers.image import Image
import miplib.processing.image as imops
from miplib.data.containers.fourier_correlation_data import FourierCorrelationData, FourierCorrelationDataCollection
from miplib.analysis.resolution import analysis as fsc_analysis, fourier_ring_correlation
from miplib.processing import windowing
import miplib.data.iterators.fourier_ring_iterators as iterators
from miplib.ui.cli import miplib_entry_point_options as options


class FRC(object):
    """
    A class for calcuating 2D Fourier ring correlation. Contains
    methods to calculate the FRC as well as to plot the results.
    """

    def __init__(self, image1, image2, iterator):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.shape != image2.shape or tuple(image1.spacing) != tuple(image2.spacing):
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.pixel_size = image1.spacing[0]

        # Expand to square
        image1 = imops.zero_pad_to_cube(image1)
        image2 = imops.zero_pad_to_cube(image2)

        self.iterator = iterator
        # Calculate power spectra for the input images.
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2))

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))

    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results.

        """
        radii = self.iterator.radii
        c1 = np.zeros(radii.shape, dtype=np.float32)
        c2 = np.zeros(radii.shape, dtype=np.float32)
        c3 = np.zeros(radii.shape, dtype=np.float32)
        points = np.zeros(radii.shape, dtype=np.float32)

        for ind_ring, idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]
            c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        # Calculate FRC
        spatial_freq = radii.astype(np.float32) / self.freq_nyq
        n_points = np.array(points)

        with np.errstate(divide="ignore", invalid="ignore"):
            frc = np.abs(c1) / np.sqrt(c2 * c3)
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)


        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set


@weighted_loss
def frc_loss(y_true, y_pred):
    # params for FRC
    args_list = ("None --bin-delta=1  --frc-curve-fit-type=smooth-spline "
            " --resolution-threshold-criterion=half-bit").split()          
    args = options.get_frc_script_options(args_list)
    average = True
    z_correction=0
    resolutions_pred = []
    resolutions_true = []
    y_true = y_true.cpu().detach().numpy()[:,0,:,:]
    y_pred = y_pred.cpu().detach().numpy()[:,0,:,:]
    # calculating FRC curves

    for n in range(len(y_pred)):
        try:
            # image = y_pred[:,:,1]
    #             image = image[:,100:200]
            frc_data1 = FourierCorrelationDataCollection()
            frc_data2 = FourierCorrelationDataCollection()
            image_p = Image(y_pred[n],[100, 100])
            image_t = Image(y_true[n],[100, 100])
            
            # Hamming Windowing
            if not args.disable_hamming:
                spacing = image_p.spacing
                image_p = Image(windowing.apply_hamming_window(image_p), spacing)
                image_t = Image(windowing.apply_hamming_window(image_t), spacing)
    
            # Split and make sure that the images are the same size
    #         image1, image2 = imops.checkerboard_split(image)
            image1p, image2p = imops.reverse_checkerboard_split(image_p)
            image1p, image2p = imops.zero_pad_to_matching_shape(image1p, image2p)
            image1t, image2t = imops.reverse_checkerboard_split(image_t)
            image1t, image2t = imops.zero_pad_to_matching_shape(image1t, image2t)
    
            # Run FRC
            iterator1 = iterators.FourierRingIterator(image1p.shape, args.d_bin)
            iterator2 = iterators.FourierRingIterator(image1t.shape, args.d_bin)
            frc_task1 = FRC(image1p, image2p, iterator1)
            frc_data1[0] = frc_task1.execute()
            frc_task2 = FRC(image1t, image2t, iterator2)
            frc_data2[0] = frc_task2.execute()
    
            if average:
    
                frc_data1[0].correlation["correlation"] *= 0.5
                frc_data1[0].correlation["correlation"] += 0.5*frc_task1.execute().correlation["correlation"]
                frc_data2[0].correlation["correlation"] *= 0.5
                frc_data2[0].correlation["correlation"] += 0.5*frc_task2.execute().correlation["correlation"]
    
    #         freqs1 = frc_data1[0].correlation["frequency"].copy()
    #         freqs2 = frc_data2[0].correlation["frequency"].copy()
    
            analyzer1 = fsc_analysis.FourierCorrelationAnalysis(frc_data1, image1p.spacing[0], args)
            analyzer2 = fsc_analysis.FourierCorrelationAnalysis(frc_data2, image1t.spacing[0], args)
    
            result1 = analyzer1.execute(z_correction=z_correction)[0]
            result2 = analyzer2.execute(z_correction=z_correction)[0]
            resolutions_pred.append(float(result1.resolution['resolution-point'][0]))
            resolutions_true.append(float(result2.resolution['resolution-point'][0]))
        except:
              print("Could not correlate patch pair "+ str(n))
    resolutions_pred = torch.tensor(resolutions_pred)
    resolutions_true = torch.tensor(resolutions_true)
    frc_loss = torch.abs(resolutions_pred - resolutions_true)                
    frc_loss = torch.sum(frc_loss, axis=0)/len(resolutions_pred)        
    
    return frc_loss.cuda()


@LOSS_REGISTRY.register()
class FRCLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='none'):
        super(FRCLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * frc_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
