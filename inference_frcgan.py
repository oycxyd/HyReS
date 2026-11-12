import glob
import os
import shutil

import cv2
from tqdm.auto import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from hyres_utils import hyres_degenerator

# x4 RRDBNet checkpoints (Real-ESRGAN architecture); add fine-tuned names here
X4_RRDB_MODELS = ('RealESRGAN_x4plus', 'RealESRNet_x4plus', 'net_g_405000_fullbrain')


def FRCGAN_restore(input_dir, output_dir, scalefactor=4, model_name='net_g_405000_fullbrain',
                   tile=0, tile_pad=10, pre_pad=0, half_precision=False, gpu_id=None,
                   model_dir='experiments/pretrained_models', file_format='h5'):
    """Restore a TIFF stack (folder) or a single TIFF with an FRC-GAN / Real-ESRGAN x4 model.

    If `input_dir` holds minmax.yaml/features.yaml (from hyres_generator), the restored stack is
    converted back into a datacube saved as `file_format` ('h5' or 'imzml').
    """
    if model_name not in X4_RRDB_MODELS:
        raise ValueError(f'Unknown model {model_name!r}: custom-trained model, check the '
                         f'hyperparameters and add it to X4_RRDB_MODELS.')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    model_path = os.path.join(model_dir, model_name + '.pth')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model weights not found: {model_path}')

    upsampler = RealESRGANer(scale=scalefactor, model_path=model_path, model=model,
                             tile=tile, tile_pad=tile_pad, pre_pad=pre_pad,
                             half=half_precision,  # only True on GPU
                             gpu_id=gpu_id)

    output_dir = os.path.join(output_dir, 'HyReS_x4')
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(input_dir):          # single image
        paths, src_dir = [input_dir], os.path.dirname(input_dir)
    else:
        paths, src_dir = sorted(glob.glob(os.path.join(input_dir, '*.tif'))), input_dir

    for path in tqdm(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        try:
            output, _ = upsampler.enhance(img, outscale=scalefactor)
        except RuntimeError as error:
            print('Warning - file may not be an image:', path, error)
            print('If you encounter CUDA out of memory, try a smaller `tile`.')
            continue
        cv2.imwrite(os.path.join(output_dir, f'{imgname}_HyReS_x4{extension}'), output)

    attrs = [os.path.join(src_dir, n) for n in ('minmax.yaml', 'features.yaml')]
    if all(os.path.isfile(a) for a in attrs):
        for a in attrs:
            shutil.copy(a, output_dir)
        hyres_degenerator(output_dir, file_format=file_format)
    else:
        print('missing attribute files (minmax/features): datacube not rebuilt.')
    return output_dir
