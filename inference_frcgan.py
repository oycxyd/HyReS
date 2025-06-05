from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import os
import glob
import shutil
from tqdm.auto import tqdm
from hyres_utils import hyres_degenerator

def FRCGAN_restore (input_dir, output_dir, scalefactor=4, model_name = 'net_g_405000_fullbrain', tile=0, tile_pad=10, pre_pad=0, half_precision=False, gpu_id=None):
    if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'net_g_405000_fullbrain']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        raise Exception(f'using custom-trained model, check the hyperparameters.')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')
        
    FRCGAN_upsampler = RealESRGANer(
    scale=scalefactor,
    model_path=model_path,
    model=model,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=half_precision, # only set to True if usin GPU
    gpu_id=gpu_id)
    
    output_dir =os.path.join(output_dir,'HyReS_x4')
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_dir):
        paths = [input_dir]# if single image
    else:
        paths = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
        
    for idx, path in enumerate(tqdm(paths)):
        imgname, extension = os.path.splitext(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        try:
            output, _ = FRCGAN_upsampler.enhance(img, outscale=scalefactor)
        except RuntimeError as error:
            print('Warning - file may not be image', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            extension = extension[1:]
            suffix = 'HyReS_x4'
            save_path = os.path.join(output_dir, f'{imgname}_{suffix}.{extension}')
            cv2.imwrite(save_path, output)
            try:
                shutil.copy(os.path.join(input_dir,'minmax.yaml'), output_dir)
                shutil.copy(os.path.join(input_dir,'features.yaml'), output_dir)
            except:
                print('missing attributes files (minmax/features).')
    _,_,_ = hyres_degenerator(output_dir)