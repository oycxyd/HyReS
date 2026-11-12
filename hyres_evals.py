import numpy as np
import skimage.measure
import cv2
import glob
from natsort import natsorted, ns
from imutils import build_montages
from pypiqe import piqe
from brisque import BRISQUE
from functools import lru_cache
from hyres_utils import _load_array_yaml

def entropy_rank(input_dir, n_top=10):
    ents = []
    for n, file in enumerate(natsorted((glob.glob(input_dir+"/*.tif")), alg=ns.IGNORECASE)):
        image = cv2.imread(file)
        ents.append(skimage.measure.shannon_entropy(image)) 
    ref = (np.max(ents)-np.min(ents))/2
    # ref = np.median(ents) #constrain by median/mean entropy image
    deviation = (ents-ref)
    scale = 1.4826 * np.median(np.abs(deviation))
    if scale == 0:
        scale = np.std(ents)
    c = 1
    z = deviation / scale
    w = 1.0 / (1.0 + (z / c)**2)
    ents = ents*w
    rank = np.argsort(ents)[::-1][:n_top]
    rank = rank.astype(int)
    return ents, rank

class _BRISQUE(BRISQUE):
    """brisque 0.2.0 calls float() on 1-element arrays, which NumPy >= 2.4 rejects."""

    def scale_features(self, features):
        f = np.array([np.asarray(x, dtype=np.float64).ravel()[0] for x in features])
        lo = np.asarray(self.scale_params['min_'], dtype=np.float64).ravel()
        hi = np.asarray(self.scale_params['max_'], dtype=np.float64).ravel()
        return -1 + 2.0 * (f - lo) / (hi - lo)


@lru_cache(maxsize=1)
def _brisque_model():
    return _BRISQUE(url=False)  # load the SVM once, not per image


def to_uint8(image):
    """Any 2D/3D image (uint8/uint16/float) -> uint8, min-max scaled if not already uint8."""
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = np.nanmin(img), np.nanmax(img)
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return np.round(np.nan_to_num(img) * 255).astype(np.uint8)


def brisque_score(image):
    img = to_uint8(image)
    if img.ndim == 2:                       # BRISQUE needs 3 channels
        img = np.dstack([img] * 3)
    elif img.shape[2] == 4:                 # drop alpha
        img = img[..., :3]
    return float(_brisque_model().score(img=img))


def piqe_score(image):
    img = to_uint8(image)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return float(piqe(img)[0])              # piqe returns (score, activity, artifact, noise masks)

def crisque(image, eps=1e-6, return_parts=False):
    """100 - harmonic mean of BRISQUE and PIQE (both clipped to [eps, 100])."""
    bq = float(np.clip(brisque_score(image), eps, 100))
    pq = float(np.clip(piqe_score(image), eps, 100))
    score = 100 - 2 * (1 / bq + 1 / pq) ** -1
    return (score, bq, pq) if return_parts else score

def info_montage(input_dir, rank, use_features=True):
    info_images = []
    paths = natsorted(glob.glob(input_dir+"/*.tif"), alg=ns.IGNORECASE)  # same order as entropy_rank
    paths = np.array(paths)[rank]
    if use_features:
        features = _load_array_yaml(input_dir + '/features.yaml')
    for n, imagePath in enumerate(paths):
        image = cv2.imread(imagePath)
        if use_features:
            feature = features[rank[n]]
            image = cv2.putText(image, 'feature='+str(feature), (10,10), cv2.FONT_HERSHEY_PLAIN, 
                           0.9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            image = cv2.putText(image, 'feature=feature %d'%n, (10,10), cv2.FONT_HERSHEY_PLAIN, 
                           0.9, (255, 255, 255), 1, cv2.LINE_AA)
        info_images.append(image)
    montages = build_montages(info_images, (128, 196), (5, 2))
    return montages
