import numpy as np
import skimage.measure
import cv2
import glob
import yaml
from natsort import natsorted, ns
from imutils import build_montages

def entropy_rank(input_dir):
    ents = []
    for n, file in enumerate(natsorted((glob.glob(input_dir+"/*.tif")), alg=ns.IGNORECASE)):
        image = cv2.imread(file)
        ents.append(skimage.measure.shannon_entropy(image))
    rank = np.argsort(ents)[::-1][:10]
    rank = rank.astype(int)
    return ents, rank

def info_montage(input_dir, rank, use_features=True):
    info_images = []
    paths = natsorted((glob.glob(input_dir+"/*.tif")))
    paths = np.array(paths)[rank]
    for n, imagePath in enumerate(paths):
        image = cv2.imread(imagePath)
        if use_features:
            with open(input_dir+'/features.yaml') as f:
                features = yaml.load(f,Loader=yaml.Loader)
                feature = features[rank[n]]
                image = cv2.putText(image, 'feature='+str(feature), (10,10), cv2.FONT_HERSHEY_PLAIN, 
                           0.9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            image = cv2.putText(image, 'feature=feature %d'%n, (10,10), cv2.FONT_HERSHEY_PLAIN, 
                           0.9, (255, 255, 255), 1, cv2.LINE_AA)
        info_images.append(image)
    montages = build_montages(info_images, (128, 196), (5, 2))
    return montages