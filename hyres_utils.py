import numpy as np
import h5py 
import os
import glob
import cv2
import tifffile as tiff
import yaml
from natsort import natsorted, ns

def hyres_generator(filename, output_dir, file_format = 'h5'):
    if file_format == 'h5':
        print(filename)
        h5 = h5py.File(os.path.join(output_dir, filename),'r')
        datacube = h5.get('datacube')
        dims = h5.get('dims')
        features = h5.get('mz')
        output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0])
        if os.path.isdir(output_dir) != True:
            os.mkdir(output_dir)
        minmax = np.empty([2, features.shape[0]])
        for n in range(0,features.shape[0]):
            ion_image = datacube[n,:]
            ion_image = ion_image.reshape((int(dims[1]),int(dims[0])))
            minmax[0,n] = np.min(ion_image)
            minmax[1,n] = np.max(ion_image)
            ion_image = (ion_image-minmax[0,n])/minmax[1,n]
            rgb = (np.dstack((ion_image,ion_image,ion_image))*65535.999).astype(np.uint16)
            tiff.imsave(os.path.join(output_dir, (filename+' feature %d' % n+'.tif')), rgb)
    with open(os.path.join(output_dir, 'minmax.yaml'), 'w') as f:
        yaml.dump(minmax, f)
    with open(os.path.join(output_dir, 'features.yaml'), 'w') as f:
        yaml.dump(np.array(features), f)
    return minmax, features

def hyres_degenerator(input_dir, file_format = 'h5', save = True):
    if file_format == 'h5':
        filename = os.path.basename(input_dir)
        with open(input_dir+'/minmax.yaml') as f:
            minmax = yaml.load(f,Loader=yaml.Loader)
        with open(input_dir+'/features.yaml') as f:
            features = yaml.load(f,Loader=yaml.Loader)      
        dims = [np.array(tiff.imread(glob.glob(input_dir+"/*.tif")[0])).shape[1],np.array(tiff.imread(glob.glob(input_dir+"/*.tif")[0])).shape[0]]
        datacube = np.empty([len(features),dims[0]*dims[1]])
        print(datacube.shape)
        for n, file in enumerate(natsorted((glob.glob(input_dir+"/*.tif")), alg=ns.IGNORECASE)):
            image_n = np.array(tiff.imread(file))
            image_n = (np.sum(image_n,2))/3
            image_n = (image_n-np.min(image_n))/np.max(image_n)
            image_n = image_n*minmax[1,n]+minmax[0,n]
            datacube[n,:] = np.reshape(image_n,(dims[0]*dims[1],))
    if save:
        if file_format == 'h5':
            with h5py.File(os.path.join(input_dir,'datacube_HyReS.h5'), "w") as f:
                f.create_dataset("datacube",  data=datacube)
                f.create_dataset("dimension",  data=dims)
                f.create_dataset("features",  data=features)
    return datacube,features,dims