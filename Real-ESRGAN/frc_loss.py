# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 06:57:18 2022

@author: OYCX
"""
import tensorflow as tf
import keras.backend as K
import numpy as np
from miplib.data.containers.image import Image
import miplib.processing.image as imops
from miplib.data.containers.fourier_correlation_data import FourierCorrelationData,\
FourierCorrelationDataCollection
from miplib.analysis.resolution import analysis as fsc_analysis, fourier_ring_correlation
from miplib.processing import windowing
import miplib.data.iterators.fourier_ring_iterators as iterators
from miplib.ui.cli import miplib_entry_point_options as options

def frc_loss(y_true, y_pred, pixel_size):
    # params for FRC
    args_list = ("None --bin-delta=1  --frc-curve-fit-type=smooth-spline "
            " --resolution-threshold-criterion=half-bit").split()          
    args = options.get_frc_script_options(args_list)
    average = True
    z_correction=0
    resolutions_pred = []
    resolutions_true = []
    
    # calculating FRC curves

    for n in range(len(y_pred)):
        try:
            # image = y_pred[:,:,1]
    #             image = image[:,100:200]
            frc_data1 = FourierCorrelationDataCollection()
            frc_data2 = FourierCorrelationDataCollection()
            image_p = Image(y_pred[n],[pixel_size, pixel_size])
            image_t = Image(y_true[n],[pixel_size, pixel_size])
            
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
    resolutions_pred = tf.convert_to_tensor(resolutions_pred)
    resolutions_true = tf.convert_to_tensor(resolutions_true)
    frc_loss = K.abs(resolutions_pred - resolutions_true)                
    frc_loss = K.sum(frc_loss, axis=0)/len(resolutions_pred)        
    
    return frc_loss

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