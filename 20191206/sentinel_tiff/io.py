# -*- coding: utf-8 -*-
"""
Created on 2019-06-03

@authors: gurbet, jim
"""

import numpy as np
import skimage.io
from PIL import Image

def frame_image( image, frame_size = None ):
    if frame_size is None: return image
    assert( frame_size > 0 )
    i_c = image.shape[0]
    i_x = image.shape[1]
    i_y = image.shape[2]
    frame_c = i_c
    frame_x = i_x + 2*frame_size
    frame_y = i_y + 2*frame_size
    framed_img = np.zeros((frame_c, frame_x, frame_y))
    framed_img[:, frame_size:frame_size+i_x, frame_size:frame_size+i_y] = image
    return framed_img

def read_mask_file( image_name ):
    '''
        read_mask_file
        string image_name: name of an image file in tiff format

        This function reads a mask file, where each pixels contains a color
        gradient between 0-65535 for 4 channels (Red, Green, Blue and Alpha)
        to define classes based on color.

        The algorithm then forces each RGB channel to be 0 or 1, using
        as cutoff half of 65535. Then, it turns the RGB values from binary
        to decimal, by multiplying [r g b a] and [1 2 4 0]

        returns: a numpy array with integers between 0-7 for each class
    '''
    # class_mask = skimage.io.imread( image_name )
    # class_mask = np.dot( (class_mask/65535+.5).astype(int), [1, 2, 4, 0])
    class_mask = np.asarray(Image.open( image_name ))
    class_mask = class_mask[:,:,:3]>0
    class_mask = np.dot( class_mask.astype(np.uint8), [4, 2, 1] )
    return class_mask


def read_image_file( image_name ):
    '''
        read_image_file
        string image_name: name of an image file in tiff format

        This function reads a sentinel-2 TIFF file, where each layer
        corresponds to a band.

        returns: a numpy array with 3 dimensions (width, height, channels)
    '''
    my_image = skimage.io.imread( image_name )
    my_image = preprocess_sentinel_image(my_image)
    return my_image

def preprocess_sentinel_image( sentinel_image ):
    '''
        preprocess_sentinel_image
        string sentinel_image: name of an image file in tiff format

        This function reads a sentinel-2 TIFF file, where each layer
        corresponds to a band. Output is a calibrated image

        returns: a numpy array with 3 dimensions (width, height, channels)
    '''
    my_image = np.array(sentinel_image).astype(np.float32)
    my_image[my_image>10000] = 10000.0
    my_image = my_image/10000.0
    return my_image

