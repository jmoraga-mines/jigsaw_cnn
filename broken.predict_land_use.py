# -*- coding: utf-8 -*-
"""
predict_land_cover
Created on Mon Jul 29 14:57:37 2019

@author: Jim, Gurbet
"""

# import the necessary packages
import argparse
from imutils import paths
import numpy as np
import cv2
import skimage.io
import os
from PIL import Image
import keras
from keras.models import load_model

# import user libraries
import sentinel_tiff as sf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multispectral image (i.e., image file name)")
    ap.add_argument("-o", "--output_mask", required=False,
                    help="file name of output mask (it should have a .png or .tif extension)")
    ap.add_argument("-l", "--labelbin", required=True, help="path to input label binarizer")
    ap.add_argument("-m", "--model", required=True, help="path to input model")
    args = vars(ap.parse_args())
    image_name = args["image"]
    mask_name = args["output_mask"]
    label_file = args["labelbin"]
    model_file = args["model"]
    
    sentinel_image = sf.io.read_image_file( image_name )
    #sentinel_image = sentinel_image[:,1400:-500,2700:-200]
    sentinel_image = sentinel_image[:,1600:-800,2200:]
    print('Image dimensions: ', sentinel_image.shape)
    (img_c, img_x, img_y) = sentinel_image.shape
    # By Jim: to reduce the size of the input tiff
    # img_b = img_b[:,1200:-500,2700:]
    scale = 1/np.amax(sentinel_image, axis = (1,2))
    sentinel_image_scaled = scale[:, np.newaxis, np.newaxis]+np.zeros_like( sentinel_image )
    sentinel_image_scaled = (sentinel_image*sentinel_image_scaled)*255
    KERNEL_PIXELS = 17
    PADDING = int(KERNEL_PIXELS/2)
    sentinel_image_scaled = sf.frame_image(sentinel_image_scaled, PADDING)
    sentinel_convolution = sf.SentinelConvolution(sentinel_image_scaled, KERNEL_PIXELS, KERNEL_PIXELS)
    flat_img = np.zeros((img_x, img_y))
    test_x2 = np.zeros((img_x*img_y, 17, 17, 12))
    for i in range(img_x-PADDING):
      if (i%100 == 0):
        print('Processing row: ', i)
      for j in range(img_y):
        try:
          test_x2[j+i*img_y] = (sentinel_convolution.apply_mask( PADDING+i, PADDING+j )).transpose(1,2,0)
        except:
          print('Error in row: ', i, ' ; column: ', j)
    print('Data preprocessed, ready to predict')
    print('Loading Model...')
    model3 = load_model( model_file )
    print('Model Loaded, predicting ...')
    pre_y2 = model3.predict( test_x2, verbose = 1)
    pre_y2 = pre_y2.argmax(axis=-1)+1
    flat_img = pre_y2.reshape(img_x, img_y,1)
    Blue = (flat_img.astype(np.uint8) & 4)/4
    Red = (flat_img.astype(np.uint8) & 2)/2
    Green = (flat_img.astype(np.uint8) & 1)
    print('saving raw data')
    np.save('flat_img.npy', flat_img)
    Red = Red.reshape((img_x, img_y, 1))
    Green = Green.reshape((img_x, img_y, 1))
    Blue = Blue.reshape((img_x, img_y, 1))
    flat_img = np.concatenate( (Red, Green, Blue), axis = 2)
    flat_img = Image.fromarray((flat_img*255).astype(np.uint8))
    print('saving PNG data')
    flat_img.save('land_cover_rgb.png')
    print('saving TIF data to:', mask_name)
    flat_img.save( mask_name )


