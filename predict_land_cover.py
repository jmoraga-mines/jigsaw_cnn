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
from sklearn.preprocessing import LabelBinarizer
import pickle


# import user libraries
import sentinel_tiff as sf

def calc_percentage( img_mask, img_pred, img_class ):
  '''
    It gets the postion of all pixels in img_pred where img_mask is equal to img_class
    Returns the percentage of positions where the class is the same for prediction and mask
  '''
  return np.sum(img_pred[np.where(img_mask==img_class)]==img_class)/np.sum(img_mask==img_class)

def to_rgb( s_img):
  s_img = s_img.astype(np.uint8)
  Red = (s_img & 1)*255
  Green = ((s_img & 2)/2)*255
  Blue = ((s_img & 4)/4)*255
  rgb = np.concatenate((Red.reshape(3182,4571,1),Green.reshape(3182,4571,1),Blue.reshape(3182,4571,1)), axis=2).astype(np.uint8)
  # return (Red, Green, Blue)
  return rgb


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
    
    f = open(label_file, 'rb')
    lb_pickle = f.read()
    lb = pickle.loads( lb_pickle )
    sentinel_image = sf.io.read_image_file( image_name )
    print('Image dimensions: ', sentinel_image.shape)
    # scale = 1/np.amax(sentinel_image, axis = (1,2))
    # By Jim: to reduce the size of the input tiff
    # img_b = img_b[:,1200:-500,2700:]
    # sentinel_image = sentinel_image[:,2000:-500,2700:-200]
    # sentinel_image = sentinel_image[:,1400:-500,2000:]
    # By Jim: to reduce the size of the input tiff - Only runoff area
    # sentinel_image = sentinel_image[:,1850:2300,3300:4100]
    (img_c, img_x, img_y) = sentinel_image.shape
    KERNEL_PIXELS = 17
    PADDING = int(KERNEL_PIXELS/2)
    # Scale image based on each channel's maximum
    # sentinel_image_scaled = scale[:, np.newaxis, np.newaxis]+np.zeros_like( sentinel_image )
    # Turn each value to a number between, and including, 0-255
    # sentinel_image_scaled = (sentinel_image*sentinel_image_scaled)*255
    sentinel_image_scaled = sentinel_image
    sentinel_image_scaled = sf.frame_image(sentinel_image_scaled, PADDING)
    sentinel_convolution = sf.SentinelConvolution(sentinel_image_scaled, KERNEL_PIXELS, KERNEL_PIXELS)
    model3 = load_model( model_file )
    '''
    pixel = sentinel_convolutionc.apply_mask( 3000, 3000 )
    pre_y = model3.predict( np.asarray([pixel.transpose(1,2,0)]), verbose = 1)
    pre_y2 = pre_y.argmax(axis=-1)+1
    Red = (pre_y2 & 4) >> 2
    Green = (pre_y2 & 2) >> 1
    Blue = (pre_y2 & 1)
    rgb = np.zeros( (img_x, img_y, 3) )
    testX = np.zeros((img_y, 17, 17, 12))
    for y in range(img_y):
        testX[y] = (sentinel_convolution.apply_mask( 3000, y+PADDING)).transpose(1,2,0)
    pre_y2 = model3.predict( testX, verbose = 1)
    pre_y2 = pre_y2.argmax(axis=-1)+1
    '''
    testX = np.zeros((img_y, 17, 17, 12))
    flat_img = np.zeros((img_x, img_y))
    for i in range(img_x):
      if (i%100 == 0):
        print('Processing row: ', i)
      for j in range(img_y):
        testX[j] = (sentinel_convolution.apply_mask( PADDING+i, PADDING+j )).transpose(1,2,0)
      testX=testX.astype(np.float)
      # testX=(testX)/255.0
      pre_y2 = model3.predict( testX, verbose = 0 )
      pre_y2 = pre_y2.argmax(axis=-1)+1
      flat_img[i] = pre_y2
    print('saving raw data')
    np.save(mask_name+'.flat.npy', flat_img)
    flat_img = flat_img.astype(np.uint8)
    Red = (flat_img & 4)/4
    Green = (flat_img & 2)/2
    Blue = (flat_img & 1)
    Red = Red.reshape((img_x, img_y, 1))
    Green = Green.reshape((img_x, img_y, 1))
    Blue = Blue.reshape((img_x, img_y, 1))
    flat_img = np.concatenate( (Red, Green, Blue), axis = 2)
    flat_img = Image.fromarray((flat_img*255).astype(np.uint8))
    print('saving PNG data')
    flat_img.save(mask_name+'.png')
    # We can save a lossless TIF, but it is redundant. use .npy instead
    # print('saving TIF data to:', mask_name)
    # flat_img.save( mask_name+'.tif' )


