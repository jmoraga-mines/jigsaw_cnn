# -*- coding: utf-8 -*-
"""
Create Dataset from Sentinel Multispectra TIFFs

Created 2019-06-10
Updated 209-07-29

@authors: gurbet, jim
"""

import argparse
import numpy as np
import sentinel_tiff as sf
import cv2
import skimage.io
import os


SAMPLES_TO_CREATE = 1200
KERNEL_PIXELS = 17








if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multispectral image (i.e., image file name)")
    ap.add_argument("-m", "--mask", required=True,
                    help="file name of mask (it should have a .png or .tif extension)")
    ap.add_argument("-d", "--directory", required=True, help="path to directory where samples will be written to")
    ap.add_argument("-s", "--samples", required=False, help="number of samples to create", default = SAMPLES_TO_CREATE)
    ap.add_argument("-k", "--kernel_size", required=False, help="number of pixels per side in the kernel (use an odd number)", default = KERNEL_PIXELS)
    args = vars(ap.parse_args())
    image_name = args["image"]
    mask_name = args["mask"]
    output_directory = args["directory"]
    num_samples = args["samples"]
    kernel_size = args["kernel_size"]


    # img_b = sf.io.read_image_file('data/before.tif')
    img_b = sf.io.read_image_file('data/before.tif')
    # By Jim: to reduce the size of the input tiff
    # img_b = img_b[:,1200:-500,2700:]
    mask_b = sf.io.read_mask_file('seven_classes_before2.tif')
    print('# Items with class 1: ', len(mask_b))

    scale = 1/np.amax(img_b, axis = (1,2))
    img_b_scaled = scale[:, np.newaxis, np.newaxis]+np.zeros_like( img_b )
    img_b_scaled = (img_b*img_b_scaled)*255
    print('original image:')
    print( img_b[:3,:4, :4] )
    print('scaled image:')
    print( img_b_scaled[:3,:4, :4] )
    PADDING = int(KERNEL_PIXELS/2)

    img_b_scaled = sf.frame_image(img_b_scaled, PADDING)
    sc = sf.SentinelConvolution(img_b_scaled, KERNEL_PIXELS, KERNEL_PIXELS)

    for land_type_class in range(1,8):
        mask_b_class = np.array(np.where(mask_b==land_type_class)).T
        print('Items in class ', land_type_class, ': ', len(mask_b_class))
        mask_b_scaled = mask_b_class+[PADDING, PADDING]
        choices = np.random.choice( len(mask_b_scaled), SAMPLES_TO_CREATE, replace = False)
        path_name = 'dataset/'+str(land_type_class)
        os.makedirs( path_name )
        for c in choices:
            my_slice = sc.apply_mask(mask_b_scaled[c][0], mask_b_scaled[c][1])
            '''
            my_slice = my_slice.transpose((1, 2, 0))
            my_slice = my_slice[:,:,:3] # only 3 channels (RGB)
            '''
            file_name = 'dataset/'+str(land_type_class)+'/slice_'+str(c)+'.tif'
            print('saving file:', file_name)
            skimage.io.imsave(file_name, my_slice)
            # cv2.imwrite(file_name, my_slice)
