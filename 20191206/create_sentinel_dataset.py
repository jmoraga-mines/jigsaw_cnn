# -*- coding: utf-8 -*-
"""
Create Dataset from Sentinel Multispectra TIFFs

Created 2019-06-10
Updated 2019-08-12

@authors: gurbet, jim
"""

import numpy as np
import sentinel_tiff as sf
import cv2
import skimage.io
import os
import argparse

# Default values
SAMPLES_TO_CREATE = 1200
KERNEL_PIXELS = 17
DEFAULT_DIRECTORY = 'dataset'

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multispectral image (i.e., image file name)")
    ap.add_argument("-m", "--mask", required=True,
                    help="file name of mask (it should have a .png or .tif extension)")
    ap.add_argument("-d", "--directory", required=False,
                    help="path to directory where samples will be written to",
                    default = DEFAULT_DIRECTORY)
    ap.add_argument("-s", "--samples", required=False, help="number of samples to create",
                    default = SAMPLES_TO_CREATE)
    ap.add_argument("-k", "--kernel_size", required=False,
                    help="number of pixels per side in the kernel (use an odd number)", default = KERNEL_PIXELS)
    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = parse_arguments( )
    image_name = args["image"]
    mask_name = args["mask"]
    output_directory = args["directory"]
    num_samples = args["samples"]
    kernel_size = args["kernel_size"]
    PADDING = int(kernel_size/2)
    img_b = sf.io.read_image_file( image_name )
    # By Jim: to reduce the size of the input tiff
    # img_b = img_b[:,1200:-500,2700:]
    mask_b = sf.io.read_mask_file( mask_name )
    img_b_scaled = img_b
    # print('# Items with class 1: ', len(mask_b))
    # print('original image:')
    # print( img_b[:3,:4, :4] )
    # print('scaled image:')
    # print( img_b_scaled[:3,:4, :4] )

    img_b_scaled = sf.frame_image(img_b_scaled, PADDING)
    sc = sf.SentinelConvolution(img_b_scaled, kernel_size, kernel_size)

    # Land classes go from 1 to 7
    for land_type_class in range(1,8):
        mask_b_class = np.array(np.where(mask_b==land_type_class)).T
        print('Items in class ', land_type_class, ': ', len(mask_b_class))
        mask_b_scaled = mask_b_class+[PADDING, PADDING]
        choices = np.random.choice( len(mask_b_scaled), num_samples, replace = False)
        path_name = output_directory + '/' + str(land_type_class) + '/'
        os.makedirs( path_name )
        for c in choices:
            my_slice = sc.apply_mask(mask_b_scaled[c][0], mask_b_scaled[c][1])
            my_slice = my_slice.transpose((1, 2, 0))
            file_name = path_name + 'slice_'+str(c)+'.npy'
            # print('saving file:', file_name)
            f = open( file_name, 'wb' )
            np.save(f, my_slice)
