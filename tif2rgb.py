import skimage
import numpy as np
from PIL import Image
import argparse

# MAX_COLOR_VALUE = 1024.0
MAX_COLOR_VALUE = 2048.0
# 
# i = skimage.io.imread('data/before.tif')
def tif2rgb( input_file, output_file):
    # i = skimage.io.imread('data/after.tif')
    i = skimage.io.imread( input_file )
    (channels, height, width) = i.shape
    i2 = i[1:4,:,:]
    i4 = np.concatenate((i2[2].reshape(1,height,width), i2[1].reshape(1,height,width), i2[0].reshape(1,height,width)), axis = 0)
    #  if (height > 3000):
      #  i4 = i4[:,1850:2300,3300:3900]
    
    i3=(i4-1)/MAX_COLOR_VALUE
    i3=np.clip(i3, 0.0, 1.0)
    i3 = i3.transpose(1,2,0)
    i3[i3>1] = 1.0
    im2 = Image.fromarray((i3*255).astype(np.uint8))
    # im2.save("before_rgb_2.crop.png")
    # im2.save("before_rgb_2.crop.tif")
    im2.save( output_file )
    return 0

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", required=True,
                    help="path to input multispectral image (i.e., image file name)")

    ap.add_argument("-o", "--output_file", required=True,
                    help="path to output RGB image (i.e., image file name)")

    args = vars(ap.parse_args())
    input_file = args["input_file"]
    output_file = args["output_file"]
    tif2rgb( input_file, output_file )
