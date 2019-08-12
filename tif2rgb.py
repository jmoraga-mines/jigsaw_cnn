import skimage
import numpy as np
from PIL import Image
import argparse

# This is a number that forces a maximum value
# input will scaled to fit (i.e. defines value for white)
MAX_COLOR_VALUE = 2048.0
def tif2rgb( input_file, output_file):
    i = skimage.io.imread( input_file )
    i = np.asarray( i ).astype( np.float )
    i[i>10000] = 10000 # Standardizes output to eliminate saturated pixels
    (channels, height, width) = i.shape
    i2 = i[1:4,:,:]
    i4 = np.concatenate((i2[2].reshape(1,height,width),
                         i2[1].reshape(1,height,width),
                         i2[0].reshape(1,height,width)), axis = 0)
    i3=(i4-1)/MAX_COLOR_VALUE
    i3=np.clip(i3, 0.0, 1.0)
    i3 = i3.transpose(1,2,0)
    i3[i3>1] = 1.0
    im2 = Image.fromarray((i3*255).astype(np.uint8))
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
