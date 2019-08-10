import skimage
import numpy as np
from PIL import Image

MAX_COLOR_VALUE = 1024.0
MAX_COLOR_VALUE = 2048.0

i = skimage.io.imread('data/after.tif')
(channels, height, width) = i.shape
i2 = i[1:4,:,:]
i4 = np.concatenate((i2[2].reshape(1,3182,4571), i2[1].reshape(1,3182,4571), i2[0].reshape(1,3182,4571)), axis = 0)

i3=(i4-1)/MAX_COLOR_VALUE
i3 = i3.transpose(1,2,0)
i3[i3>1] = 1.0
im2 = Image.fromarray((i3*255).astype(np.uint8))
im2.save("after_rgb_2.png")

