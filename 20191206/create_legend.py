import numpy as np
from PIL import Image
'''
creates legend picture with the colors for each transition from 
one class to another
'''

color_matrix = 48 - np.arange( 49 ).reshape(7,7) 
color_matrix[color_matrix%8 == 0] = 0
color_matrix = (color_matrix * 255./47).astype(np.uint8)
greys = color_matrix[:,3] # we only care about class 'mine & tailings'
legend = np.zeros((40,200))
for i in range(200):
  legend[:,i] = greys[int((7*i)/200)]
image = Image.fromarray(legend.astype(np.uint8))
image.save('legend.png')

