import numpy as np
import argparse
from PIL import Image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--ground_truth_image", required=True,
                    help="path to ground truth mask (i.e., inital mask file name)")
    ap.add_argument("-p", "--prediction_image", required=True,
                    help="file name of prediction mask (i.e.: second file to compare)")
    ap.add_argument("-c", "--change_map", required=True,
                    help="file name of change map (i.e.: file with extension .PNG or .JPG)")
    args = vars(ap.parse_args())
    ground_truth_image = args["ground_truth_image"]
    prediction_image = args["prediction_image"]
    change_map = args["change_map"]

    b01 =  np.load(ground_truth_image).astype(np.int)-1
    a01 = np.load(prediction_image).astype(np.int)-1
    color_matrix = 48 - np.arange( 49 ).reshape(7,7)
    color_matrix[color_matrix%8 == 0] = 0
    # color_matrix = color_matrix.T * 255./47
    color_matrix = color_matrix * 255./47
    c01 = np.zeros_like( b01 )
    c01 = color_matrix[b01,a01]
    image = Image.fromarray(c01.astype(np.uint8))
    # image.save('change_map.01.jpg', quality = 95)
    # image.save(change_map+'.jpg', quality = 95)
    image.save(change_map+'.png' )
    (max_row, max_col) = b01.shape
    results = np.zeros((7,7)).astype(np.int)

    for before_class in range(7):
        k = a01[b01 == before_class]
        for after_class in range(7):
            results[before_class, after_class] = np.sum(k==after_class)
    print('Change matrix')
    print(results)
    non_mine = np.where(a01 != 3) # Just reds
    a01[non_mine] = b01[non_mine]
    c01 = np.zeros_like( b01 )
    c01 = color_matrix[b01,a01]
    image = Image.fromarray(c01.astype(np.uint8))
    # image.save('change_map.01.jpg', quality = 95)
    # image.save(change_map+'.mine.jpg', quality = 95)
    image.save(change_map+'.mine.png' )
    
    results = np.zeros((7,7)).astype(np.int)

    for before_class in range(7):
        k = a01[b01 == before_class]
        for after_class in range(7):
            results[before_class, after_class] = np.sum(k==after_class)
    print('Change matrix 2')
    print(results)
