import numpy as np
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--ground_truth_image", required=True,
                    help="path to ground truth mask (i.e., inital mask file name)")
    ap.add_argument("-p", "--prediction_image", required=False,
                    help="file name of prediction mask (i.e.: second file to compare)")
    args = vars(ap.parse_args())
    ground_truth_image = args["ground_truth_image"]
    prediction_image = args["prediction_image"]

    b01 =  np.load(ground_truth_image)
    a01 = np.load(prediction_image)
    (max_row, max_col) = b01.shape
    results = np.zeros((8,8)).astype(np.int)

    for before_class in range(8):
        k = a01[b01 == before_class]
        for after_class in range(8):
            results[before_class, after_class] = np.sum(k==after_class)
    print('Change matrix')
    print(results)
