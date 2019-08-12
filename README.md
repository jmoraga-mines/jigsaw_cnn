# jigsaw_cnn
Uses Satellite Multispectral data and CNNs to create land cover maps 

It requires:
- Python 3.7 (conda use is reccomended)
- an input image from Sentinel 2A, as a 12 channel TIFF file
- a mask file that defines the classes to use and ground truth for a subset of the image

Example use:
- Create training set:
  - python create_sentinel_dataset.py -i <image path> -d <training dataset directory name>
  - By default, it will create from the image and mask 1,200 random samples for each class, with a kerle size of 17x17 pixels

- Create evaluation set:
  - python create_sentinel_dataset.py -i <image path> -d <evaluation dataset directory name>
  - By default, it will create a dataset from the image and mask, containing 1,200 random samples for each class, with a kernel size of 17x17 pixels

- train the network:
  - python jigsaw_cnn.py -d <training dataset directory name> -m <trained model name> -l <traned model labels filename> \
                          -p <output picture name for training graph [use.png extension]> \
                          -r -e 200
  - Using the dataset, creates a trained model by running the model for 200 epochs

- evaluate the network:
  - python jigsaw_cnn.py -d <evaluation dataset directory name> -m <trained model name> -l <traned model labels filename> \
                          -v [-t]
  - Using the dataset, uses a trained model to evaluate accuracy. the "-t" flag ensures true randomization of input
