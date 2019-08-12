## jigsaw_cnn
# Uses Satellite Multispectral data and CNNs to create land cover maps 

It requires:
- Python 3.7 (conda use is reccomended)
- an input image from Sentinel 2A, **as a 12 channel TIFF file**
- a **mask file** that defines the classes to use and ground truth for a subset of the image

Example use:
- * Create training set:
  ```
  python create_sentinel_dataset.py -i <image_path> \
                                    -d <training_dataset_directory_name>
  ```
  By default, it will create from the image and mask 1,200 random samples for each class, with a kernel size of 17x17 pixels

- Create evaluation set:
  ```
  python create_sentinel_dataset.py -i <image_path> \
                                    -d <evaluation_dataset_directory_name>
  ```
  - By default, it will create a dataset from the image and mask, containing 1,200 random samples for each class, with a kernel size of 17x17 pixels

- Train the network:
  ```
  python jigsaw_cnn.py -d <training_dataset_directory_name> \
                       -m <trained_model_name> \
                       -l <trained_model_labels_filename> \
                       -p <output_picture_name_for_training_graph> \
                       -r -e <number_of_epochs>
  ```
  - Using the dataset, creates a trained model by running the model for 200 epochs. Use .png extension for history graph file

- Evaluate the network:
  ```
  python jigsaw_cnn.py -d <evaluation_dataset_directory_name> \
                       -m <trained_model_name> \
                       -l <trained_model_labels_filename> \
                       -v [-t]
  ```
  - Using the dataset, uses a trained model to evaluate accuracy. *The "-t" flag ensures true randomization of input*
