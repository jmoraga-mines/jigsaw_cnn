# -*- coding: utf-8 -*-
"""
Create Dataset from Sentinel Multispectra TIFFs

Created 2019-06-13
Updated 2019-08-12

@authors: gurbet, jim
"""

# import the necessary packages
import argparse
from imutils import paths
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling2D, Input, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
import pickle
import random
import skimage
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
# from sklearn.metrics import roc_curve, auc

matplotlib.use("Agg")
DEFAULT_WEIGHTS_FILE_NAME = 'new_cnn.weights'
DEFAULT_MODEL_FILE_NAME = 'new_cnn.model'
DEFAULT_LABEL_FILE_NAME = 'new_cnn.labels.pickle'

class2code = {'none': 0,
              'mine':1,
              'forest':2,
              'build_up':3,
              'river':4,
              'agricultural':5,
              'clear water':6,
              'grasland':7}

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_set, labels, batch_size=32, dim=(17,17,12),
            n_channels=12, n_classes=8, shuffle=True, augment_data=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_set = data_set
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment_data = augment_data
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_set) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_raw = self.data_set[ID]
            if self.augment_data:
                if np.random.randint(1):
                    image_raw = np.rot90( image_raw, 2 )
                if np.random.randint(1):
                    image_raw = np.fliplr( image_raw )
                if np.random.randint(1):
                    image_raw = np.flipud( image_raw )
            X[i,] = image_raw
            # Store class
            y[i] = self.labels[ID]
        return X, y

def inception_m( input_net, first_layer = None ):
    conv1 = Conv2D(128, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(input_net)
    inception_t1_1x1 = Conv2D(256, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(conv1)
    inception_t1_3x3_reduce = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(conv1)
    inception_t1_3x3 = Conv2D(128, (3,3), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(inception_t1_3x3_reduce)
    inception_t1_5x5_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(conv1)
    inception_t1_5x5 = Conv2D(32, (5,5), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(inception_t1_5x5_reduce)
    inception_t1_7x7_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(conv1)
    inception_t1_7x7 = Conv2D(32, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(inception_t1_7x7_reduce)
    inception_t1_pool = MaxPooling2D(pool_size=(3,3), strides = (1,1), padding='same')(conv1)
    inception_t1_pool_proj = Conv2D(32, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(inception_t1_pool)
    if first_layer is None:
        inception_t1_output = Concatenate(axis = -1)([inception_t1_1x1, inception_t1_3x3, inception_t1_5x5,
                                                      inception_t1_7x7, inception_t1_pool_proj])
    else:
        inception_t1_first = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(first_layer)
        inception_t1_output = Concatenate(axis = -1)([inception_t1_first, inception_t1_1x1, inception_t1_3x3,
                                                      inception_t1_5x5, inception_t1_7x7, inception_t1_pool_proj])
    return inception_t1_output

def inception_m_end( input_net, num_classes = 7, first_layer = None ):
    avg_pooling = AveragePooling2D(pool_size=(3,3), strides=(1,1), name='avg_pooling')(input_net)
    flat = Flatten()(avg_pooling)
    flat = Dense(16, kernel_regularizer=l2(0.0002))(flat)
    flat = Dropout(0.4)(flat)
    if first_layer is not None:
        input_pixel = Flatten()(first_layer)
        input_pixel = Dense(16, kernel_regularizer=l2(0.0002))(input_pixel)
        input_pixel = Dropout(0.2)(input_pixel)
        input_pixel = Dense(16, kernel_regularizer=l2(0.0002))(input_pixel)
        input_pixel = Dropout(0.2)(input_pixel)
        flat = Concatenate(axis = -1)([input_pixel, flat])
    flat = Dense(32, kernel_regularizer=l2(0.0002))(flat)
    avg_pooling = Dropout(0.4)(flat)
    loss3_classifier = Dense(num_classes, kernel_regularizer=l2(0.0002))(avg_pooling)
    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)
    return loss3_classifier_act

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", required=False,
                    help="Augment images by flippipng horizontally, vertically and diagonally",
                    dest='augment', action = 'store_true', default = False)
    ap.add_argument("-b", "--batch_size", required=False, help="Defines batch size", default = 32, type=int)
    ap.add_argument("-c", "--channels", required=False, help='Number of channels in each image',
                    default=12, type=int)
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-e", "--epochs", required=False, type=int,
                    help="Number of epochs to train)", default=500)
    ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-n", "--num_classes", required=False, help='Number of classes',
                    default=7, type=int)
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    ap.add_argument("-r", "--reset", required=False,
                    help="Don't load setup files, train from scratch",
                    dest='reset', action = 'store_true', default = False)
    ap.add_argument("-t", "--true_random", required=False,
                    help="Ensure true random shuffling of training and test sets",
                    dest='true_random', action = 'store_true', default = False)
    ap.add_argument("-v", "--validate", required=False,
                    help="Don't train, only validate with random images from dataset",
                    dest='validate', action = 'store_true', default = False)
    ap.add_argument("-w", "--weights", required=False, help="path to input or output model weights",
                    default = None)
    args = vars(ap.parse_args())
    augment_data = args["augment"]
    batch_size = args["batch_size"]
    image_channels = args["channels"]
    dataset_path = args["dataset"]
    num_epochs = args["epochs"]
    label_file = args["labelbin"]
    model_file = args["model"]
    num_classes = args["num_classes"]
    plot_file = args["plot"]
    reset_model = args["reset"]
    true_random = args["true_random"]
    validate_only = args["validate"]
    weights_file = args["weights"]
    if reset_model:
        print("[INFO] Reset model")
        model_exist = False
        weights_exist = False
    else:
        print("[INFO] Don't reset model")
        # Ensures model file exists and is really a file
        if model_file is not None:
            assert path.exists(model_file), 'weights file {} does not exist'.format(model_file)
            assert path.isfile(model_file), 'weights path {} is not a file'.format(model_file)
            model_exist = True
        else:
            model_file = DEFAULT_MODEL_FILE_NAME
            model_exist = False
        # Ensures weights file exists and is really a file
        if weights_file is not None:
            assert path.exists(weights_file), 'weights file {} does not exist'.format(weights_file)
            assert path.isfile(weights_file), 'weights path {} is not a file'.format(weights_file)
            weights_exist = True
        else:
            weights_file = DEFAULT_WEIGHTS_FILE_NAME
            weights_exist = False
    EPOCHS = num_epochs
    INIT_LR = 1e-3
    BS = batch_size
    KERNEL_PIXELS = 17
    CHANNELS = image_channels
    IMAGE_DIMS = (KERNEL_PIXELS, KERNEL_PIXELS, CHANNELS)
    BATCH_DIMS = (None, KERNEL_PIXELS, KERNEL_PIXELS, CHANNELS)
    # initialize the data and labels
    data = []
    labels = []
    '''
    Creates the network
    '''
    if (reset_model or not model_exist):
        print('[INFO] Building model from scratch...')
        my_input = Input( shape=IMAGE_DIMS, batch_shape=BATCH_DIMS )
        
        # One inception modules 
        inception_01 = inception_m( my_input )
        # Attaches end to inception modules, returns class within num_classes
        loss3_classifier_act = inception_m_end( inception_01, num_classes = num_classes, first_layer = my_input )
        
        # Builds model
        model3 = Model( inputs = my_input, outputs = [loss3_classifier_act] )
        model3.summary()
    else:
        # Builds model
        print('[INFO] Loading model from file...')
        model3 = load_model( model_file )
        model3.summary()
    if (not reset_model and (weights_exist and not model_exist)):
        model3.load_weights(weights_file)
    ## grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_files(dataset_path)))
    print('Number of images:', len(imagePaths))
    # Ensure 'random' numbers are not too random to compare networks
    if (not true_random):
        random.seed(42)
    random.shuffle(imagePaths)
    #
    ## loop over the input images
    img_count = 0
    for imagePath in imagePaths:
        # Reads image file from dataset
        image = np.load(imagePath)
        # Our Model uses (width, height, depth )
        data.append(image)
        # Gets label from subdirectory name and stores it
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    
    print('Read images:', len(data))
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype=np.float)
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1024.0)))
     
    # binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # partition the data into training and testing splits using 50% of
    # the data for training and the remaining 50% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.5, random_state=42)
    params = {'dim':(17,17), 'batch_size': BS, 'n_classes': 7, 'n_channels': 12,
            'shuffle': True, 'augment_data': augment_data}
    # construct the image generator for data augmentation
    my_batch_gen = DataGenerator(trainX, trainY, **params)
    print('creating generator with trainX, trainY of shapes: (%s, %s)'%(trainX.shape, trainY.shape))
    ## initialize the model
    print("[INFO] compiling model...")
    print('SmallInception: (depth, width, height, classes) = (%s, %s, %s, %s)' % (IMAGE_DIMS[0], IMAGE_DIMS[1], 
           IMAGE_DIMS[2], len(lb.classes_)))
    if (validate_only):
        print("[INFO] Skipping training...")
        print("[INFO] Validate-only model:", validate_only)
        pass
    else:
        print("[INFO] Training...")
        print("[INFO] Reset model:", reset_model)
        print("[INFO] Validate-only model:", validate_only)
        opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model3.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # define the network's early stopping
        print("[INFO] define early stop and auto save for network...")
        auto_save = ModelCheckpoint(model_file, monitor = 'val_acc', verbose = 0,
                                    save_best_only = True, save_weights_only=False,
                                    mode='auto', period=10)
        # can use validation set loss or accuracy to stop early
        # early_stop = EarlyStopping( monitor = 'val_acc', mode='max', baseline=0.97)
        early_stop = EarlyStopping( monitor = 'val_loss', mode='min', verbose=1, patience=50 )
        # train the network
        print("[INFO] training network...")
        # Train the model
        H = model3.fit_generator(
            generator = my_batch_gen,
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // BS,
            callbacks=[early_stop, auto_save],
            epochs=EPOCHS, verbose=1)
        # save the model to disk
        print("[INFO] serializing network...")
        model3.save( model_file )
        # save the label binarizer to disk
        print("[INFO] serializing label binarizer...")
        f = open(label_file, "wb")
        f.write(pickle.dumps(lb))
        f.close()
    testY = lb.inverse_transform( testY ).astype(np.int64)
    print('[INFO] Predicting ...')
    pre_y2 = model3.predict(testX, verbose = 1)
    pre_y2 = pre_y2.argmax(axis=-1)+1
    acc2 = accuracy_score(testY, pre_y2)
    print('Accuracy on test set: {0:.3f}'.format(acc2))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, pre_y2))
    print()
    print("Classification Report")
    print(classification_report(testY, pre_y2))
    if (not validate_only):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig(plot_file)
