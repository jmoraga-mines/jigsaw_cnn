# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:59:22 2019

@author: gurbe
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dropout
#from keras.layers.core import Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from googlenet_custom_layers import PoolHelper,LRN
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
import sentinel_tiff as st
import skimage
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc

class SmallInception:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        inputShape = (height, width, depth) # ???, kernel_size = (3, 3))
        chanDim = -1

        print('(height, width, depth): ', inputShape)
        print('ChanDim: ', chanDim)
        '''
        # CONV => RELU => POOL
        kernel_size = (3,3)
        model.add(Conv2D(32, kernel_size, padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size = (3, 3)))
        model.add(Dropout(0.25))
        '''
        # creates Inception v1 (Szegedy, 2015)
        input = Input(shape=inputShape)
        # conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input)
        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(input)
        pool1_helper = PoolHelper()(conv1_zero_pad)
        pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1/3x3_s2')(pool1_helper)
        pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)
        input = pool1_norm1

        inception_1a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_1a/1x1', 
                                  kernel_regularizer=l2(0.0002), input_shape=inputShape)(input)
        inception_1a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_1a/3x3_reduce', kernel_regularizer=l2(0.0002))(input)
        inception_1a_3x3 = Conv2D(128, (3,3), padding='same', activation='relu', name='inception_1a/3x3', kernel_regularizer=l2(0.0002))(inception_1a_3x3_reduce)
        inception_1a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_1a/5x5_reduce', kernel_regularizer=l2(0.0002))(input)
        inception_1a_5x5 = Conv2D(32, (5,5), padding='same', activation='relu', name='inception_1a/5x5', kernel_regularizer=l2(0.0002))(inception_1a_5x5_reduce)
        inception_1a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_1a/pool')(input)
        inception_1a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_1a/pool_proj', kernel_regularizer=l2(0.0002))(inception_1a_pool)
        inception_1a_output = Concatenate(axis=1, name='inception_1a/output')([inception_1a_1x1,inception_1a_3x3,inception_1a_5x5,inception_1a_pool_proj])
    
        #inception_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/1x1', kernel_regularizer=l2(0.0002))(inception_1a_output)

        loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1/ave_pool')(inception_1a_output)
        loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
        loss1_drop_fc = Dropout(0.7)(loss1_fc)
        loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)


        pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5/7x7_s2')(inception_1a_output)
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
        loss3_classifier = Dense(1000, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        googlenet = Model(inputs=input, outputs=[loss1_classifier_act, loss3_classifier_act])



        # first (and only) set of FC => RELU layers
        googlenet = Sequential(googlenet)

        googlenet.add(Flatten())
        googlenet.add(Dense(256))
        googlenet.add(Activation("relu"))
        googlenet.add(BatchNormalization())
        googlenet.add(Dropout(0.5))
        # softmax classifier
        googlenet.add(Dense(classes))
        googlenet.add(Activation("softmax"))
        # return the constructed network architecture
        return googlenet

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 500
INIT_LR = 1e-3
BS = 32
# IMAGE_DIMS = (96, 96, 3)
KERNEL_PIXELS = 17
IMAGE_DIMS = (12, KERNEL_PIXELS, KERNEL_PIXELS)
# initialize the data and labels
data = []
labels = []
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    # image = cv2.imread(imagePath)
    image = skimage.io.imread(imagePath)
    # image = image[:3,:,:]  # By Jim: reduce dimension of tiff to 3 channels
    # GoogLeNet uses (depth, width, height)
    # image = image.transpose((1,2,0))
    # image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # image = img_to_array(image)
    data.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=90, vertical_flip=True,
    horizontal_flip=True, fill_mode="reflect")


# initialize the model
print("[INFO] compiling model...")
print('SmallInception: (depth, width, height, classes) = (%s, %s, %s, %s)' % (IMAGE_DIMS[0], IMAGE_DIMS[1], 
    IMAGE_DIMS[2], len(lb.classes_)))
model = SmallInception.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[2],
    depth=IMAGE_DIMS[0], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# define the network's early stopping
print("[INFO] define ear;y stop for network...")
early_stop = EarlyStopping( patience=3, monitor = 'val_acc')

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    #(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()


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
plt.savefig(args["plot"])

prediction = model.predict( testX )

acc = accuracy_score(testY, prediction)
print('Accuracy: ', acc)
print()
print('Classification Report')
print(classification_report(testY, prediction))
print()
print('Confusion Matrix:')
print(confusion_matrix(testY, prediction))
print()

