# import the necessary packages
import cv2
import argparse
from imutils import paths
import keras
from keras.callbacks import EarlyStopping
from keras.layers import AveragePooling2D, Input, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.regularizers import l2

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sentinel_tiff as st
import skimage
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

matplotlib.use("Agg")

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_set, labels, batch_size=32, dim=(17,17,12),
            n_channels=12, n_classes=8, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_set = data_set
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_set) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_set_temp = [self.data_set[k] for k in indexes]
        label_temp = [self.labels[k] for k in indexes]

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
            X[i,] = self.data_set[ID]
            # Store class
            y[i] = self.labels[ID]
        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)

def preprocess_input(x):
    x = np.divide(x, 255.0)   # Normalizes to range 0.0 to 1.0
    # x = np.subtract(x, 0.5) # Normalizes to range -0.5 to 0.5
    # x = np.multiply(x, 2.0) # Normalizes to range -1.0 to 1.0
    return x

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
        # print('input_net:', input_net)
        # input_shape = tuple(first_layer.shape[1:])
        # np_ones = np.ones(input_shape).astype(np.int)
        # input_pixel = tf.constant(np_ones, dtype = np.float32)
        # input_pixel = tf.expand_dims(input_pixel, 0)
        # input_zeros = tf.zeros_like(first_layer)
        # input_pixel = tf.math.multiply(first_layer, input_zeros)
        # input_pixel = tf.math.add(input_zeros, input_pixel)
        # print('input_net.shape:', input_net.shape)
        # input_pixel = input_zeros+input_pixel
        # print('input_pixel:', input_pixel)
        # print('first_layer:', first_layer)
        # pixel_conv = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(input_pixel)
        inception_t1_first = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.0002))(first_layer)
        # inception_t1_first = pixel_conv
        inception_t1_output = Concatenate(axis = -1)([inception_t1_first, inception_t1_1x1, inception_t1_3x3,
                                                      inception_t1_5x5, inception_t1_7x7, inception_t1_pool_proj])
    return inception_t1_output

def inception_m_end( input_net, num_classes = 7, first_layer = None ):
    avg_pooling = AveragePooling2D(pool_size=(3,3), strides=(1,1), name='avg_pooling')(input_net)
    flat = Flatten()(avg_pooling)
    flat = Dense(16, kernel_regularizer=l2(0.0002))(flat)
    flat = Dropout(0.4)(flat)
    # flat = Flatten()(input_net)
    if first_layer is not None:
        # input_pixel = first_layer[:,8,8,:]
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


'''
Creates the network
'''
my_input = Input( shape=( 17, 17, 12 ), batch_shape=(None, 17, 17, 12) )

# Two inception modules 
inception_01 = inception_m( my_input )
# inception_01 = Dropout(0.4)(inception_01)
# inception_01 = inception_m( inception_01, my_input )
# inception_01 = Dropout(0.4)(inception_01)
# inception_01 = inception_m( inception_01, my_input )
# inception_01 = Dropout(0.4)(inception_01)
# Attaches end to inception modules, returns class within num_classes
loss3_classifier_act = inception_m_end( inception_01, num_classes = 7, first_layer = my_input )
# loss3_classifier_act = inception_m_end( inception_02, num_classes = 7 )

# Builds model
model3 = Model( inputs = my_input, outputs = [loss3_classifier_act] )
model3.summary()



EPOCHS = 2500
INIT_LR = 1e-3
BS = 32
KERNEL_PIXELS = 17
CHANNELS = 12
IMAGE_DIMS = (KERNEL_PIXELS, KERNEL_PIXELS, CHANNELS)
# initialize the data and labels
data = []
labels = []
## grab the image paths and randomly shuffle them
print("[INFO] loading images...")
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
imagePaths = sorted(list(paths.list_images('dataset17')))
random.seed(42)
random.shuffle(imagePaths)
#
## loop over the input images
MAX_IMAGES = 99999
img_count = 0
for imagePath in imagePaths:
    if img_count >= MAX_IMAGES: break
    # load the image, pre-process it, and store it in the data list
    # image = cv2.imread(imagePath)
    image = skimage.io.imread(imagePath)
    # image = image[:3,:,:]  # By Jim: reduce dimension of tiff to 3 channels
    # GoogLeNet uses (depth, width, height)
    # Our Model uses (width, height, depth )
    image = image.transpose((1,2,0))
    # image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # image = img_to_array(image)
    data.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

## scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)
#print("[INFO] data matrix: {:.2f}MB".format(
#    data.nbytes / (1024 * 1000.0)))
#
## binarize the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

#print("data shape: ", data.shape)


# scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
data = np.array(data, dtype="float")
data = np.divide(data, 255.0)
#data = preprocess_input( data )
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1024.0)))
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

params = {'dim':(17,17), 'batch_size': BS, 'n_classes': 7, 'n_channels': 12,
        'shuffle': True}

# construct the image generator for data augmentation
my_batch_gen = DataGenerator(trainX, trainY, **params)
print('creating generator with trainX, trainY of shapes: (%s, %s)'%(trainX.shape, trainY.shape))


## initialize the model
print("[INFO] compiling model...")
print('SmallInception: (depth, width, height, classes) = (%s, %s, %s, %s)' % (IMAGE_DIMS[0], IMAGE_DIMS[1], 
       IMAGE_DIMS[2], len(lb.classes_)))
#model = SmallInception.build(width=IMAGE_DIMS[0], height=IMAGE_DIMS[1],
#    depth=IMAGE_DIMS[2], classes=len(lb.classes_))

opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model3.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# define the network's early stopping
print("[INFO] define early stop for network...")
early_stop = EarlyStopping( patience=3, monitor = 'val_acc')

# train the network
print("[INFO] training network...")
H = model3.fit_generator(
    generator = my_batch_gen,
    # aug.flow(trainX, trainY, batch_size=BS),
    #(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model3.save('Inception_mx2_learning.model')
print("[INFO] saving network weights...")
weight_f = 'inception.mx2.weights.h5'
model3.save_weights( weight_f )

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('labelbin.mx2.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
# print('history type: ', type(H.history))
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('inception_mx2_plot.png')

testY = lb.inverse_transform( testY ).astype(np.int64)
#1AA
print('[INFO] Predicting ...')
pre_y2 = model3.predict(testX, verbose = 1)
pre_y2 = pre_y2.argmax(axis=-1)+1
# print('Prediction output type: ', type(pre_y2))
# print('Type testY[0]', type(testY[0]))
# print('Type pre_y2[0]', type(pre_y2[0]))
# print('Length testY', len(testY))
# print('Length pre_y2', len(pre_y2))
# print('Value testY[0]', (testY[0]))
# print('Value pre_y2[0]', (pre_y2[0]))
acc2 = accuracy_score(testY, pre_y2)
print('Accuracy on test set: {0:.3f}'.format(acc2))


#gradient boosting 

print("Confusion Matrix:")
print(confusion_matrix(testY, pre_y2))
print()
print("Classification Report")
print(classification_report(testY, pre_y2))

