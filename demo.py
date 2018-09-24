import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.datasets import cifar10

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# ----------------------------------------------------------------
# load and normalize the dataset and learn about its shapes.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape)

# Normalize image vectors
X_train = X_train/255.
X_test = X_test/255.

# Reshape
Y_train = y_train
Y_test = y_test

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


# ----------------------------------------------------------------
# GRADED FUNCTION: Cifar10Model

def Cifar10Model(input_shape):
    """
    Implementation of the Cifar10Model.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    x_input = Input(input_shape)

    x = ZeroPadding2D((3, 3))(x_input)
    x = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(x)
    x = BatchNormalization(axis=3, name='btn0')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), name='max_pool')(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', name='fc')(x)

    model = Model(inputs=x_input, outputs=x, name='cifar10Model')

    return model


# ----------------------------------------------------------------
# create the model.
cifar10Model = Cifar10Model(X_train.shape[1:])

# compile the model.
cifar10Model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train the model.
cifar10Model.fit(x=X_train, y=Y_train, epochs=40, batch_size=16)

