# This file is contributed by Shuen-Jen Chen (based on vgg_model.py)

import ssl

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def generate_resnet_model_advance(classes_len: int):
    """
    Function to create a ResNet50 model pre-trained with custom FC Layers.
    If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
    larger images.
    :param classes_len: The number of classes (labels).
    :return: The ResNet50 model.
    """
    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 1))
    
    # Add convolution and pooling layers
    model = Sequential()
    model.add(img_input)
    for i in range (0, config.CONV_CNT):
        model.add(Conv2D(3, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Generate a ResNet50 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
    model_base = ResNet50(include_top=False, weights='imagenet')
    
    # Start with base model consisting of convolutional layers
    model.add(model_base)

    # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
    model.add(Flatten())
    
    # Possible dropout for regularisation can be added later and experimented with:
    if config.DROPOUT != 0:
        model.add(Dropout(config.DROPOUT, name='Dropout_Regularization_1'))

    # Add fully connected hidden layers.
    model.add(Dense(units=512, activation='relu', kernel_initializer='random_uniform', name='Dense_Intermediate_1'))
    
    model.add(Dense(units=32, activation='relu', kernel_initializer='random_uniform', name='Dense_Intermediate_2'))

    # Final output layer that uses softmax activation function (because the classes are exclusive).
    if classes_len == 2:
        model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform', name='Output'))
    else:
        model.add(Dense(classes_len, kernel_initializer='random_uniform', activation='softmax', name='Output'))

    # Print model details if running in debug mode.
    if config.verbose_mode:
        print(model.summary())

    return model
