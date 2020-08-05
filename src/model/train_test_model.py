# This file is contributed by Adam Jaamour, and Ashay Patel
import json
import numpy as np
import pandas as pd

from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

import config
from data_visualisation.output import plot_training_results


def train_network(classes_len: int, model, train_x, train_y, val_x, val_y, batch_s, epochs1, epochs2):
    """
    Function to train network in two steps:
        * Train network with initial VGG base layers frozen
        * Unfreeze all layers and retrain with smaller learning rate
    :param model: CNN model
    :param train_x: training input
    :param train_y: training outputs
    :param val_x: validation inputs
    :param val_y: validation outputs
    :param batch_s: batch size
    :param epochs1: epoch count for initial training
    :param epochs2: epoch count for training all layers unfrozen
    :return: trained network
    """
    # Freeze VGG19 pre-trained layers.
    if config.model.startswith('basic'):
        if config.model == 'basic':
            model.layers[0].trainable = False
            print ('Freeze layer\'s name: ', model.layers[0].name)
        else:
            model.layers[1].trainable = False
            print ('Freeze layer\'s name: ', model.layers[1].name)
        
    else:
        if config.CONV_CNT == 0:
            print ("Need to set CONV_CNT > 0 for advance model")
        elif config.model == 'advance':
            model.layers[config.CONV_CNT*2].trainable = False
            print ('Freeze layer\'s name: ', model.layers[config.CONV_CNT*2].name)
        else:
            model.layers[config.CONV_CNT*2+1].trainable = False
            print ('Freeze layer\'s name: ', model.layers[config.CONV_CNT*2+1].name)

    # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).

    if config.dataset == "mini-MIAS":
        if len(config.CLASS_TYPE.split('-')) == 2:
            if config.CLASS_WEIGHT != 'x':
                if config.CLASS_WEIGHT == 'balanced':
                    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
                else: 
                    class_weights = {0:float(config.CLASS_WEIGHT.split(':')[0]), 1:float(config.CLASS_WEIGHT.split(':')[1])}
            else:
                class_weights = class_weight.compute_class_weight(None, np.unique(train_y), train_y)
        else:
            class_weights = class_weight.compute_class_weight(None, np.unique(np.argmax(train_y, axis=1)), np.argmax(train_y, axis=1))
        print (class_weights)
        
        if classes_len == 2:
            model.compile(optimizer=Adam(lr=1e-3),
                      loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy()])
            
            hist_1 = model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs1,
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(monitor='val_binary_accuracy', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )
            
        else:
            model.compile(optimizer=Adam(1e-3),
                          loss=CategoricalCrossentropy(),
                          metrics=[CategoricalAccuracy()])

            hist_1 = model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs1,
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(monitor='val_categorical_accuracy', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=4)
                ]
            )

    elif config.dataset == "CBIS-DDSM":
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy()])

        hist_1 = model.fit(x=train_x,
                           validation_data=val_x,
                           epochs=epochs1,
                           callbacks=[
                               EarlyStopping(monitor='val_binary_accuracy', patience=8, restore_best_weights=True),
                               ReduceLROnPlateau(patience=6)]
                           )

    # Plot the training loss and accuracy.
    plot_training_results(classes_len, hist_1, "Initial_training", True)
    
    hist_1_df = pd.DataFrame(hist_1.history) 
    with open('../output/hist_1.json', mode='w') as f:
        hist_1_df.to_json(f)

    # Train a second time with a smaller learning rate and with all layers unfrozen
    # (train over fewer epochs to prevent over-fitting).
    if config.model.startswith('basic'):
        if config.model == 'basic':
            model.layers[0].trainable = True
        else:
            model.layers[1].trainable = True
    else:
        model.layers[config.CONV_CNT*2].trainable = True

    if config.dataset == "mini-MIAS":
        if classes_len == 2:
            model.compile(optimizer=Adam(lr=1e-5),
                      loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy()])
            
            hist_2 = model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs2,
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(monitor='val_binary_accuracy', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=6)
                ]
            )
            
        else:
            model.compile(optimizer=Adam(1e-5),  # Very low learning rate
                          loss=CategoricalCrossentropy(),
                          metrics=[CategoricalAccuracy()])

            hist_2 = model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_s,
                steps_per_epoch=len(train_x) // batch_s,
                validation_data=(val_x, val_y),
                validation_steps=len(val_x) // batch_s,
                epochs=epochs2,
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(monitor='val_categorical_accuracy', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(patience=6)
                ]
            )
    elif config.dataset == "CBIS-DDSM":
        model.compile(optimizer=Adam(lr=1e-5),  # Very low learning rate
                      loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy()])

        hist_2 = model.fit(x=train_x,
                           validation_data=val_x,
                           epochs=epochs2,
                           callbacks=[
                               EarlyStopping(monitor='val_binary_accuracy', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(patience=6)]
                           )

    # Plot the training loss and accuracy.
    plot_training_results(classes_len, hist_2, "Fine_tuning_training", False)
    
    hist_2_df = pd.DataFrame(hist_2.history) 
    with open('../output/hist_2.json', mode='w') as f:
        hist_2_df.to_json(f)
        
    return model


def make_predictions(model, x_values):
    """
    :param model: The CNN model.
    :param x: Input.
    :return: Model predictions.
    """
    if config.dataset == "mini-MIAS":
        y_predict = model.predict(x=x_values.astype("float32"), batch_size=8)
    elif config.dataset == "CBIS-DDSM":
        y_predict = model.predict(x=x_values)
    return y_predict
