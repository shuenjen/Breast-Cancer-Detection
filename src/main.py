# This file is contributed by Adam Jaamour, Ashay Patel, and Shuen-Jen Chen

import argparse
import time
import tensorflow as tf
import numpy as np
from datetime import datetime

import config
from data_operations.dataset_feed import create_dataset
from data_operations.data_preprocessing import import_cbisddsm_training_dataset, import_cbisddsm_testing_dataset, import_minimias_dataset, \
    dataset_stratified_split, generate_image_transforms, import_minimias_dataset_roi, generate_image_transforms_upsample, generate_image_transforms_downsample
from data_visualisation.output import evaluate
from model.train_test_model import make_predictions, train_network
from model.vgg_model import generate_vgg_model
from model.vgg_model_advance import generate_vgg_model_advance
from model.vgg_model_add_density import generate_vgg_model_and_density
from model.vgg_model_advance_add_density import generate_vgg_model_advance_and_density
from model.resnet_model import generate_resnet_model
from model.resnet_model_add_density import generate_resnet_model_and_density
from utils import create_label_encoder, print_error_message, print_num_gpus_available, print_runtime, print_config
from tensorflow.keras.models import load_model

def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which dataset and model to use.
    :return: None.
    """
    parse_command_line_arguments()
    print_num_gpus_available()
    
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

    # Start recording time.
    start_time = time.time()

    # Create label encoder.
    l_e = create_label_encoder()

    # Run in training mode.
    if config.run_mode == "train":

        # Multiclass classification (mini-MIAS dataset)
        if config.dataset == "mini-MIAS":
            # Import entire dataset.
            images, chars, labels = import_minimias_dataset(data_dir="../data/{}/images".format(config.dataset),
                                                     label_encoder=l_e)
            
            # Split dataset into training/test/validation sets (60%/20%/20% split).
            X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=X_train, labels=y_train)
            
            if config.SAMPLING == 'x':
                X_train_rebalanced = X_train
                y_train_rebalanced = y_train
            else:
                print (len(y_train))
                print (l_e.classes_)
                print (y_train.sum(axis=0))

                if len(config.CLASS_TYPE.split('-')) == 2:
                    if config.SAMPLING == 'up':
                        X_train_rebalanced, y_train_rebalanced = generate_image_transforms_upsample(X_train, y_train)
                    elif config.SAMPLING == 'down':
                        X_train_rebalanced, y_train_rebalanced = generate_image_transforms_downsample(X_train, y_train)

                if len(config.CLASS_TYPE.split('-')) != 2 and config.SAMPLING == 'up':
                    X_train_rebalanced, y_train_rebalanced = generate_image_transforms(X_train, y_train)
                    
                print (len(y_train_rebalanced))
                print (l_e.classes_)
                print (y_train_rebalanced.sum(axis=0))
            
            # Create and train CNN model.
            if config.cnn == "ResNet":
                model = generate_resnet_model(l_e.classes_.size)
            elif config.cnn == "VGG":
                if config.model == 'basic':
                    model = generate_vgg_model(l_e.classes_.size)
                else:
                    model = generate_vgg_model_advance(l_e.classes_.size)
            
            model = train_network(l_e.classes_.size, model, X_train_rebalanced, y_train_rebalanced, X_val, y_val, config.BATCH_SIZE, config.EPOCH_1,
                                  config.EPOCH_2)

        # Binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels, density, cc, mlo = import_cbisddsm_training_dataset(l_e)
            images_test, labels_test, density_test, cc_test, mlo_test = import_cbisddsm_testing_dataset(l_e)
            
            if len(config.model.split('-')) > 1 and config.model.split('-')[1] == '3':
                X = np.vstack((images, density, cc, mlo))
                X_test = np.vstack((images_test, density_test, cc_test, mlo_test))
                X_test = X_test.transpose()
            else:
                X = np.vstack((images, density))
                X_test = np.vstack((images_test, density_test))
                X_test = X_test.transpose()
                
            y_test = labels_test
            
            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=X.transpose(), labels=labels)
            # X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=images, labels=labels)
            
            dataset_train = create_dataset(X_train, y_train)
            dataset_val = create_dataset(X_val, y_val)
            dataset_test = create_dataset(X_test, y_test)
            
            # Create and train CNN model.
            if config.cnn == "ResNet":
                if len(config.model.split('-')) == 1:
                    model = generate_resnet_model(l_e.classes_.size)
                else:
                    model = generate_resnet_model_and_density(l_e.classes_.size)

            elif config.cnn == "VGG":
                if config.model.startswith('basic'):
                    if len(config.model.split('-')) == 1:
                        model = generate_vgg_model(l_e.classes_.size)
                    else:
                        model = generate_vgg_model_and_density(l_e.classes_.size)
                else:
                    if len(config.model.split('-')) == 1:
                        model = generate_vgg_model_advance(l_e.classes_.size)
                    else:
                        model = generate_vgg_model_advance_and_density(l_e.classes_.size)
                    
            model = train_network(l_e.classes_.size, model, dataset_train, None, dataset_val, None, config.BATCH_SIZE, config.EPOCH_1, config.EPOCH_2)

        else:
            print_error_message()

        try:
            # Save the model
            # model.save("../saved_models/dataset-{}_model-{}-{}_" + datetime.now().strftime("%d%Y%H%M%S") + ".h5".format(config.dataset, config.model, config.cnn))
            save_time = datetime.now().strftime("%Y%m%d%H%M")
            model.save_weights("/cs/tmp/sjc29/saved_models/dataset-{}_model-{}-{}_{}.h5".format(config.dataset, config.model, config.cnn, save_time))
        except:
            print ('save model error: ' + sys.exc_info()[0])

    elif config.run_mode == "test":
        model.load("/cs/tmp/sjc29/saved_models/dataset-{}_model-{}-{}_{}.h5".format(config.dataset, config.model, config.cnn, config.MODEL_SAVE_TIME))

    # print config
    print_config()
    print ('save_time: ', save_time)
    print_runtime("Finish Training", round(time.time() - start_time, 2))

    # Evaluate model results.
    if config.dataset == "mini-MIAS":
        if config.run_mode == "train":
            y_pred = make_predictions(model, X_val)
            evaluate(y_val, y_pred, l_e, config.dataset, config.CLASS_TYPE, 'output')
            print_runtime("Finish Prediction Validation Set", round(time.time() - start_time, 2))
        elif config.run_mode == "test":
            y_pred_test = make_predictions(model, X_test)
            evaluate(y_test, y_pred_test, l_e, config.dataset, config.CLASS_TYPE, 'output_test')
            print_runtime("Finish Prediction Testing Set", round(time.time() - start_time, 2))
    elif config.dataset == "CBIS-DDSM":
        if config.run_mode == "train":
            y_pred = make_predictions(model, dataset_val)
            evaluate(y_val, y_pred, l_e, config.dataset, config.CLASS_TYPE, 'output')
            print_runtime("Finish Prediction Validation Set", round(time.time() - start_time, 2))
        elif config.run_mode == "test":
            y_pred_test = make_predictions(model, dataset_test)
            evaluate(y_test, y_pred_test, l_e, config.dataset, config.CLASS_TYPE, 'output_test')
            print_runtime("Finish Prediction Testing Set", round(time.time() - start_time, 2))

    # Print the prediction
    # print(y_pred)

    # Print training runtime.
    print_runtime("Total", round(time.time() - start_time, 2))


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="mini-MIAS",
                        required=True,
                        help="The dataset to use. Must be either 'mini-MIAS' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-c", "--cnn",
                        default="ResNet",
                        help="The CNN architecture to use. Must be either 'VGG' or 'ResNet'."
                        )
    parser.add_argument("-m", "--model",
                        default="basic",
                        required=True,
                        help="The model to use. Must be either 'basic' or 'advanced'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="Running mode: train model from scratch and make predictions, otherwise load pre-trained "
                             "model for predictions. Must be either 'train' or 'test'."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )

    args = parser.parse_args()
    config.dataset = args.dataset
    config.cnn = args.cnn
    config.model = args.model
    config.run_mode = args.runmode
    config.verbose_mode = args.verbose
    

if __name__ == '__main__':
    main()
