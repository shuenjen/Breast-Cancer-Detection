# This file is contributed by Adam Jaamour, and Ashay Patel

import os
import random

from imutils import paths
import numpy as np
import pandas as pd
import skimage as sk
import skimage.transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

import config


def import_minimias_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    :param data_dir: Directory to the mini-MIAS images.
    :param label_encoder: The label encoder.
    :return: Two NumPy arrays, one for the processed images and one for the encoded labels.
    """
    # Initialise variables.
    images = list()
    labels = list()
    chars = list()

    df = pd.read_csv('/'.join(data_dir.split('/')[:-1]) + '/data_description.csv', header=None)
    df_cnt = pd.DataFrame(df.groupby([0])[3].nunique()).reset_index()
    df = df[~df[0].isin(list(df_cnt[df_cnt[3] > 1][0]))]
    df = df.drop_duplicates(subset=[0], keep='first')
    for row in df.iterrows():
        if (row[1][3] is not np.nan) & (row[1][4] is np.nan):
            continue
        
        if config.CLASS_TYPE == 'N-A':
            label = '0'
            if row[1][3] == 'B':
                label = '1'
            elif row[1][3] == 'M':
                label = '1'
        elif config.CLASS_TYPE == 'N-B-M':
            label = 'normal'
            if row[1][3] == 'B':
                label = 'benign'
            elif row[1][3] == 'M':
                label = 'malignant'
        elif config.CLASS_TYPE == 'B-M':
            if row[1][3] == 'B':
                label = 'benign'
            elif row[1][3] == 'M':
                label = 'malignant'
            else:
                continue
        
        images.append(preprocess_image(data_dir + '/' + row[1][0] + '.png'))        
        labels.append(label)
        chars.append(row[1][1])
        
    # Convert the data and labels lists to NumPy arrays.
    images = np.array(images, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)
    chars = np.array(chars)
    
    # Encode labels.
    labels = encode_labels(labels, label_encoder)
    chars = encode_labels(chars, LabelEncoder())
    
    # return images, labels
    return images, chars, labels

def import_minimias_dataset_roi(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    :param data_dir: Directory to the mini-MIAS images.
    :param label_encoder: The label encoder.
    :return: Two NumPy arrays, one for the processed images and one for the encoded labels.
    """
    # Initialise variables.
    images = list()
    labels = list()
    chars = list()
    
    df = pd.read_csv('/'.join(data_dir.split('/')[:-1]) + '/data_description.csv', header=None)
    
    for row in df.iterrows():
        if (row[1][3] is not np.nan) & (row[1][4] is np.nan):
            continue
        
        if (row[1][2] != 'NORM') & (row[1][4] == '*NO'):
            continue
        
        image = load_img(data_dir + '/' + row[1][0] + '.png', color_mode="grayscale")
        image = img_to_array(image)
        image /= 255.0
        
        y2 = 0
        x2 = 0
    
        if row[1][2] != 'NORM':
            y1 = image.shape[1] - int(row[1][5]) - 112
            if y1 < 0:
                y1 = 0
                y2 = 224
            
            if y2 != 224:
                y2 = image.shape[1] - int(row[1][5]) + 112
                if y2 > image.shape[1]:
                    y2 = image.shape[1]
                    y1 = image.shape[1] - 224

            x1 = int(row[1][4]) - 112
            if x1 < 0:
                x1 = 0
                x2 = 224

            if x2 != 224:
                x2 = int(row[1][4]) + 112
                if x2 > image.shape[0]:
                    x2 = image.shape[0]
                    x1 = image.shape[0] - 224
        else:
            y1 = int(image.shape[1]/2 - 112)
            y2 = int(image.shape[1]/2 + 112)
            x1 = int(image.shape[0]/2 - 112)
            x2 = int(image.shape[0]/2 + 112)
        
        if config.CLASS_TYPE == 'N-A':
            label = '0'
            if row[1][3] == 'B':
                label = '1'
            elif row[1][3] == 'M':
                label = '1'
        elif config.CLASS_TYPE == 'N-B-M':
            label = 'normal'
            if row[1][3] == 'B':
                label = 'benign'
            elif row[1][3] == 'M':
                label = 'malignant'
        elif config.CLASS_TYPE == 'B-M':
            if row[1][3] == 'B':
                label = 'benign'
            elif row[1][3] == 'M':
                label = 'malignant'
            else:
                continue
        
        images.append(image[y1:y2, x1:x2, :])    
        labels.append(label)
        chars.append(row[1][1])
        
    # Convert the data and labels lists to NumPy arrays.
    images = np.array(images, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)
    chars = np.array(chars)

    # Encode labels.
    labels = encode_labels(labels, label_encoder)
    chars = encode_labels(chars, LabelEncoder())

    # return images, labels
    return images, chars, labels


def import_cbisddsm_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    df = pd.read_csv("../data/CBIS-DDSM/training.csv")
    # df = df[df['img'].str.endswith('_MLO')]
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    density = df['breast_density'].values
    cc = df['img'].map(lambda row: 1 if 'CC' in row else 0).values
    mlo = df['img'].map(lambda row: 1 if 'MLO' in row else 0).values
    # print (np.unique(cc), np.unique(mlo))
    return list_IDs, labels, density, cc, mlo

def import_cbisddsm_testing_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    df = pd.read_csv("../data/CBIS-DDSM/testing.csv")
    # df = df[df['img'].str.endswith('_MLO')]
    list_IDs = df['img_path'].values
    labels = label_encoder.transform(df['label'].values)
    density = df['breast_density'].values
    cc = df['img'].map(lambda row: 1 if 'CC' in row else 0).values
    mlo = df['img'].map(lambda row: 1 if 'MLO' in row else 0).values
    # print (np.unique(cc), np.unique(mlo))
    return list_IDs, labels, density, cc, mlo

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pre-processing steps:
        * Load the input image in grayscale mode (1 channel),
        * resize it to 224x224 pixels for the VGG19 CNN model,
        * transform it to an array format,
        * normalise the pixel intensities.
    :param image_path: The path to the image to preprocess.
    :return: The pre-processed image in NumPy array format.
    """
    image = load_img(image_path,
                     color_mode="grayscale",
                     target_size=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"]))
    image = img_to_array(image)
    image /= 255.0
    return image


def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    :param label_encoder: The label encoder.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)


def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Partition the data into training and testing splits. Stratify the split to keep the same class distribution in both
    sets and shuffle the order to avoid having imbalanced splits.
    :param split: Dataset split (e.g. if 0.2 is passed, then the dataset is split in 80%/20%).
    :param dataset: The dataset of pre-processed images.
    :param labels: The list of labels.
    :return: the training and testing sets split in input (X) and label (Y).
    """
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    return train_X, test_X, train_Y, test_Y


def random_rotation(image_array: np.ndarray):
    """
    Randomly rotate the image
    :param image_array: input image
    :return: randomly rotated image
    """
    random_degree = random.uniform(-180, 180)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: np.ndarray):
    """
    Add random noise to image
    :param image_array: input image
    :return: image with added random noise
    """
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: np.ndarray):
    """
    Flip image
    :param image_array: input image
    :return: horizantally flipped image
    """
    return image_array[:, ::-1]


def generate_image_transforms(images, labels):
    """
    oversample data by tranforming existing images
    :param images: input images
    :param labels: input labels
    :return: updated list of images and labels with extra transformed images and labels
    """
    images_with_transforms = images
    labels_with_transforms = labels

    available_transforms = {'rotate': random_rotation,
                            'noise': random_noise,
                            'horizontal_flip': horizontal_flip}

    class_balance = get_class_balances(labels)
    max_count = max(class_balance) * config.SAMPLING_TIMES
    to_add = [max_count - i for i in class_balance]

    for i in range(len(to_add)):
        if int(to_add[i]) == 0:
            continue
        label = np.zeros(len(to_add))
        label[i] = 1
        indices = [j for j, x in enumerate(labels) if np.array_equal(x, label)]
        indiv_class_images = [images[j] for j in indices]

        for k in range(int(to_add[i])):
            # a = create_individual_transform(indiv_class_images[k % len(indiv_class_images)], available_transforms)
            transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
                                                            available_transforms)
            transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'],
                                                          config.VGG_IMG_SIZE['WIDTH'], 1)
            
            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            transformed_label = label.reshape(1, len(label))
            labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)

    return images_with_transforms, labels_with_transforms

def generate_image_transforms_upsample(images, labels):
    """
    oversample data by tranforming existing images
    :param images: input images
    :param labels: input labels
    :return: updated list of images and labels with extra transformed images and labels
    """
    images_with_transforms = images
    labels_with_transforms = labels

    available_transforms = {'rotate': random_rotation,
                            'noise': random_noise,
                            'horizontal_flip': horizontal_flip}

    class_balance = np.array([np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)])
    max_count = max(class_balance) * config.SAMPLING_TIMES
    to_add = [max_count - i for i in class_balance]

    for i in range(len(to_add)):
        if int(to_add[i]) == 0:
            continue
        label = np.zeros(len(to_add))
        label[i] = str(i)
        indices = [j for j, x in enumerate(labels) if x == label[i]]
        indiv_class_images = [images[j] for j in indices]

        for k in range(int(to_add[i])):
            # a = create_individual_transform(indiv_class_images[k % len(indiv_class_images)], available_transforms)
            transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
                                                            available_transforms)
            transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'],
                                                          config.VGG_IMG_SIZE['WIDTH'], 1)
            
            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            labels_with_transforms = np.append(labels_with_transforms, label[i])

    return images_with_transforms, labels_with_transforms

def generate_image_transforms_downsample(images, labels):
    """
    oversample data by tranforming existing images
    :param images: input images
    :param labels: input labels
    :return: updated list of images and labels with extra transformed images and labels
    """
    images_with_downsample = images
    labels_with_dowmsample = labels

    class_balance = np.array([np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)])
    min_count = min(class_balance)
    to_del = [i - min_count for i in class_balance]
    
    for i in range(len(to_del)):
        if int(to_del[i]) == 0:
            continue
        
        indices = [j for j, x in enumerate(labels) if x == i]
        del_lst = random.sample(indices, to_del[i])

        images_with_downsample = [im for i, im in enumerate(images_with_downsample) if i not in del_lst]
        labels_with_dowmsample = [label for i, label in enumerate(labels_with_dowmsample) if i not in del_lst]
    
    return np.array(images_with_downsample), np.array(labels_with_dowmsample)


def create_individual_transform(image: np.array, transforms: dict):
    """
    Create transformation of an individual image
    :param image: input image
    :param transforms: the possible transforms to do on the image
    :return: transformed image
    """
    num_transformations_to_apply = random.randint(1, len(transforms))
    num_transforms = 0
    transformed_image = None
    while num_transforms <= num_transformations_to_apply:
        key = random.choice(list(transforms))
        transformed_image = transforms[key](image)
        num_transforms += 1

    return transformed_image


def get_class_balances(y_vals):
    """
    Count occurrences of each class.
    :param y_vals: labels
    :return: array count of each class
    """
    num_classes = len(y_vals[0])
    counts = np.zeros(num_classes)
    for y_val in y_vals:
        for i in range(num_classes):
            counts[i] += y_val[i]

    return (counts.tolist())
