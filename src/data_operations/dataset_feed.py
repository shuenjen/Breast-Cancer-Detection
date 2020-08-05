# This file is contributed by Adam Jaamour, and Ashay Patel

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

import config


def create_dataset(x, y):
    """
    Genereate a tensorflow dataset for feeding in the data
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset_x = tf.data.Dataset.from_tensor_slices((x[:, 0], np.array(x[:, 1:], dtype=np.int)))
    dataset_y = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    
    
    dataset = dataset.map(parse_function_small, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def parse_function_small(dataset, label):
    """
    mapping function to convert filename to array of pixel values
    :param filename:
    :param label:
    :return:
    """
    filename, features = dataset
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, scale="auto", dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    # image = tf.image.resize(decoded_png, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['HEIGHT']])
    image = tf.image.resize_with_pad(decoded_png, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    image /= 255
    
    return (image, features), label


def parse_function_large(filename, label):
    """
    mapping function to convert filename to array of pixel values for larger images we use resize with padding
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True,  dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize_with_pad(decoded_png, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['HEIGHT'])
    image /= 255

    return image, label
