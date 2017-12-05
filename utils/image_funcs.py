""" Common routines for image pre-processing and data augmentation.
"""

import skimage
import skimage.io
import skimage.color
import numpy as np
import cv2


def read_image(file_path, to_float=True):
    """
    Read the image from file_path with an option to convert it to float format.
    """
    if to_float:
        return convert_image_data_type_to_float(skimage.io.imread(file_path))
    else:
        return skimage.io.imread(file_path)


def save_image(image, file_path, to_ubyte=True):
    """
    Save the image to file_path.
    """
    if to_ubyte:
        skimage.io.imsave(file_path, convert_image_data_type_to_uint64(image))
    else:
        skimage.io.imsave(file_path, image)


def rgb_to_gray(image):
    """
    Convert RGB image to gray-scale image.
    """
    return skimage.color.rgb2grey(image)


def convert_image_data_type_to_float(image):
    """
    Convert the image format to float64.
    """
    return skimage.img_as_float(image)


def convert_image_data_type_to_uint64(image):
    """
    Convert the image format to float64.
    """
    return skimage.img_as_ubyte(image)


def resize_bilinear_preserve_resoltion(image, smaller_dimension):
    """
    Resize the image using bilinear interpolation while preserving the resolution.
    """
    print(image.shape)
    old_smaller_dimension = np.min(image.shape[:2])
    scale = float(smaller_dimension) / old_smaller_dimension
    print(scale)
    new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    print(new_size)
    return resize_bilinear(image, new_size)


def resize_bilinear(image, shape):
    """
    Resize the image using bilinear interpolation.
    """
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def crop_random(image, shape):
    """
    Randomly crop the image.
    """
    assert shape < image.shape
    i_diff = image.shape[0] - shape[0] + 1
    j_diff = image.shape[1] - shape[1] + 1
    i_offset = np.random.randint(0, i_diff)
    j_offset = np.random.randint(0, j_diff)
    return image[i_offset:i_offset+shape[0], j_offset:j_offset+shape[1]]


def crop_center(image, shape):
    """
    Crop the image at the center.
    """
    assert shape < image.shape
    i = image.shape[0]//2-(shape[0]//2)
    j = image.shape[1]//2-(shape[1]//2)
    return image[i:i+shape[0], j:j+shape[1]]


def recenter_to_neg_one_and_one(image):
    """
    Rescale the image value to the range [-1, 1]
    """
    return image * 2.0 - 1.0
