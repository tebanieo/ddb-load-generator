"""Main module."""
import sys

import cv2 as cv
import numpy as np

max_value = 255
max_type = 4
max_binary_value = 255


def first_nonzero(array, axis, invalid_value=-1):
    """This function finds the first non zero occurency in the graph and returns an array with the position.

    Args:
        array ([type]): [description]
        axis ([type]): [description]
        invalid_value (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    mask = array != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_value)


def process_image(file_name=None):
    """
    Process the image as specified in file name, including the filepath and returns an array
    with the first ocurrances of a black pixel.

    Args:
        file_name ([String], required): The file name where the image is located, int requires
        the full path to work. Defaults to None (and not processing anything)

    Returns:
        [load_function]: A normalized numpy array that represents the value of the graphic for
        each pixel on the picture.
    """
    src = cv.imread("sample_load.png")
    if src is None:
        sys.exit("Could not read the image.")

    src_gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    _, dst = cv.threshold(src_gray, 125, max_binary_value, 1)

    height, width = dst.shape[:2]

    # Find the first dark pixel from top to bottom.
    load_function = first_nonzero(dst, axis=0)
    load_function = load_function[load_function != -1]
    load_function = height - load_function
    load_function = load_function / load_function.argmax()
    print(f"The load function vector has {load_function.shape[0]} elements")
    print(load_function)
    return load_function


if __name__ == "__main__":
    process_image()
