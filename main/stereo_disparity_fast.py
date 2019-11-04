import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *


def sum_of_absolute_differences(matrix_a, matrix_b):
    difference = abs(matrix_a - matrix_b).flatten()
    SAD = sum(difference)
    return SAD


def window(image, x, y, d):
    left = x - d if x - d > 0 else 0
    right = x + d if x + d < image.shape[0] else image.shape[0]
    top = y - d if y - d > 0 else 0
    bottom = y + d if y + d < image.shape[1] else image.shape[1]
    return image[top:bottom, left:right]


def window_bounds(Il, Ir, x, y, d):
    left = x - d if x - d > 0 else 0
    right = min(x + d, Il.shape[1], Ir.shape[1])
    top = y - d if y - d > 0 else 0
    bottom = min(y + d, Il.shape[0], Ir.shape[0])
    return [left, right, top, bottom]



def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    # --- FILL ME IN ---

    # Your code goes here.

    # ------------------
    # Max disparity value = 63 pixels
    UL = np.array([bbox[0][0], bbox[1][0]])
    UR = np.array([bbox[0][1], bbox[1][0]])
    BR = np.array([bbox[0][1], bbox[1][1]])
    BL = np.array([bbox[0][0], bbox[1][1]])

    left = bbox[0][0]
    right = bbox[0][1]
    up = bbox[1][0]
    down = bbox[1][1]

    window_size = 5
    Id = np.zeros((Il.shape[0], Il.shape[1]))
    for y in range(up, down):
        for x in range(left, right):
            disparity_list = []
            for disparity in range(-maxd, maxd):
                # Define windows of equal size in left and right images:
                [L, R, T, B] = window_bounds(Il, Ir, y, x - disparity, window_size)
                Il_window = Il[T:B, L:R]
                Ir_window = Ir[T:B, L:R]

                # Compute the SAD score between the windows:
                SAD = sum_of_absolute_differences(Il_window, Ir_window)

                # Add score to list to be sorted later:
                disparity_list.append([SAD, disparity])

            # Choose the disparity with the lowest SAD score
            disparity_list = np.sort(np.array(disparity_list), axis=0)
            Id[y, x] = disparity_list[0][1]
    return Id
