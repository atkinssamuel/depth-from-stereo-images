import numpy as np
from scipy.ndimage.filters import *


def sum_of_absolute_differences(x, y, x_search, window_size, Il, Ir):
    SAD = np.mean(np.absolute(
        Il[y:y + window_size, x:x + window_size] - Ir[y:y + window_size, x_search:x_search + window_size]))
    return SAD


def sum_of_squared_differences(x, y, x_search, window_size, Il, Ir):
    SSD = np.sum(np.square(np.absolute(Il[y:y + window_size, x:x + window_size] -
                                       Ir[y:y + window_size, x_search:x_search + window_size])))
    return SSD

# Explanation of Stereo Disparity Best:
# I calculated an original disparity mapping using a windowed method. I calculated the gradient SADs in the
# horizontal and vertical directions and I calculated the SAD. After I got an initial disparity mapping, I computed
# a weighted smoothing map. This weighted smoothing map was computed by adding 4 matrices together that described the
# SAD confidence of neighbouring pixels. This smoothing matrix was used in the next iteration to better define the
# disparity mapping.
def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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

    # ------------------
    # Initializing Id and SAD matrices:
    Id = np.zeros(Il.shape)
    SAD_matrix = np.zeros(Il.shape)
    # Defining the window size and the bounds on which we will search through later:
    window_size = 2
    UR = bbox[1][0]
    BR = bbox[1][1] + 1 - window_size
    UL = bbox[0][0]
    BL = bbox[0][1] + 1 - window_size

    Il_grad_0 = sobel(Il, axis=0)
    Il_grad_1 = sobel(Il, axis=1)
    Ir_grad_0 = sobel(Ir, axis=0)
    Ir_grad_1 = sobel(Ir, axis=1)

    # Looping through y and x in left image:
    for y in range(UR, BR, window_size):
        for x in range(UL, BL, window_size):
            score_array = []
            # Searching across all x values in the disparity bounds for the right image:
            for x_search in range(x - maxd, x + 1):
                # Computing the SAD score between the constant window in the left image and the current window in
                # the right image:
                score = sum_of_absolute_differences(x, y, x_search, window_size, Il, Ir)
                score_array.append(score)
            best_index = np.argmin(np.array(score_array))
            # Appending the disparity that had the lowest SAD score and adding the SAD score to the SAD matrix:
            SAD_matrix[y:y + window_size, x:x + window_size] = score_array[best_index]
            Id[y:y + window_size, x:x + window_size] = maxd - best_index

    # Defining up, down, right, and left kernels:
    # These kernels will be used to compute the differences between the center pixel and the top, bottom, left, and
    # right pixels. These differences will then be multiplied to shifted versions of the SAD matrix.
    up_kernel = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    down_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    right_kernel = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    left_kernel = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

    # Defining left, right, up, and down convolutions. These convolutions represent the difference matrices with respect
    # to each pixel direction.
    left_conv = convolve(Id, left_kernel)
    right_conv = convolve(Id, right_kernel)
    up_conv = convolve(Id, up_kernel)
    down_conv = convolve(Id, down_kernel)

    # Defining the left, right, down, and up shifts to shift the SAD matrix:
    left_shift = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    right_shift = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    down_shift = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    up_shift = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    # Computing the shifted matrices:
    left_SAD = convolve(SAD_matrix, left_shift)
    right_SAD = convolve(SAD_matrix, right_shift)
    up_SAD = convolve(SAD_matrix, up_shift)
    down_SAD = convolve(SAD_matrix, down_shift)

    # Multiplying the matrices element-wise to determine a smoothing matrix for the next disparity map:
    left_product = np.multiply(left_SAD, left_conv)
    right_product = np.multiply(right_SAD, right_conv)
    up_product = np.multiply(up_SAD, up_conv)
    down_product = np.multiply(down_SAD, down_conv)

    # Summing the matrices and squaring the result to get a smoothing matrix that can be used in the next disparity
    # map calculation:
    smoothing = np.square(left_product + right_product + up_product + down_product)

    # Looping through y and x in left image:
    for y in range(UR, BR, window_size):
        for x in range(UL, BL, window_size):
            score_array = []
            # Searching across all x values in the disparity bounds for the right image:
            for x_search in range(x - maxd, x + 1):
                # Now that we have the smoothing matrix based on our original disparity map "guess", we can recompute
                # a distance function to get a more accurate estimate. This distance metric estimate is called the
                # "score". The following quantities were used to compute the "score":
                # 1. SAD score between left image window and current right image window
                score = sum_of_absolute_differences(x, y, x_search, window_size, Il, Ir)
                # 2. SAD score between left image window horizontal gradient and current right image window horizontal
                # gradient
                grad_score_0 = sum_of_squared_differences(x, y, x_search, window_size, Il_grad_0, Ir_grad_0)
                # 3. SAD score between left image window vertical gradient and current right image window vertical
                # gradient
                grad_score_1 = sum_of_squared_differences(x, y, x_search, window_size, Il_grad_1, Ir_grad_1)
                # 4. The smoothing score
                smoothing_score = smoothing[y, x_search]

                score_array.append(score + smoothing_score + grad_score_0 + grad_score_1)

            best_index = np.argmin(np.array(score_array))
            Id[y:y + window_size, x:x + window_size] = maxd - best_index

    return median_filter(Id, 10)
