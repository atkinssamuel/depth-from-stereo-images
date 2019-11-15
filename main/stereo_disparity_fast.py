import numpy as np


def sum_of_absolute_differences(x, y, x_search, window_size, Il, Ir):
    SAD = np.mean(np.absolute(
        Il[y:y + window_size, x:x + window_size] - Ir[y:y + window_size, x_search:x_search + window_size]))
    return SAD


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
    # Initializing Id matrix:
    Id = np.zeros(Il.shape)

    # Defining the window size and the bounds on which we will search through later:
    window_size = 7
    UR = bbox[1][0]
    BR = bbox[1][1] + 1 - window_size
    UL = bbox[0][0]
    BL = bbox[0][1] + 1 - window_size

    # Looping through y and x in left image:
    for y in range(UR, BR, window_size):
        for x in range(UL, BL, window_size):
            SAD_array = []
            # Searching across all x values in the disparity bounds for the right image:
            for x_search in range(x - maxd, x + 1):
                # Computing the SAD score between the constant window in the left image and the current window in
                # the right image:
                SAD_array.append(sum_of_absolute_differences(x, y, x_search, window_size, Il, Ir))
            # Appending the disparity that had the lowest SAD score:
            Id[y:y + window_size, x:x + window_size] = maxd - np.argmin(np.array(SAD_array))
    # ------------------

    return Id

