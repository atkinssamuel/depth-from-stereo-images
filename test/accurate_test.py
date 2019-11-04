import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from main.stereo_disparity_best import stereo_disparity_best

Il = imread("../images/teddy_image_02.png", as_gray = True)
Ir = imread("../images/teddy_image_06.png", as_gray = True)

# Load the appropriate bounding box.
bboxes = loadmat("../images/bboxes.mat")
bbox = np.array(bboxes["teddy_02"]["bbox"])

Id = stereo_disparity_best(Il, Ir, bbox, 52)
plt.imshow(Id, cmap = "gray")
plt.show()