import numpy as np
from matplotlib.path import Path
import  matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from matplotlib import patches

from main.stereo_disparity_fast import stereo_disparity_fast

Il = imread("../images/teddy_image_02.png", as_gray = True)
Ir = imread("../images/teddy_image_06.png", as_gray = True)

# Load the appropriate bounding box.
bboxes = loadmat("../images/bboxes.mat")
bbox = np.array(bboxes["teddy_02"]["bbox"])

# Setting the UL, UR, BR, and BL corners from the bounds array:
UL = np.array([bbox[0][0], bbox[1][0]])
UR = np.array([bbox[0][1], bbox[1][0]])
BR = np.array([bbox[0][1], bbox[1][1]])
BL = np.array([bbox[0][0], bbox[1][1]])

bounds_list = UL, UR, BR, BL
bounding_poly = [UL, UR, BR, BL, UL]

# Defining inner and outer borders using matplotlib's Path:
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY
]
path = Path(bounding_poly, codes)

patch = patches.PathPatch(path, facecolor="none", lw=2)

fig, ax = plt.subplots()
ax.add_patch(patch)
plt.imshow(Il, cmap="gray")
plt.show()
Id = stereo_disparity_fast(Il, Ir, bbox, 52)

plt.imshow(Id, cmap="gray")
plt.savefig('../results/Id')
plt.show()