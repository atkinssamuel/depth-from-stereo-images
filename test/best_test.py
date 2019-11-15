import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
import time

from main.stereo_disparity_best import stereo_disparity_best
maxd = 52

###########################################################
# Teddy Depth Mapping:
###########################################################
Il = imread("../images/teddy_image_02.png", as_gray = True)
Ir = imread("../images/teddy_image_06.png", as_gray = True)

# Load the appropriate bounding box.
bboxes = loadmat("../images/bboxes.mat")
bbox = np.array(bboxes["teddy_02"]["bbox"])

start = time.time()
Id = stereo_disparity_best(Il, Ir, bbox, maxd)
elapsed_time = round(time.time() - start, 3)
print("Elapsed time for best algorithm operating on teddy images =", elapsed_time)

plt.imshow(Id, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Teddy Image Depth Mapping for Best Algorithm ({}s)".format(elapsed_time))
plt.savefig('../results/best/depth_mapping_best_teddy.png')
plt.show()

###########################################################
# Cones Depth Mapping:
###########################################################
Il = imread("../images/cones_image_02.png", as_gray = True)
Ir = imread("../images/cones_image_06.png", as_gray = True)

# Load the appropriate bounding box.
bbox = np.array(bboxes["cones_02"]["bbox"])

start = time.time()
Id = stereo_disparity_best(Il, Ir, bbox, maxd)
elapsed_time = round(time.time() - start, 3)
print("Elapsed time for best algorithm operating on cones images =", elapsed_time)

plt.imshow(Id, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cones Image Depth Mapping for Best Algorithm ({}s)".format(elapsed_time))
plt.savefig('../results/best/depth_mapping_best_cones.png')
plt.show()

###########################################################
# Books Depth Mapping:
###########################################################
Il = imread("../images/books_image_01.png", as_gray = True)
Ir = imread("../images/books_image_05.png", as_gray = True)

# Load the appropriate bounding box.
bbox = np.array(bboxes["books_01"]["bbox"])

start = time.time()
Id = stereo_disparity_best(Il, Ir, bbox, 52)
elapsed_time = round(time.time() - start, 3)
print("Elapsed time for best algorithm operating on books images =", elapsed_time)

plt.imshow(Id, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Books Image Depth Mapping for Best Algorithm ({}s)".format(elapsed_time))
plt.savefig('../results/best/depth_mapping_best_books.png')
plt.show()