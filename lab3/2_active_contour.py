import numpy as np
import cv2
from skimage.morphology import disk
from skimage import segmentation as seg
from skimage.filters import gaussian, rank
import skimage.util as util
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...

plt.rcParams["figure.figsize"] = [12, 6]


im = cv2.imread("violin-and-hand.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray scale
im_l = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0]  # lightness channel
im_v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 2]  # value channel

# after checking, value channel has the highest contrast among the 3 above
im_cleaned = im_v.copy()
im_cleaned = rank.median(im_cleaned, footprint=disk(3))
# im_cleaned = opening(im_cleaned, footprint=disk(2))
im_cleaned = gaussian(im_cleaned, 3)


# ==================== method 1: geodesic active contour ==================== #

# gim = seg.inverse_gaussian_gradient(im_cleaned)
# init_ls = np.zeros_like(im_gray, dtype=np.int8)
# init_ls[:, 20:-50] = 1
# evolution = []
# ls = seg.morphological_geodesic_active_contour(
#     gim,
#     num_iter=500,
#     init_level_set=init_ls,
#     smoothing=1,
#     balloon=-0.65,
#     threshold=0.55,
#     iter_callback=lambda x: evolution.append(x),
# )
#
# plt.subplot(2, 2, 1)
# plt.imshow(im)
# plt.contour(ls, [0.5], colors='r')
# plt.title("Geodesic active counter")
# plt.axis("off")
#
# plt.subplot(2, 2, 2)
# plt.imshow(ls, cmap="gray", vmin=0, vmax=255)
# plt.title("Geodesic active counter evolution")
# plt.axis("off")
# contour_labels = []
# for n, color in ((0, 'g'), (100, 'y'), (300, 'b'), (500, 'r')):
#     plt.contour(evolution[n], [0.5], colors=color)
#     # Use empty line to represent this contour in the legend
#     legend_line = mlines.Line2D([], [], color=color, label=f"Iteration {n}")
#     contour_labels.append(legend_line)
# plt.legend(handles=contour_labels, loc="lower right")



#
# plt.subplot(2, 3, 6)
# plt.imshow(mask_otsu, cmap="gray", vmin=0, vmax=255)
# plt.title("Image with Otsu's thresholding")
# plt.axis("off")
#
# plt.subplot(2, 3, 4)
# plt.imshow(mask_local, cmap="gray", vmin=0, vmax=255)
# plt.title("Image with local thresholding")
# plt.axis("off")

plt.tight_layout()
plt.show()


