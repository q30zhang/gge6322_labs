import numpy as np
import cv2
from skimage import segmentation as seg
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


im = cv2.imread("violin-and-hand.jpg", cv2.IMREAD_COLOR_RGB)
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # gray scale
im_l = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[:, :, 0]  # lightness channel
im_v = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 2]  # value channel

# ======================== pre-processing for SLIC ========================== #

im_cleaned = im.copy()
# im_cleaned[:,:,0] = rank.median(im_cleaned[:,:,0], footprint=disk(3))
# im_cleaned[:,:,1] = rank.median(im_cleaned[:,:,1], footprint=disk(3))
# im_cleaned[:,:,2] = rank.median(im_cleaned[:,:,2], footprint=disk(3))
# im_cleaned = gaussian(im_cleaned, 1.5)

# no preprocessing is found to work the best for SLIC...

# =========================== method 2: watershed =========================== #

parameters_sets = [
    {"n_segments": 100, "compactness": 0.3, "convert2lab": True},
    {"n_segments": 500, "compactness": 0.3, "convert2lab": True},
    {"n_segments": 900, "compactness": 0.3, "convert2lab": True},

    {"n_segments": 100, "compactness": 0.5, "convert2lab": False},
    {"n_segments": 500, "compactness": 0.5, "convert2lab": False},
    {"n_segments": 900, "compactness": 0.5, "convert2lab": False},

    {"n_segments": 100, "compactness": 0.3, "convert2lab": False},
    {"n_segments": 500, "compactness": 0.3, "convert2lab": False},  # *best*
    {"n_segments": 900, "compactness": 0.3, "convert2lab": False},
]

segments = []
for parameters_set in parameters_sets:
    segments.append(seg.slic(im_cleaned, **parameters_set, start_label=1))

# apply segments to the images for some selected segments (good ones)
selected_segments = segments[-3:]
ims_seg = []

for s in selected_segments:
    props = regionprops_table(
        s,
        intensity_image=im_l,
        properties=["label", "centroid", "intensity_mean", "intensity_std"]
    )

    labels = props["label"]
    violin_hand_labels = labels[np.where(props["intensity_mean"] > 50)]
    im_seg = im.copy()
    im_seg[~np.isin(s, violin_hand_labels)] = (0, 0, 0)
    ims_seg.append(im_seg)

cv2.imwrite("violin-and-hand_slic.jpg", cv2.cvtColor(ims_seg[1], cv2.COLOR_RGB2BGR))

# ================================== plots ================================== #

plt.rcParams["figure.figsize"] = [16, 9]

col = 3
row = int(np.ceil(len(parameters_sets)/col)) + 1

def show_params(parameters):
    return ", ".join([f"{k}={v}" for k, v in parameters.items()])

for i in range(len(parameters_sets)):
    plt.subplot(row, col, 1 + i)
    plt.imshow(seg.mark_boundaries(im, segments[i], color=(1,0,0)))
    plt.title(show_params(parameters_sets[i]))
    plt.axis("off")

for i in range(len(selected_segments)):
    plt.subplot(row, col, len(parameters_sets) + i + 1)
    plt.imshow(ims_seg[i])
    plt.title("segmented image using this segmentation ↑")
    plt.axis("off")

plt.tight_layout()
plt.show()




