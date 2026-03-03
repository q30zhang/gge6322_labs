import numpy as np
import cv2
from skimage.morphology import disk
from skimage import segmentation as seg
from skimage import color
from skimage.measure import regionprops_table
from skimage.filters import gaussian, rank
from skimage.util import img_as_float
import matplotlib.pyplot as plt

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


im = cv2.imread("violin-and-hand.jpg", cv2.IMREAD_COLOR_RGB)
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # gray scale
im_l = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[:, :, 0]  # lightness channel
im_v = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 2]  # value channel

# ==================== pre-processing for FH Algorithm ====================== #

# im_cleaned = img_as_float(im)
im_cleaned = im.copy()
# im_cleaned = rank.median(im_cleaned, footprint=disk(3))
# im_cleaned = gaussian(im_cleaned, 3)

# ========================= method 4: FH Algorithm ========================== #

parameters_sets = [
    {"scale": 250, "sigma": 1.5, "min_size": 400},
    {"scale": 500, "sigma": 1.5, "min_size": 400},
    {"scale": 750, "sigma": 1.5, "min_size": 400},

    {"scale": 250, "sigma": 2, "min_size": 600},
    {"scale": 500, "sigma": 2, "min_size": 600},
    {"scale": 750, "sigma": 2, "min_size": 600},

    {"scale": 250, "sigma": 1.5, "min_size": 600},
    {"scale": 500, "sigma": 1.5, "min_size": 600},  # *best*
    {"scale": 750, "sigma": 1.5, "min_size": 600},
]

segments = []
outs = []
outs1 = []
for parameters_set in parameters_sets:
    segments.append(seg.felzenszwalb(im_cleaned, **parameters_set))
    outs.append(color.label2rgb(segments[-1], im_cleaned, kind='avg', bg_label=0))
    outs1.append(seg.mark_boundaries(im_cleaned, segments[-1]))

# apply segments to the images for some selected segments (good ones)
ims_seg = []

for s in segments:
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

cv2.imwrite("violin-and-hand_fh.jpg", cv2.cvtColor(ims_seg[6], cv2.COLOR_RGB2BGR))

# ================================== plots ================================== #

plt.rcParams["figure.figsize"] = [16, 9]

col = 3
row = int(np.ceil(len(parameters_sets)/col))

def show_params(parameters):
    return ", ".join([f"{k}={v}" for k, v in parameters.items()])

plt.figure(1)
for i in range(len(parameters_sets)):
    plt.subplot(row, col, 1 + i)
    plt.imshow(seg.mark_boundaries(im, segments[i], color=(1,0,0)))
    plt.title(show_params(parameters_sets[i]))
    plt.axis("off")
plt.suptitle("FH Algorithm boundary labels")
plt.tight_layout()

plt.figure(2)
for i in range(len(parameters_sets)):
    plt.subplot(row, col, i + 1)
    plt.imshow(ims_seg[i])
    plt.title(show_params(parameters_sets[i]))
    plt.axis("off")
plt.suptitle("FH Algorithm segmented images")
plt.tight_layout()

plt.show()
