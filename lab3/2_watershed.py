import numpy as np
import cv2
from skimage.morphology import disk
from skimage import segmentation as seg
from skimage.filters import gaussian, rank, threshold_otsu
from skimage.morphology import opening, closing, erosion, remove_small_objects
import skimage.util as util
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


im = cv2.imread("violin-and-hand.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray scale
im_l = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0]  # lightness channel
im_v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 2]  # value channel

# ====================== pre-processing for watershed ======================= #

# after checking, value channel has the highest contrast among the 3 above
im_cleaned = im_v.copy()
im_cleaned = rank.median(im_cleaned, footprint=disk(3))
im_cleaned = gaussian(im_cleaned, 2)

# pre-process to generate hand mask (as violin is dark)
thres_ostu = threshold_otsu(im_cleaned)
mask = im_cleaned > thres_ostu
mask = opening(mask, disk(5))
mask = closing(mask, disk(5))
mask = remove_small_objects(mask, max_size=1000)


# =========================== method 2: watershed =========================== #

im_cleaned = util.img_as_ubyte(im_cleaned)
distance = ndi.distance_transform_edt(im_cleaned)

# 3×3
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=im_cleaned)
# 2×2
coords_2 = peak_local_max(distance, footprint=np.ones((2, 2)), labels=im_cleaned)
# 5×5
coords_3 = peak_local_max(distance, footprint=np.ones((5, 5)), labels=im_cleaned)

distance_mask = np.zeros_like(im_cleaned, dtype=bool)
distance_mask_2 = np.zeros_like(im_cleaned, dtype=bool)
distance_mask_3 = np.zeros_like(im_cleaned, dtype=bool)
distance_mask[tuple(coords.T)] = True
distance_mask_2[tuple(coords_2.T)] = True
distance_mask_3[tuple(coords_3.T)] = True

markers = np.zeros(im_cleaned.shape, dtype=np.int32)
markers_2 = np.zeros(im_cleaned.shape, dtype=np.int32)
markers_3 = np.zeros(im_cleaned.shape, dtype=np.int32)
markers[~mask] = 1
markers_2[~mask] = 1
markers_3[~mask] = 1
markers[distance_mask] = 2
markers_2[distance_mask_2] = 2
markers_3[distance_mask_3] = 2

labels = seg.watershed(-distance, markers, mask=im_cleaned)
labels_2 = seg.watershed(-distance, markers_2, mask=im_cleaned)
labels_3 = seg.watershed(-distance, markers_3, mask=im_cleaned)
violin_hands = (labels == 2)
violin_hands_2 = (labels_2 == 2)
violin_hands_3 = (labels_3 == 2)

violin_hands = opening(violin_hands, disk(2))
violin_hands = remove_small_objects(violin_hands, max_size=100)
violin_hands_2 = opening(violin_hands_2, disk(2))
violin_hands_2 = remove_small_objects(violin_hands_2, max_size=100)
violin_hands_3 = opening(violin_hands_3, disk(2))
violin_hands_3 = remove_small_objects(violin_hands_3, max_size=100)

# best result is saved to file
im[np.where(~violin_hands)] = (0, 0, 0)
cv2.imwrite("violin-and-hand_watershed.jpg", im)

# ================================== plots ================================== #

plt.rcParams["figure.figsize"] = [16, 12]
row = 4
col = 4

plt.subplot(row, col, 1)
plt.imshow(im_cleaned, cmap="gray")
plt.title("pre-proccessed denoised")
plt.axis("off")

plt.subplot(row, col, 2)
plt.imshow(mask, cmap="gray")
plt.title("pre-proccessed hand mask")
plt.axis("off")

plt.subplot(row, col, 5)
plt.imshow(distance_mask, cmap="gray")
plt.title("distance mask with 3×3 footprint")
plt.axis("off")

plt.subplot(row, col, 6)
plt.imshow(markers, cmap=plt.cm.nipy_spectral)
plt.title("3-level markers: bg, hands, distance (with 3×3)")
plt.axis("off")

plt.subplot(row, col, 7)
plt.imshow(labels, cmap=plt.cm.nipy_spectral)
plt.title("labeled regions (with 3×3)")
plt.axis("off")

plt.subplot(row, col, 8)
plt.imshow(violin_hands, cmap="gray")
plt.title("cleaned violin & hand filter (with 3×3, best)")
plt.axis("off")

plt.subplot(row, col, 9)
plt.imshow(distance_mask_2, cmap="gray")
plt.title("distance mask with 2×2 footprint")
plt.axis("off")

plt.subplot(row, col, 10)
plt.imshow(markers_2, cmap=plt.cm.nipy_spectral)
plt.title("3-level markers: bg, hands, distance (with 2×2)")
plt.axis("off")

plt.subplot(row, col, 11)
plt.imshow(labels_2, cmap=plt.cm.nipy_spectral)
plt.title("labeled regions (with 2×2)")
plt.axis("off")

plt.subplot(row, col, 12)
plt.imshow(violin_hands_2, cmap="gray")
plt.title("cleaned violin & hand filter (with 2×2)")
plt.axis("off")

plt.subplot(row, col, 13)
plt.imshow(distance_mask_3, cmap="gray")
plt.title("distance mask with 5×5 footprint")
plt.axis("off")

plt.subplot(row, col, 14)
plt.imshow(markers_3, cmap=plt.cm.nipy_spectral)
plt.title("3-level markers: bg, hands, distance (with 5×5)")
plt.axis("off")

plt.subplot(row, col, 15)
plt.imshow(labels_3, cmap=plt.cm.nipy_spectral)
plt.title("labeled regions (with 5×5)")
plt.axis("off")

plt.subplot(row, col, 16)
plt.imshow(violin_hands_3, cmap="gray")
plt.title("cleaned violin & hand filter (with 5×5)")
plt.axis("off")

plt.tight_layout()
plt.show()