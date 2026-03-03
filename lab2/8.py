import numpy as np
import cv2
from skimage.morphology import opening, closing
from skimage.morphology import disk
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


im = cv2.imread("blood-cells.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_R = im[:, :, 2]  # red channel is missing the blue neutrophils

# apply threshold on gray level image to find everything (cells & neutrophils)
im_threshold = threshold_otsu(im_gray)
_, mask = cv2.threshold(im_gray, im_threshold, 255, cv2.THRESH_BINARY)
mask_inv = 255 - mask
opened_mask_inv = opening(mask_inv, footprint=disk(5))

# Only at the pixels where cells/neutrophils are present, apply threshold
# on Red channel to distinguish between cells and neutrophils.
# This can work because cells and neutrophils has the most different pixel
# values in the Red channel.
im_R_threshold = threshold_otsu(im_R[np.where(opened_mask_inv > 0)])
_, mask_R = cv2.threshold(im_R, im_R_threshold, 255, cv2.THRESH_BINARY)
mask_R_inv = 255 - mask_R
opened_mask_R_inv = opening(mask_R_inv, footprint=disk(3))
closed_opened_mask_R_inv = closing(opened_mask_R_inv, footprint=disk(3))

# Recreate the image with only neutrophils
im_filtered = im.copy()
im_filtered[np.where(closed_opened_mask_R_inv == 0)] = (255, 255, 255)

# convert color for matplotlib style (RGB instead of BGR)
im_RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_filtered_RGB = cv2.cvtColor(im_filtered, cv2.COLOR_BGR2RGB)

plt.rcParams["figure.figsize"] = [8, 6]

plt.subplot(2, 2, 1)
plt.imshow(opened_mask_inv, cmap="gray", vmin=0, vmax=255)
plt.title("Cleaned mask for cells + neutrophils")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(closed_opened_mask_R_inv, cmap="gray", vmin=0, vmax=255)
plt.title("Cleaned mask for neutrophils only")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(im_RGB)
plt.title("Original image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(im_filtered_RGB)
plt.title("Filters applied")
plt.axis("off")

plt.tight_layout()
plt.show()




