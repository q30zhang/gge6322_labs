import numpy as np
import cv2
import skimage.data
from skimage.filters import threshold_isodata, threshold_otsu, threshold_local
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


# 1. (c)
im = np.array([[6, 5, 8, 7],
               [4, 2, 3, 8],
               [1, 8, 6, 1]])
print("Question 1. (c)")
print(f"Ridler-Calvard threshold value: {threshold_isodata(im)}")
print(f"Otsu's method threshold value: {threshold_otsu(im)}")

im_gray = skimage.data.page()

thres_ridler = threshold_isodata(im_gray)
thres_otsu = threshold_otsu(im_gray)
# print(f"Ridler-Calvard threshold value: {thres_ridler}")
# print(f"Otsu's method threshold value: {thres_otsu}")

_, mask_ridler = cv2.threshold(im_gray, thres_ridler, 255, cv2.THRESH_BINARY)
_, mask_otsu = cv2.threshold(im_gray, thres_otsu, 255, cv2.THRESH_BINARY)

# 1. (d)
thres_local = threshold_local(im_gray, block_size=151, offset=15).astype(np.uint8)

mask_local = np.zeros_like(im_gray)
mask_local[np.where(im_gray > thres_local)] = 255

# plots

plt.rcParams["figure.figsize"] = [12, 6]

plt.subplot(2, 3, 1)
plt.imshow(im_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Original gray image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.hist(im_gray.ravel(), bins=256)
plt.vlines(thres_ridler, 0, 2000, "red")
plt.title("Ridler-Calvard threshold")

plt.subplot(2, 3, 3)
plt.hist(im_gray.ravel(), bins=256)
plt.vlines(thres_otsu, 0, 2000, "red")
plt.title("Otsu's method threshold")

plt.subplot(2, 3, 5)
plt.imshow(mask_ridler, cmap="gray", vmin=0, vmax=255)
plt.title("Image with Ridler-Calvard thresholding")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(mask_otsu, cmap="gray", vmin=0, vmax=255)
plt.title("Image with Otsu's thresholding")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(mask_local, cmap="gray", vmin=0, vmax=255)
plt.title("Image with local thresholding")
plt.axis("off")

plt.tight_layout()
plt.show()

# cv2.imshow("Ridler", mask_ridler)
# cv2.waitKey(0)
# cv2.imshow("Otsu's", mask_otsu)
# cv2.waitKey(0)
# cv2.imshow("local", mask_local)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



