import numpy as np
from scipy import ndimage as ndi
from skimage import data
import matplotlib.pyplot as plt

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


im = data.camera().astype("int32")
gx = ndi.sobel(im, axis=0)
gy = ndi.sobel(im, axis=1)
g_mag = np.abs(gx) + np.abs(gy)
g_phase = np.arctan2(gy, gx)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(im, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gx, cmap="gray")
plt.title("Sobel-X")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gy, cmap="gray")
plt.title("Sobel-Y")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(g_mag, cmap="gray")
plt.title("Magnitude")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(g_phase, cmap="gray")
plt.title("Phase")
plt.axis("off")

plt.tight_layout()
plt.show()
