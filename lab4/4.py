from skimage.feature import corner_harris, corner_shi_tomasi
from skimage.feature import corner_subpix, corner_peaks
from skimage import data
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


image = data.camera()

coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
coords_subpix = corner_subpix(image, coords, window_size=13)

coords2 = corner_peaks(corner_shi_tomasi(image), min_distance=5, threshold_rel=0.02)
coords_subpix2 = corner_subpix(image, coords2, window_size=13)

fig, axs = plt.subplots(1, 2, figsize=(12, 7))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.plot(coords[:, 1], coords[:, 0], label="peaks",
         color='cyan', marker='o', linestyle='None',markersize=6)
plt.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r',
         label="subpix", markersize=15)
plt.title("Harris corner detection")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image, cmap=plt.cm.gray)
plt.plot(coords2[:, 1], coords2[:, 0], label="peaks",
         color='cyan', marker='o', linestyle='None', markersize=6)
plt.plot(coords_subpix2[:, 1], coords_subpix2[:, 0], '+r',
         label="subpix", markersize=15)
plt.title("Shi-Tomasi corner detection")
plt.axis("off")

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout()
plt.show()
