import cv2
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


im = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
im_eq = cv2.equalizeHist(im)

plt.subplot(1, 2, 1)
plt.imshow(im, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(im_eq, cmap="gray", vmin=0, vmax=255)
plt.title("Equalized")
plt.axis("off")

plt.show()
