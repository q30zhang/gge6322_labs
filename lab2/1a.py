import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...

# (a)
tform = transform.SimilarityTransform(
    scale=0.5, rotation=3*np.pi/8, translation=(20, 30))
print(tform.params)

im = cv2.imread("arrow.png", cv2.IMREAD_COLOR_RGB)
im = cv2.resize(im, None, fx=3.0, fy=3.0)
transformed_im = transform.warp(im, tform.inverse)

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(transformed_im)
plt.title("Transformed")
plt.axis("off")

plt.show()

# (b)

matrix = np.array([[1, -0.5, 100],
                   [0.1, 0.9, 50],
                   [0.0015, 0.0015, 1]])
tform = transform.ProjectiveTransform(matrix=matrix)
transformed_im = transform.warp(im, tform.inverse)

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(transformed_im)
plt.title("Transformed")
plt.axis("off")

plt.show()
