import numpy as np
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


# 1. (b)
matrix = np.array([[1, -0.5, 100],
                   [0.1, 0.9, 50],
                   [0.0015, 0.0015, 1]])
tform = transform.ProjectiveTransform(matrix=matrix)
im = data.text()
transformed_im = transform.warp(im, tform.inverse)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].imshow(im)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(transformed_im)
ax[1].set_title("Transformed")
ax[1].axis("off")

plt.show()
