import numpy as np
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


np.random.seed(6322)
im = (np.random.random(size=(4, 5)) * 256).astype(int)

plt.imshow(im, cmap="gray", vmin=0, vmax=255)
plt.title("A random gray-level 8 bit 4×5 image")
plt.axis("off")
plt.show()
