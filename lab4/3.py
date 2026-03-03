import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import timeit
import functools
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


# 3. a.
image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=0.1)
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)

blobs_list = [[], blobs_log, blobs_dog, blobs_doh]
colors = ['', 'yellow', 'lime', 'red']
titles = ['Original Image', 'Laplacian of Gaussian',
          'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
axs = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    axs[idx].set_title(title)
    axs[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        axs[idx].add_patch(c)
for ax in axs:
    ax.axis("off")

ax_nav = fig.add_subplot(axes[1,0].get_gridspec()[1, :])
ax_nav.text(0, 0.5, "please close this window, and it will start executing "
                    "100 loops of each of the three methods. It will take "
                    "some time... (about 1 minute on my laptop).")
ax_nav.axis("off")


plt.tight_layout()
plt.show()


# 3. b.
def time_exec(func, *, number=100):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        for _ in range(number):
            result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"finished {func.__name__} for {number} loops, " +
              f"average execution time: {(end - start) / number:.6f} s.")
        return result
    return wrapper


image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

# laplacian of gaussian
@time_exec
def laplacian_of_gaussian(image_gray):
    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=0.1)
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# difference of gaussian
@time_exec
def difference_of_gaussian(image_gray):
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

# determinant of hessian
@time_exec
def determinant_of_hessian(image_gray):
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)


laplacian_of_gaussian(image_gray)
difference_of_gaussian(image_gray)
determinant_of_hessian(image_gray)
