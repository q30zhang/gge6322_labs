import cv2
from skimage.morphology import opening, closing
from skimage.morphology import disk
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...

TIMES_APPLYING_OPENING = 6  # can change the number if needed

im = cv2.imread("fruit.PNG", cv2.IMREAD_GRAYSCALE)
_, im = cv2.threshold(im, 255 // 2, 255, cv2.THRESH_BINARY)

# 6.
# using the same binary fruit image from question 5.
opened = [opening(im, footprint=disk(2))]
closed = [closing(im, footprint=disk(2))]

for i in range(TIMES_APPLYING_OPENING - 1):
    # apply opening again on the last "opened" image in the sequence
    opened.append(opening(opened[-1], footprint=disk(2)))
    # apply closing again on the last "closed" image in the sequence
    closed.append(closing(closed[-1], footprint=disk(2)))

plt.rcParams["figure.figsize"] = [10, 10]

for i in range(len(opened)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(opened[i], cmap="gray", vmin=0, vmax=255)
    plt.title(f"Applied opening {i + 1} times")
    plt.axis("off")

for i in range(len(closed)):
    plt.subplot(4, 3, i + 1 + len(opened))
    plt.imshow(closed[i], cmap="gray", vmin=0, vmax=255)
    plt.title(f"Applied closing {i + 1} times")
    plt.axis("off")

plt.tight_layout()
plt.show()
