import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


# b.
im_test = np.array(
    [[176, 94 , 201, 219],
     [37 , 161, 16 , 88 ],
     [71 , 129, 177, 81 ],
     [41 , 198, 107, 19 ]]
)

print("Flip:")
print(np.flip(im_test, axis=0))

print("\nFlop:")
print(np.flip(im_test, axis=1))

print("\nInvert:")
print(255 - im_test)

print("\nRotate by 90 degrees clockwise:")
print(np.rot90(im_test, -1))


# c.
im = cv2.imread("cat.jpg", cv2.IMREAD_COLOR_RGB)

plt.subplot(2, 2, 1)
plt.imshow(np.flip(im, axis=0))
plt.title("Flip")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(np.flip(im, axis=1))
plt.title("Flop")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(255 - im)
plt.title("Invert")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(np.rot90(im, -1))
plt.title("Rotate")
plt.axis("off")

plt.show()
