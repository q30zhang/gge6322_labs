import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...

np.set_printoptions(suppress=True)  # disable scientific notation


# 2.
im_oil = cv2.imread("Oil painting.jpg", cv2.IMREAD_COLOR_RGB)
im_highway = cv2.imread("Highway billboard.jpg", cv2.IMREAD_COLOR_RGB)

# manually find the top/bottom-left/right coordinates for the two images
oil_tl = [634, 153]
oil_tr = [1374, 391]
oil_bl = [607, 1134]
oil_br = [1369, 987]

highway_tl = [333, 138]
highway_tr = [743, 122]
highway_bl = [329, 343]
highway_br = [744, 341]

src = np.array([oil_tl, oil_tr, oil_br, oil_bl])
dst = np.array([highway_tl, highway_tr, highway_br, highway_bl])

tform = transform.ProjectiveTransform.from_estimate(src, dst)
if not tform:
    raise RuntimeError(f'Failed estimation: {tform}')  # ensure valid transform
print(tform.params)

im_oil_transformed = transform.warp(im_oil, tform.inverse,
                                    output_shape=im_highway.shape[:2])
im_oil_transformed = cv2.normalize(im_oil_transformed, None, 255, 0,
                                   cv2.NORM_MINMAX, cv2.CV_8U)

mask = np.zeros(im_highway.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [dst], (255, 255, 255))

combined = cv2.add(src1=im_oil_transformed,
                   src2=np.zeros_like(im_oil_transformed),
                   dst=im_highway, mask=mask)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
plt.tight_layout()

ax[0].imshow(im_oil)
ax[0].plot(src[:, 0], src[:, 1], '.r')
ax[0].set_title("Oil painting")
ax[0].axis("off")

ax[1].imshow(im_oil_transformed)
ax[1].plot(dst[:, 0], dst[:, 1], '.r')
ax[1].set_title("Transformed Oil painiting")
ax[1].axis("off")

ax[2].imshow(combined)
ax[2].plot(dst[:, 0], dst[:, 1], '.r')
ax[2].set_title("Combined")
ax[2].axis("off")

plt.show()
