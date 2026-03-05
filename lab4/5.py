import numpy as np
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
import cv2
import functools
import timeit
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")  # due to a bug on MacOS, just ignore it...


# 5. a.
im = cv2.imread("book.jpg", cv2.IMREAD_COLOR_RGB)


# 5. b.
tform = AffineTransform(scale=(1.3, 1.1), rotation=0.3, shear=0.7,
                        translation=(500, 0))
im_transformed = warp(im, tform.inverse, output_shape=(450, 800))
im_transformed = (im_transformed * 255).astype(np.uint8)
cv2.imwrite("transformed.jpg", cv2.cvtColor(im_transformed, cv2.COLOR_RGB2BGR))


# 5. c.
im_g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
im_t_g = cv2.cvtColor(im_transformed, cv2.COLOR_RGB2GRAY)

methods = ["CENSURE", "ORB", "SIFT"]
results_1 = []
results_2 = []
labeled_1 = []
labeled_2 = []

try:
    censure = cv2.xfeatures2d.StarDetector_create()
except AttributeError:
    import subprocess
    import sys
    if subprocess.check_call([sys.executable, "-m", "pip", "install",
                              "opencv-contrib-python"]) == 0:
        censure = cv2.xfeatures2d.StarDetector_create()
    else:
        raise ModuleNotFoundError("please install opencv-contrib-python...")

detectors = [
        censure,
        cv2.ORB_create(),
        cv2.SIFT_create()
    ]

fig_1 = plt.figure(1, figsize=(12, 8))
fig_1.canvas.manager.set_window_title("Question 5. c.")

for i in range(3):
    try:
        results_1.append(detectors[i].detectAndCompute(im_g, None))
        results_2.append(detectors[i].detectAndCompute(im_t_g, None))
    except:
        results_1.append((detectors[i].detect(im_g, None), None))
        results_2.append((detectors[i].detect(im_t_g, None), None))
    labeled_1.append(
        cv2.drawKeypoints(im_g, results_1[-1][0], im.copy(),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    labeled_2.append(
        cv2.drawKeypoints(im_t_g, results_2[-1][0], im_transformed.copy(),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    plt.subplot(2, 3, i + 1)
    plt.imshow(labeled_1[-1])
    plt.axis("off")
    plt.title(f"{methods[i]} descriptor on oringal image")

    plt.subplot(2, 3, i + 4)
    plt.imshow(labeled_2[-1])
    plt.axis("off")
    plt.title(f"{methods[i]} descriptor on transformed image")

plt.tight_layout()
plt.show()


# 5. d.
n_matches = 10

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
matches_orb = bf_orb.knnMatch(results_1[1][1], results_2[1][1], k=2)
matches_orb_good = []
for m, n in matches_orb:
    if m.distance < 0.75 * n.distance:
        matches_orb_good.append([m])
labeled_orb = cv2.drawMatchesKnn(im, results_1[1][0],
                                 im_transformed, results_2[1][0],
                                 matches_orb_good, None, flags=2)

bf_sift = cv2.BFMatcher()
matches_sift = bf_sift.knnMatch(results_1[2][1], results_2[2][1], k=2)
matches_sift_good = []
for m, n in matches_sift:
    if m.distance < 0.75 * n.distance:
        matches_sift_good.append([m])
labeled_sift = cv2.drawMatchesKnn(im, results_1[2][0],
                                  im_transformed, results_2[2][0],
                                  matches_sift_good, None, flags=2)

fig_2 = plt.figure(2, figsize=(12, 10))
fig_2.canvas.manager.set_window_title("Question 5. d.")

plt.subplot(2, 1, 1)
plt.imshow(labeled_orb)
plt.axis("off")
plt.title("ORB matching")

plt.subplot(2, 1, 2)
plt.imshow(labeled_sift)
plt.axis("off")
plt.title("SIFT matching")

plt.tight_layout()
plt.show()


# 5. e
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


@time_exec
def match_descriptors_orb():
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf_orb.knnMatch(results_1[1][1], results_2[1][1], k=2)

@time_exec
def match_descriptors_sift():
    bf_sift = cv2.BFMatcher()
    bf_sift.knnMatch(results_1[2][1], results_2[2][1], k=2)

putative_match_ratio_orb = len(matches_orb_good) / len(results_1[1][0])
putative_match_ratio_sift = len(matches_sift_good) / len(results_1[2][0])

print(f"Putative Match Ratio for ORB: {putative_match_ratio_orb}")
match_descriptors_orb()

print(f"Putative Match Ratio for SIFT: {putative_match_ratio_sift}")
match_descriptors_sift()


# 5. f
im_1 = cv2.imread("IMG_1.jpg", cv2.IMREAD_COLOR_RGB)
im_2 = cv2.imread("IMG_2.jpg", cv2.IMREAD_COLOR_RGB)
im_1_g = cv2.cvtColor(im_1, cv2.COLOR_RGB2GRAY)
im_2_g = cv2.cvtColor(im_2, cv2.COLOR_RGB2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(im_1_g, None)
kp2, des2 = sift.detectAndCompute(im_2_g, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])

labeled_sift = cv2.drawMatchesKnn(im_1, kp1,
                                  im_2, kp2,
                                  good, None, flags=2)

fig_3 = plt.figure(3, figsize=(12, 4))
fig_3.canvas.manager.set_window_title("Question 5. f.")

plt.imshow(labeled_sift)
plt.axis("off")
plt.title("SIFT matching")
plt.show()
