import cv2
import numpy as np
from skimage.morphology import opening, closing
from skimage.morphology import disk
from skimage.measure import label, regionprops


im = cv2.imread("fruit.PNG", cv2.IMREAD_GRAYSCALE)
_, im = cv2.threshold(im, 255 // 2, 255, cv2.THRESH_BINARY)

# 5. (a)
# first, remove white noise outside of the fruits
opened = opening(im, footprint=disk(2))
# then, close up some black region in some fruits (apples especially)
closed = closing(opened, footprint=disk(3))

cv2.imshow("5. (a) - press any key to continue", closed)
cv2.waitKey(0)  # press any key to continue...


# 5. (b)
# remove the white border frame to avoid being labeled as another region
truncated = closed[5:-5, 5:-5]
# label the image by separate connected regions
labeled = label(truncated, connectivity=2)
# get geometric properties of the labeled regions
properties = regionprops(labeled)

# prepare for classification
region_eccentricities = []
region_areas = []
for region in properties:
    region_eccentricities.append(region.eccentricity)
    region_areas.append(region.area)
mean_eccentricity = np.mean(region_eccentricities)
mean_area = np.mean(region_areas)

# prepare the image for showing the labeled regions
im_with_labels = cv2.cvtColor(truncated, cv2.COLOR_GRAY2BGR)

for region in properties:
    if region.eccentricity > mean_eccentricity:
        fruit_type = "banana"  # long, non-circular: banana
    elif region.area < mean_area:
        fruit_type = "apple"  # round and small: apple
    else:
        fruit_type = "orange"  # round and large: orange
    # show frame around each region
    x1, y1, x2, y2 = region.bbox
    # show fruit type of each region
    x, y = [int(n) for n in region.centroid]
    text_size, _ = cv2.getTextSize(fruit_type, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x += text_size[1] // 2
    y -= text_size[0] // 2  # adjust for matching the region centroid with text
    im_with_labels = cv2.rectangle(im_with_labels, (y1, x1), (y2, x2),
                                   (0, 0, 255), 2)
    im_with_labels = cv2.putText(im_with_labels, fruit_type, (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow("5. (b) - press any key to close", im_with_labels)
cv2.waitKey(0)  # press any key to close
