import cv2

# a.
im = cv2.imread("cat.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imwrite("../lab3/cat_gray.png", im_gray)

# b.
im_resized = cv2.resize(im_gray, None, fx=0.5, fy=0.5)
cv2.imshow("Half-sized gray level image", im_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()  # press any key to close the window
