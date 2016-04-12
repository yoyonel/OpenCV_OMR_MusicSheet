import cv2
import numpy as np

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
filename = "Page_09_Pattern_26.png"

src = cv2.imread(filename)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(gray, 150, 700, apertureSize=5)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)
# bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

# bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

minThreshold = 0  #
maxThreshold = 255
ret, bw = cv2.threshold(~gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("bw", bw)
cv2.waitKey(0)
# url: http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
cv2.imwrite("bw.png", bw)

# url: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# cv2.dilate(bw, cv2.ones(2, 2, CV_8UC1))
# kernel = np.ones((2,2),np.uint8)
kernel = np.ones((2, 2), np.uint8)
bw = cv2.dilate(bw, kernel, iterations=1)

cv2.imshow("bw", bw)
cv2.waitKey(0)
cv2.imwrite("bw_after_dilate.png", bw)

# url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
horizontal = bw.copy()
# Specify size on horizontal axis
scale = 20  # play with this variable in order to increase/decrease the amount of lines to be detected
height, width = horizontal.shape[:2]
horizontalsize = width / scale

# Create structure element for extracting horizontal lines through morphology operations
# Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));
# url:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
print horizontalStructure

# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

cv2.imshow("horizontal", horizontal)
cv2.waitKey(0)
cv2.imwrite("extract_horizontal.png", horizontal)

# url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
vertical = bw.copy()
# Specify size on horizontal axis
scale = 15  # play with this variable in order to increase/decrease the amount of lines to be detected
height, width = vertical.shape[:2]
verticalsize = height / scale

# Create structure element for extracting horizontal lines through morphology operations
# Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));
# url:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
print verticalStructure

# Apply morphology operations
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)

cv2.imshow("vertical", vertical)
cv2.waitKey(0)
cv2.imwrite("extract_vertical.png", vertical)

# url: http://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
# im2, contours, hierarchy =
# contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# im2, contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # version 3.1.0
tup_results = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  #
global contours, hierarchy
if len(tup_results) == 3:
    im2, contours, hierarchy = tup_results
    cv2.imshow("image from findContour", im2)
    cv2.waitKey(0)
    cv2.imwrite("extract_contours_image.png", im2)
else:
    contours, hierarchy = tup_results

# url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
print hierarchy


img_contours = src.copy()

cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 1)

cv2.imshow("img_contours", img_contours)
cv2.waitKey(0)
cv2.imwrite("extract_contours.png", img_contours)

