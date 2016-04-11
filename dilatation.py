import cv2
import numpy as np

filename = "Page_09_Pattern_23.png"
# filename = "Page_09.jpg"

src = cv2.imread(filename)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

cv2.imshow("bw", bw)
cv2.waitKey(0)

# url: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# kernel = np.ones((2,2),np.uint8)
kernel = np.ones((2, 2), np.uint8)
bw = cv2.dilate(bw, kernel, iterations=1)
# cv2.dilate(bw, cv2.ones(2, 2, CV_8UC1))

cv2.imshow("bw", bw)
cv2.waitKey(0)

# url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
horizontal = bw.copy()
# Specify size on horizontal axis
scale = 15  # play with this variable in order to increase/decrease the amount of lines to be detected
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

# url: http://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
# im2, contours, hierarchy =
# contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
print hierarchy

img_contours = src.copy()

cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 1)

cv2.imshow("img_contours", img_contours)
cv2.waitKey(0)


# url: https://github.com/spmallick/learnopencv/blob/master/BlobDetector/blob.py
# Read image
# im = cv2.imread("BlobTest.jpg", cv2.IMREAD_GRAYSCALE)
# im = cv2.imread("Page_09_Pattern_23.png", cv2.IMREAD_GRAYSCALE)
im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 80

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
# params.minInertiaRatio = 0.01
params.minInertiaRatio = 0.33

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(
    im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
