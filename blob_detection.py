import cv2
import numpy as np


# url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
def nothing(x):
    pass

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
filename = "Page_09_Pattern_26.png"

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
params.minArea = 60
params.maxArea = 140

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.075

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
# params.minInertiaRatio = 0.01
params.minInertiaRatio = 0.33


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#
cv2.createTrackbar('minThreshold', 'image', int(params.minThreshold), 255, nothing)
cv2.createTrackbar('maxThreshold', 'image', int(params.maxThreshold), 255, nothing)
#
cv2.createTrackbar('minArea', 'image', int(params.minArea), 1000, nothing)
cv2.createTrackbar('maxArea', 'image', int(params.maxArea), 1000, nothing)
cv2.createTrackbar('minCircularity', 'image', int(params.minCircularity)*1000, 1000, nothing)
#
switch_filterByArea = 'filterByArea\n0 : OFF \n1 : ON'
switch_filterByCircularity = 'filterByCircularity\n0 : OFF \n1 : ON'
switch_filterByConvexity = 'filterByConvexity\n0 : OFF \n1 : ON'
switch_filterByInertia = 'filterByInertia\n0 : OFF \n1 : ON'
cv2.createTrackbar(switch_filterByArea, 'image', params.filterByArea, 1, nothing)
cv2.createTrackbar(switch_filterByCircularity, 'image', params.filterByCircularity, 1, nothing)
cv2.createTrackbar(switch_filterByConvexity, 'image', params.filterByConvexity, 1, nothing)
cv2.createTrackbar(switch_filterByInertia, 'image', params.filterByInertia, 1, nothing)

# cv2.imwrite("extract_circles_notes.png", im_with_keypoints)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    canny_lowThreshold = cv2.getTrackbarPos('canny_lowThreshold', 'image')
    # canny_param2 = cv2.getTrackbarPos('canny_param2', 'image')
    # canny_apertureSize = cv2.getTrackbarPos('canny_apertureSize', 'image')

    params.filterByArea = bool(cv2.getTrackbarPos(switch_filterByArea, 'image'))
    params.filterByCircularity = bool(cv2.getTrackbarPos(switch_filterByCircularity, 'image'))
    params.filterByConvexity = bool(cv2.getTrackbarPos(switch_filterByConvexity, 'image'))
    params.filterByInertia = bool(cv2.getTrackbarPos(switch_filterByInertia, 'image'))

    params.minArea = cv2.getTrackbarPos('minArea', 'image')
    params.maxArea = cv2.getTrackbarPos('maxArea', 'image')
    params.minCircularity = cv2.getTrackbarPos('minCircularity', 'image')/1000.0

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
    cv2.imshow("image", im_with_keypoints)
