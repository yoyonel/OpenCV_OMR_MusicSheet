import cv2
import numpy as np

#
filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
# filename = "Page_09_Pattern_26.png"

src = cv2.imread(filename)

im = src

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

imgray = cv2.medianBlur(imgray, 3)

# url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
def nothing(x):
    pass

height, width = im.shape[:2]

# url: http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
# Create a black image, a window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

d = 9
sigmaColor = 75
sigmaSpace = 75

# create trackbars for color change
cv2.createTrackbar('d', 'image', d, 100, nothing)
cv2.createTrackbar('sigmaColor', 'image', sigmaColor, 100, nothing)
cv2.createTrackbar('sigmaSpace', 'image', sigmaSpace, 100, nothing)
#
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    d = cv2.getTrackbarPos('d', 'image')
    sigmaColor = cv2.getTrackbarPos('sigmaColor', 'image')
    sigmaSpace = cv2.getTrackbarPos('sigmaSpace', 'image')

    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img = imgray
    else:
        imgray_filtered = cv2.bilateralFilter(imgray, d, sigmaColor, sigmaSpace)

        minThreshold = 121  # valeur dans le tuto = 127
        maxThreshold = 255
        ret, thresh = cv2.threshold(imgray_filtered, minThreshold, maxThreshold, 0)

        img = thresh

    cv2.imshow('image', img)

cv2.destroyAllWindows()
