import numpy as np
import cv2

# filename = "Page_09_Pattern_26.png"
filename = "Page_09_HD.jpg"

im = cv2.imread(filename)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# bug: marrant la valeur de threshold est sensible ...
# surement lie a la compression ou format de l'image
minThreshold = 121  # valeur dans le tuto = 127
maxThreshold = 255


# url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
def nothing(x):
    pass

height, width = im.shape[:2]

# url: http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
# Create a black image, a window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# create trackbars for color change
cv2.createTrackbar('minThreshold', 'image', 128, 255, nothing)
cv2.createTrackbar('maxThreshold', 'image', 255, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    minThreshold = cv2.getTrackbarPos('minThreshold', 'image')
    maxThreshold = cv2.getTrackbarPos('maxThreshold', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img = imgray
    else:
        print minThreshold, maxThreshold
        ret, thresh = cv2.threshold(imgray, minThreshold, maxThreshold, 0)
        img = thresh

    cv2.imshow('image', img)

cv2.destroyAllWindows()
