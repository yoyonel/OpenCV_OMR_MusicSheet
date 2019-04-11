import cv2
import numpy as np

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
filename = "Page_09_Pattern_26.png"

src = cv2.imread(filename)

im = src

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# url: http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html#gsc.tab=0
# radius_blur = 3
# kernel_size_blur = (radius_blur, radius_blur)
# imgray = cv2.GaussianBlur(imgray, kernel_size_blur, 2, 1)
# bilateralFilter_size = 75
# imgray = cv2.bilateralFilter(imgray, 9, bilateralFilter_size, bilateralFilter_size)
# imgray = cv2.medianBlur(imgray, 9)

minThreshold = 121  # valeur dans le tuto = 127
maxThreshold = 255
ret, thresh = cv2.threshold(imgray, minThreshold, maxThreshold, 0)
cv2.imshow('thresh', thresh)

max_lowThreshold = 200
#
canny_lowThreshold = max_lowThreshold
canny_ratio = 3
canny_apertureSize = 3


# url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
def nothing(x):
    pass

height, width = im.shape[:2]

# url: http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
# Create a black image, a window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 320, 200)

# create trackbars for color change
cv2.createTrackbar('canny_lowThreshold', 'image', canny_lowThreshold, max_lowThreshold, nothing)
# cv2.createTrackbar('canny_ratio', 'image', canny_param2, canny_param2*10, nothing)
# cv2.createTrackbar('canny_apertureSize', 'image', canny_apertureSize, canny_apertureSize, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    canny_lowThreshold = cv2.getTrackbarPos('canny_lowThreshold', 'image')
    # canny_param2 = cv2.getTrackbarPos('canny_param2', 'image')
    # canny_apertureSize = cv2.getTrackbarPos('canny_apertureSize', 'image')

    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img = imgray
    else:
        edges = cv2.Canny(imgray, canny_lowThreshold, canny_lowThreshold * canny_ratio, apertureSize=canny_apertureSize)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print len(contours)
        # coef_approx = 0.0  # suit exactement les contours detectes
        coef_approx = 0.01  # suit (presque) les contours detectes
        # coef_approx = 0.10  # affiche un rectanble englobant
        for h, cnt in enumerate(contours):
            approx = cv2.approxPolyDP(cnt, coef_approx * cv2.arcLength(cnt, True), True)
            print len(approx)
            cv2.drawContours(edges, [approx], -1, (255, 0, 0), 10)

        img = edges

    cv2.imshow('image', img)

cv2.destroyAllWindows()
