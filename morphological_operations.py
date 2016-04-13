import cv2
import numpy as np
import math

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
# filename = "Page_09_Pattern_24.png"
# filename = "Page_09_Pattern_25.png"
# filename = "Page_09_Pattern_26.png"
#
# filename = "Page_09_Pattern_23_rot90.png"
filename = "rotate_image.png"

# cv2.namedWindow("window", cv2.WINDOW_NORMAL | cv2.WINDOW_OPENGL)
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

src = cv2.imread(filename)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 150, 700, apertureSize=5)
cv2.imshow("window", edges)
cv2.waitKey(0)
# bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

minThreshold = 15  #
maxThreshold = 250
# ret, bw = cv2.threshold(~gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# url: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# cv2.dilate(bw, cv2.ones(2, 2, CV_8UC1))
kernel = np.ones((2, 2), np.uint8)
bw = cv2.dilate(bw, kernel, iterations=1)

cv2.imshow("window", bw)
cv2.waitKey(0)
cv2.imwrite("bw_after_dilate.png", bw)

# url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
horizontal = bw.copy()
# Specify size on horizontal axis
scale = 40  # play with this variable in order to increase/decrease the amount of lines to be detected
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

cv2.imshow("window", horizontal)
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

cv2.imshow("window", vertical)
cv2.waitKey(0)
cv2.imwrite("extract_vertical.png", vertical)

# url: http://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
# im2, contours, hierarchy =
# contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# im2, contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # version 3.1.0
# tup_results = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  #
tup_results = cv2.findContours(horizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)  #
global contours, hierarchy
if len(tup_results) == 3:
    im2, contours, hierarchy = tup_results
    cv2.imshow("window", im2)
    cv2.waitKey(0)
    cv2.imwrite("extract_contours_image.png", im2)
else:
    contours, hierarchy = tup_results

# url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
print hierarchy


img_contours = src.copy()
img_contours_2 = np.zeros((height,width,3), np.uint8)

# for i, c in enumerate(contours):
#     peri = cv2.arcLength(c, True)
#     cv2.drawContours(img_contours, c, i, np.random.randint(255, size=3), 1)
minPerimeter = 1000
thickness = 1
for i, contour in enumerate(contours):
    perimeter = cv2.arcLength(contour, True)
    # print "contours - perimeter= ", perimeter
    if perimeter >= minPerimeter:
        color_rand = np.random.randint(255, size=3)
        cv2.drawContours(img_contours, contours, i, color_rand, thickness)
        cv2.drawContours(img_contours_2, contours, i, color_rand, thickness)

        contour_flat = [item for sublist in contour for item in sublist]
        for point in contour_flat:
            cv2.circle(img_contours, tuple(point), 5, 255 - color_rand)
        # contour_gradients = np.gradient(contour_flat)
        # print "gradient: ", np.gradient(contour_gradients)

        # url: https://github.com/Itseez/opencv/blob/master/samples/python/fitline.py
        # url: http://stackoverflow.com/questions/14184147/detect-lines-opencv-in-object
        # # then apply fitline() function
        # # [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        # [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_HUBER, 0, 0.01, 0.01)
        # # Now find two extreme points on the line to draw line
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((width-x)*vy/vx)+y)

        # # Finally draw the line
        # cv2.line(img_contours, (width-1, righty), (0, lefty), (255, 0, 0), 1)
cv2.imshow("window", img_contours)
cv2.waitKey(0)

minLineLength = 0
img_contours_2 = cv2.Canny(img_contours_2, 150, 700, apertureSize=5)
cv2.imshow("window", img_contours_2)
cv2.waitKey(0)

lines = cv2.HoughLinesP(img_contours_2, rho=1,
                        theta=math.pi / 180, threshold=70,
                        minLineLength=50, maxLineGap=25
                        )
for line in lines:
    # print line
    x1, y1, x2, y2 = line[0]
    cv2.line(img_contours_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("window", img_contours_2)
cv2.waitKey(0)

cv2.imwrite("extract_contours.png", img_contours)
