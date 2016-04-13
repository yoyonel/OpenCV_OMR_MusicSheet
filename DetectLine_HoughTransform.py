import cv2
import numpy as np
import math

img = cv2.imread('Page_09_Pattern_23.png')
cv2.imshow("img", img)
cv2.waitKey(0)

img_in = img

gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)
img_out = gray

img_in = img_out

# url: http://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholdimg_ing.html#gsc.tab=0
ret, thresh1 = cv2.threshold(img_in, 127, 255, cv2.THRESH_BINARY)
#thresh1 = cv2.adaptiveThreshold(img_in, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("thresh1", thresh1)
cv2.waitKey(0)
img_out = thresh1

img_in = img_out

edges = cv2.Canny(img_in, 150, 700, apertureSize=5)
cv2.imshow("edges", edges)
cv2.waitKey(0)
img_out = edges

img_in = img_out
img_out = img

minLineLength = 0
# lines = cv2.HoughLines(img_in, 1, np.pi / 180, minLineLength)
lines = cv2.HoughLinesP(img_in, rho=1,
                        theta=math.pi / 180, threshold=70,
                        minLineLength=50, maxLineGap=25
                        )
# url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# print lines
for line in lines:
    # print line
    x1, y1, x2, y2 = line[0]
    cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("houghlines3", img_out)
cv2.waitKey(0)