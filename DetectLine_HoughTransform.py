import cv2
import numpy as np

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

minLineLength = 300
# lines = cv2.HoughLines(img_in,1,np.pi/180,200)
lines = cv2.HoughLines(img_in, 1, np.pi / 180, minLineLength)
height, width = img_in.shape[:2]
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + width * (-b))
    y1 = int(y0 + height * (a))
    x2 = int(x0 - width * (-b))
    y2 = int(y0 - height * (a))
    #
    cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("houghlines3", img_out)
cv2.waitKey(0)

minLineLength = 200
maxLineGap = 10
lines = cv2.HoughLinesP(img_in, 1, np.pi / 180, 100, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(edges, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("HoughLinesP", edges)
cv2.waitKey(0)
