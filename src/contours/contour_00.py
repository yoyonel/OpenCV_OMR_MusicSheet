# url: http://opencvpython.blogspot.fr/2012/06/hi-this-article-is-tutorial-which-try.html
import numpy as np
import cv2

im = cv2.imread('test.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print "nombre de contours: ", len(contours)
cnt = contours[0]
print "nombre de points dans le contour 1: ", len(cnt)

cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
cv2.imshow("contours - borders", im)
cv2.waitKey(0)

cv2.drawContours(im, contours, -1, (0, 0, 255), -1)
cv2.imshow("contours - fill", im)
cv2.waitKey(0)
