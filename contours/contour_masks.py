# url: http://opencvpython.blogspot.fr/2012/06/hi-this-article-is-tutorial-which-try.html
import numpy as np
import cv2

im = cv2.imread('balls.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("imgray", imgray)
cv2.waitKey(0)

# bug: marrant la valeur de threshold est sensible ...
# surement lie a la compression ou format de l'image
minThreshold = 121  # valeur dans le tuto = 127
maxThreshold = 255
ret, thresh = cv2.threshold(imgray, minThreshold, maxThreshold, 0)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print len(contours)

for h, cnt in enumerate(contours):
    mask = np.zeros(imgray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    mean = cv2.mean(im, mask=mask)
    cv2.imshow("Masks", mask)
    cv2.waitKey(0)
