import cv2
import numpy as np

filename = "Page_09_HD.jpg"


src = cv2.imread(filename)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

minThreshold = 0  #
maxThreshold = 255
bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

cv2.imshow("bw", bw)
cv2.waitKey(0)
cv2.imwrite("bw.png", bw)
