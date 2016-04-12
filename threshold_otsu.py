import cv2
import numpy as np

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
filename = "Page_09_Pattern_23.png"
# filename = "Page_09_Pattern_26.png"

src = cv2.imread(filename)

im = src

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# url: http://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
# "The Study on An Application of Otsu Method in Canny Operator"
# - url: http://www.academypublisher.com/proc/isip09/papers/isip09p109.pdf
# Otsu Thresholding - http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
#
minThreshold = 0  #
maxThreshold = 255
ret, thresh = cv2.threshold(imgray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print 'ret: ', ret

cv2.namedWindow('Thresholding Otsu', cv2.WINDOW_OPENGL)
cv2.imshow('thresh', thresh)
cv2.imwrite("thresh_otsu_{0}.jpg".format(filename[:-4]), thresh)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
