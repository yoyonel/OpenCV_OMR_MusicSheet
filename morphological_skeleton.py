import cv2
import numpy as np


def showImage(
        img,
        namedWindow="",
        width=640, height=480
):
    cv2.namedWindow(namedWindow, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(namedWindow, width, height)
    cv2.imshow(namedWindow, img)


def morphological_skeleton(img, maxIter=1024):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    height, width = src.shape[:2]
    skel = np.zeros((height, width, 1), np.uint8)
    temp = np.zeros((height, width, 1), np.uint8)

    done = False
    nbIteration = 0
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        cv2.subtract(img, temp, temp)
        cv2.bitwise_or(skel, temp, skel)
        img = eroded
        nbIteration += 1
        done = (cv2.countNonZero(img) == 0) and (nbIteration < maxIter)

    return skel, nbIteration


# filename = "opencv.png"
filename = "rotate_image.png"

src = cv2.imread(filename)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

minThreshold = 127
maxThreshold = 255
ret, img = cv2.threshold(gray, minThreshold, maxThreshold, cv2.THRESH_BINARY)
# cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);

skel, nbIter = morphological_skeleton(img)

showImage(skel, "Skeleton")

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
