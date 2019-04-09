"""Summary

Attributes:
    filename (str): Description
    im (TYPE): Description
    img_contours (TYPE): Description
    img_for_contours (TYPE): Description
    imgray (TYPE): Description
    maxThreshold (int): Description
    minThreshold (int): Description
    src (TYPE): Description
"""
from random import randint

import cv2
import numpy as np


def findContours(in_img):
    """Summary

    Args:
        in_img (TYPE): Description

    Returns:
        TYPE: Description
    """
    # tup_results = cv2.findContours(in_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  #
    # tup_results = cv2.findContours(in_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #
    tup_results = cv2.findContours(in_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)  #

    contours, hierarchy = tup_results[len(tup_results) == 3:]

    # url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
    print(hierarchy)

    return contours, hierarchy


def draw_contours(out_img, contours, color=(255, 0, 0), thickness=3):
    """Summary

    Args:
        out_img (TYPE): Description
        contours (TYPE): Description
        color (tuple, optional): Description
        thickness (int, optional): Description

    Returns:
        TYPE: Description
    """
    cv2.drawContours(out_img, contours, -1, color, thickness)


def draw_contours_2(out_img, contours, thickness=3):
    """Summary

    Args:
        out_img (TYPE): Description
        contours (TYPE): Description
        thickness (int, optional): Description

    Returns:
        TYPE: Description
    """
    # url: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.randint.html
    for i, contour in enumerate(contours):
        cv2.drawContours(out_img, contours, i,
                         color=[randint(0, 255), randint(0, 255), randint(0, 255)],
                         thickness=thickness)


def draw_contours_3(out_img, contours, thickness=3, minPerimeter=10):
    """Summary

    Args:
        out_img (TYPE): Description
        contours (TYPE): Description
        thickness (int, optional): Description
        minPerimeter (int, optional): Description

    Returns:
        TYPE: Description
    """
    # url: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.randint.html
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        # print "contours - perimeter= ", perimeter
        if perimeter >= minPerimeter:
            cv2.drawContours(out_img, contours, i,
                             color=[randint(0, 255), randint(0, 255),
                                    randint(0, 255)],
                             thickness=thickness)


def main():
    #
    # filename = "Page_09_HD.jpg"
    # filename = "Page_09.jpg"
    #
    filename = "Page_09_Pattern_23.png"
    # filename = "Page_09_Pattern_26.png"

    src = cv2.imread(filename)

    im = src

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    imgray = ~imgray

    # url: http://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    # "The Study on An Application of Otsu Method in Canny Operator"
    # - url: http://www.academypublisher.com/proc/isip09/papers/isip09p109.pdf
    # Otsu Thresholding - http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
    #
    minThreshold = 0  #
    maxThreshold = 255
    ret, thresh = cv2.threshold(imgray, minThreshold, maxThreshold,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('ret: ', ret)

    img_for_contours = thresh.copy()
    contours, hierarchy = findContours(img_for_contours)
    # img_contours = src.copy()
    height, width = src.shape[:2]
    img_contours = np.zeros((height, width, 3), np.uint8)
    # draw_contours(img_contours, contours)
    # draw_contours_2(img_contours, contours, thickness=2)
    draw_contours_3(img_contours, contours, thickness=2, minPerimeter=10)

    # cv2.namedWindow('Thresholding Otsu', cv2.WINDOW_OPENGL)
    # cv2.namedWindow('Thresholding Otsu + findContours', cv2.WINDOW_OPENGL)
    cv2.namedWindow('Thresholding Otsu', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Thresholding Otsu + findContours', cv2.WINDOW_NORMAL)

    cv2.imshow('thresh otsu', thresh)
    cv2.imshow('thresh_otsu+findContours', img_contours)

    cv2.imwrite("thresh_otsu_{0}.jpg".format(filename[:-4]), thresh)
    cv2.imwrite("thresh_otsu_findContours_{0}.jpg".format(filename[:-4]),
                img_contours)

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    main()
