#!/usr/bin/env python3
"""
http://opencvpython.blogspot.fr/2012/06/hi-this-article-is-tutorial-which-try.html
"""
import cv2
import logging

from omr_musicsheet.tools.logger import init_logger
from omr_musicsheet.datasets import get_image_path

logger = logging.getLogger(__name__)


def main():
    im = cv2.imread(str(get_image_path('test.jpg')))
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    logger.info(f"nombre de contours: {len(contours)}")
    cnt = contours[0]
    logger.info(f"nombre de points dans le contour 1: {len(cnt)}")

    cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
    cv2.imshow("contours - borders", im)

    cv2.drawContours(im, contours, -1, (0, 0, 255), -1)
    cv2.imshow("contours - fill", im)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    init_logger(logger)
    main()
