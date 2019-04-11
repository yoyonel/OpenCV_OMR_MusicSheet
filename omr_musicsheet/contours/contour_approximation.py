"""
http://opencvpython.blogspot.fr/2012/06/hi-this-article-is-tutorial-which-try.html
"""
from pathlib import Path

import cv2
import logging

from omr_musicsheet.datasets import get_module_path_datasets
from omr_musicsheet.tools.logger import init_logger

logger = logging.getLogger(__name__)


def main():
    fn = Path(get_module_path_datasets()) / 'balls.png'

    im = cv2.imread(str(fn))
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgray", imgray)

    # bug: marrant la valeur de threshold est sensible ...
    # surement lie a la compression ou format de l'image
    minThreshold = 121  # valeur dans le tuto = 127
    maxThreshold = 255
    ret, thresh = cv2.threshold(imgray, minThreshold, maxThreshold, 0)
    cv2.imshow("thresh", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    logger.info(f"len(contours): {len(contours)}")
    # coef_approx = 0.0  # suit exactement les contours detectes
    coef_approx = 0.01  # suit (presque) les contours detectes
    # coef_approx = 0.10  # affiche un rectanble englobant
    for h, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, coef_approx * cv2.arcLength(cnt, True),
                                  True)
        logger.info(f"len(approx): {len(approx)}")
        cv2.drawContours(im, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("approx", im)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    init_logger(logger)
    main()
