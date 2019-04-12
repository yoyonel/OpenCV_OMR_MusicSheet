"""
http://opencvpython.blogspot.fr/2012/06/hi-this-article-is-tutorial-which-try.html
"""
import cv2
import logging
#
from omr_musicsheet.datasets import get_image_path
from omr_musicsheet.tools.logger import init_logger

logger = logging.getLogger(__name__)


def main():
    im = cv2.imread(str(get_image_path('balls.png')))
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgray", imgray)

    # bug: marrant la valeur de threshold est sensible ...
    # surement lie a la compression ou format de l'image
    min_threshold = 121  # valeur dans le tuto = 127
    max_threshold = 255
    ret, thresh = cv2.threshold(imgray, min_threshold, max_threshold, 0)
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
        print(f"len(approx): {len(approx)}")
        cv2.drawContours(im, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("approx", im)

    # url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
    def nothing(_x):
        pass

    # height, width = im.shape[:2]

    # Create a black image, a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('minThreshold', 'image', 1, 255, nothing)
    cv2.createTrackbar('maxThreshold', 'image', 1, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        min_threshold = cv2.getTrackbarPos('minThreshold', 'image')
        max_threshold = cv2.getTrackbarPos('maxThreshold', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img = imgray
        else:
            logger.info(
                f"minThreshold={min_threshold}, maxThreshold={max_threshold}")
            ret, thresh = cv2.threshold(imgray, min_threshold, max_threshold, 0)
            img = thresh

        cv2.imshow('image', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    init_logger(logger)
    main()
