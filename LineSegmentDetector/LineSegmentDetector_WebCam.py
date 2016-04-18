#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

# relative module
import video

# built-in module
import sys


def fore_back_ground(img1, img2):
    """Summary
        Dessine img2 dans img1 en créant un mask (binaire) d'image lié à ces intensités de couleurs.
        Le noir (0, 0, 0) est considéré totalement transparent.
        Toute les autres couleurs (à partir d'une certaine intensité => threshold de 10 à 255) sont opaques.

    Args:
        img1 (TYPE): image
        img2 (TYPE): image

    Returns:
        TYPE: image
    """
    # url: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)

    result = img1.copy()
    result[0:rows, 0:cols] = dst

    return result

if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    cap = video.create_capture(fn)
    ls = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    while True:
        flag, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        width, height = gray.shape
        blank_image = np.zeros((width,height,3), np.uint8)

        # Canny Edge Detector
        # thrs1 = cv2.getTrackbarPos('thrs1', 'edge')        
        # thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        # edge = cv2.Canny(blank_image, thrs1, thrs2, apertureSize=5)
        # vis = img.copy()
        # vis = np.uint8(vis/2.)
        # vis[edge != 0] = (0, 255, 0)
        # cv2.imshow('edge', vis)

        # LSD Edge Detector
        # url: http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
        edge = blank_image.copy()

        tup_results = ls.detect(cv2.medianBlur(gray, 5))
        lines = tup_results[0]
        if lines is not None:
            print("lines", len(lines))            
            ls.drawSegments(edge, lines)
            
            # cv2.imshow('edge', edge)    

            vis = img.copy()
            vis = np.uint8(vis/2.)            
            # vis[edge != 0] = (0, 255, 0)
            cv2.imshow('edge', fore_back_ground(vis, edge))

        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            break
    cv2.destroyAllWindows()
