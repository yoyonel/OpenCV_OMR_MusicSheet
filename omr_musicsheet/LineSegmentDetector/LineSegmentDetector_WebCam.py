#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

# Python 2/3 compatibility
# from __future__ import print_function

import cv2
import numpy as np

# relative module
import omr_musicsheet.LineSegmentDetector.video as video

# built-in module
import sys


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles
        # than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h),
                      (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


def main():
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('edge')
    # cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    # cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    cap = video.create_capture(fn)
    ls = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        flag, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        width, height = gray.shape
        blank_image = np.zeros((width, height, 3), np.uint8)

        # LSD Edge Detector
        edge = blank_image
        # url: http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
        # on detecte les lignes dans l'image des gris de la webcam
        # on applique un filtre blur median dessus avant
        # (pour stabiliser l'image et les discontinuites)
        tup_results = ls.detect(cv2.medianBlur(gray, 5))
        # extract lines from result
        lines = tup_results[0]
        # Si on trouve des lignes
        if lines is not None:
            print("lines", len(lines))
            # On les affiche
            # Ps: edge est l'image de destination
            # qui est une image RGB (car blank_image est RGB)
            # Il faut une image RGB,
            # car drawSegment dessine dans une RGB (lignes rouges)
            ls.drawSegments(edge, lines)

            # extract red component of edge image
            # maintenant edge est au format (widht, height, 1)
            # urls:
            # - http://knowpapa.com/greyscale-opencv/
            # - http://knowpapa.com/opencv-rgb-split/
            edge = edge[:, :, 2]

            # copy de l'image issue de la webcam
            vis = img.copy()
            # on réduit par 2 l'intensite de l'image (webcam)
            vis = np.uint8(vis / 2.)

            # Utilisation d'edge comme un masque
            # Fusion d'edge et visu avec priorite edge
            # On remplace les pixels actifs (anciennement rouge) d'edge
            # par des pixels vers dans vis
            # => les lignes de detections seront vertes
            vis[edge != 0] = (0, 255, 0)

            cv2.imshow('edge', vis)

        #
        found, w = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32),
                                        scale=1.05)
        print(found)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found)))
        cv2.imshow('edge', img)

        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
