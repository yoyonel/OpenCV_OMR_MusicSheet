#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

'''
Floodfill sample.

Usage:
  floodfill.py [<image>] [<fix_range>] [<connectivity>] [<lo_init>] [<hi_init>]

  Click on the image to set seed point

Keys:
  f     - toggle floating range
  c     - toggle 4/8 connectivity
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import ast


def findContours(img, **params):
    """Summary

    Args:
        img (TYPE): Description
        **params (TYPE): Description

    Returns:
        TYPE: Description
    """
    #
    mode = params.get("mode", cv2.RETR_CCOMP)
    method = params.get("method", cv2.CHAIN_APPROX_SIMPLE)
    #
    # url: http://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
    tup_results = cv2.findContours(img, mode, method)  #
    global contours, hierarchy
    if len(tup_results) == 3:
        im2, contours, hierarchy = tup_results
    else:
        contours, hierarchy = tup_results
    # url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
    # print hierarchy
    return contours, hierarchy


def findContourSheet(src, dst, thickness=2):
    params_findContours = {"mode": cv2.RETR_TREE, "method": cv2.CHAIN_APPROX_SIMPLE}
    contours, hierarchy = findContours(src, **params_findContours)
    # print("hierarchy: ", hierarchy)
    hierarchy = hierarchy[0]
    list_ids_contours_with_hierarchy_level0 = [
        id for id, _ in filter(lambda tup: hierarchy[tup[0]][3] == -1, list(enumerate(hierarchy)))]
    id_contours_with_max_perimeters = max(
        list_ids_contours_with_hierarchy_level0, key=lambda id: cv2.arcLength(contours[id], True))
    print("list_ids_contours_with_hierarchy_level0: ", list_ids_contours_with_hierarchy_level0)
    print("id_contours_with_max_perimeters: ", id_contours_with_max_perimeters)
    cv2.drawContours(dst, contours, id_contours_with_max_perimeters, np.random.randint(255, size=3), thickness)


def findContoursMusicSymbols(src, dst, thickness=2, checkConvexivity=True):
    params_findContours = {"mode": cv2.RETR_CCOMP, "method": cv2.CHAIN_APPROX_SIMPLE}
    contours, hierarchy = findContours(src, **params_findContours)
    nb_contours_without_defects = 0
    for i, contour in enumerate(contours):
        if not(checkConvexivity) or cv2.isContourConvex(contour):
            color_rand = np.random.randint(255, size=3)
            cv2.drawContours(dst, contours, i, color_rand, thickness)

            pts_hull = cv2.convexHull(contour, returnPoints=True)

            # url:
            # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
            hull = cv2.convexHull(contour, returnPoints=False)
            # http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
            defects = cv2.convexityDefects(contour, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    # start = tuple(contour[s][0])
                    # end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    # cv2.line(dst, start, end, [0, 255, 0], 1)
                    # url:
                    # http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/point_polygon_test/point_polygon_test.html
                    true_distance = cv2.pointPolygonTest(pts_hull, far, True)
                    min_distance = 2.5
                    color_circle = (color_rand) if (true_distance > min_distance) else (0, 0, 0)
                    radius_circle = 6 if (true_distance > min_distance) else 1
                    cv2.circle(dst, far, radius_circle, color_circle, -1)
                    print("d: ", d, " - true_distance: ", true_distance)
            else:
                nb_contours_without_defects += 1
    print("nb_contours_without_defects: ", nb_contours_without_defects)


########################################################
if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '../data/fruits.jpg'
    try:
        # url: http://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
        fixed_range = ast.literal_eval(sys.argv[2])
        connectivity = ast.literal_eval(sys.argv[3])
        lo_init = ast.literal_eval(sys.argv[4])
        hi_init = ast.literal_eval(sys.argv[5])
    except:
        fixed_range = True
        connectivity = 4
        lo_init, hi_init = (20, 20)
    print(__doc__)

    img = cv2.imread(fn, True)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    # Denoise image
    # url:
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    cv2.imshow('Denoise', gray)

    minThreshold = 15
    maxThreshold = 255
    #
    # ret, gray = cv2.threshold(gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # #
    bitwise_gray = ~gray
    # bitwise_gray = cv2.medianBlur(bitwise_gray, 3)
    cv2.imshow('bitwise_gray medianBlur', bitwise_gray)

    # gray = cv2.adaptiveThreshold(bitwise_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -2)
    ret, gray = cv2.threshold(bitwise_gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # gray = cv2.fastNlMeansDenoising(gray)
    cv2.imshow('gray', gray)

    findContoursMusicSymbols(gray, img, 1, False)
    #
    findContourSheet(gray, img, 4)

    use_mask = True

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seed_pt = None
    # fixed_range = True
    # connectivity = 4

    def update(dummy=None):
        if seed_pt is None:
            cv2.imshow('floodfill', img)
            return

        flooded = img.copy()

        if use_mask:
            # url:
            # http://stackoverflow.com/questions/7115437/how-to-embed-a-small-numpy-array-into-a-predefined-block-of-a-large-numpy-arra
            x = 1
            y = 1
            mask[x:x + gray.shape[0], y:y + gray.shape[1]] = gray
        else:
            mask[:] = 0

        lo = cv2.getTrackbarPos('lo', 'floodfill')
        hi = cv2.getTrackbarPos('hi', 'floodfill')
        flags = connectivity
        if fixed_range:
            flags |= cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(flooded, mask, seed_pt, (0, 255, 0), (lo,) * 3, (hi,) * 3, flags)
        cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
        # flooded[flooded != (0, 255, 0)] = 0
        cv2.imshow('floodfill', flooded)

    def onmouse(event, x, y, flags, param):
        global seed_pt
        if flags & cv2.EVENT_FLAG_LBUTTON:
            seed_pt = x, y
            print("fixed_range: ", fixed_range)
            update()

    update()
    cv2.setMouseCallback('floodfill', onmouse)
    cv2.createTrackbar('lo', 'floodfill', lo_init, 255, update)
    cv2.createTrackbar('hi', 'floodfill', hi_init, 255, update)

    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
        if ch == ord('f'):
            fixed_range = not fixed_range
            print('using %s range' % ('floating', 'fixed')[fixed_range])
            update()
        if ch == ord('c'):
            connectivity = 12 - connectivity
            print('connectivity =', connectivity)
            update()
    cv2.destroyAllWindows()
