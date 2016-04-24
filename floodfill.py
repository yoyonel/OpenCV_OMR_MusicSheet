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


def findContourSheet(src, dst, tup_contours, thickness=2):
    # params_findContours = {"mode": cv2.RETR_TREE, "method": cv2.CHAIN_APPROX_SIMPLE}
    # contours, hierarchy = findContours(src, **params_findContours)
    contours, hierarchy = tup_contours

    # print("hierarchy: ", hierarchy)
    hierarchy = hierarchy[0]
    list_ids_contours_with_hierarchy_level0 = [
        id for id, _ in filter(lambda tup: hierarchy[tup[0]][3] == -1, list(enumerate(hierarchy)))]
    list_ids_contours_with_hierarchy_level0 = sorted(
        list_ids_contours_with_hierarchy_level0,
        key=lambda id: cv2.arcLength(contours[id], True)
    )
    # id_contours_with_max_perimeters = max(
    #     list_ids_contours_with_hierarchy_level0, key=lambda id: cv2.arcLength(contours[id], True))
    id_contours_with_max_perimeters = list_ids_contours_with_hierarchy_level0[-1]
    print("list_ids_contours_with_hierarchy_level0: ", list_ids_contours_with_hierarchy_level0)
    print("id_contours_with_max_perimeters: ", id_contours_with_max_perimeters)
    print("max arclength: ", cv2.arcLength(contours[id_contours_with_max_perimeters], True))

    # cv2.drawContours(dst, contours, id_contours_with_max_perimeters, np.random.randint(255, size=3), thickness)
    cv2.drawContours(dst, contours, id_contours_with_max_perimeters, (255, 0, 0), thickness)

    return id_contours_with_max_perimeters


def findContoursMusicSymbols(src, dst, tup_contours, thickness=2, checkConvexivity=True, exceptId=-1):
    # params_findContours = {"mode": cv2.RETR_CCOMP, "method": cv2.CHAIN_APPROX_SIMPLE}
    # contours, hierarchy = findContours(src, **params_findContours)
    contours, hierarchy = tup_contours

    nb_contours_without_defects = 0
    print("exceptId: ", exceptId)
    for i, contour in enumerate(contours):
        # if i is not exceptId and cv2.arcLength(contour, True) < 3000:
        if i is not exceptId:
            if not(checkConvexivity) or cv2.isContourConvex(contour):
                # color_rand = np.random.randint(255, size=3)
                color_rand = (0, 0, 0)
                cv2.drawContours(dst, contours, i, color_rand, thickness)
                print("i, color_rand, arcLength: ", i, color_rand, cv2.arcLength(contour, True))

                # pts_hull = cv2.convexHull(contour, returnPoints=True)
                # url:
                # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
                # hull = cv2.convexHull(contour, returnPoints=False)
                # http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
                # defects = cv2.convexityDefects(contour, hull)

                # if defects is not None:
                #     for i in range(defects.shape[0]):
                #         s, e, f, d = defects[i, 0]
                # start = tuple(contour[s][0])
                # end = tuple(contour[e][0])
                #         far = tuple(contour[f][0])
                # cv2.line(dst, start, end, [0, 255, 0], 1)
                # url:
                # http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/point_polygon_test/point_polygon_test.html
                #         true_distance = cv2.pointPolygonTest(pts_hull, far, True)
                #         min_distance = 2.5
                #         color_circle = (color_rand) if (true_distance > min_distance) else (0, 0, 0)
                #         radius_circle = 2 if (true_distance > min_distance) else 1
                #         cv2.circle(dst, far, radius_circle, color_circle, -1)
                #         print("d: ", d, " - true_distance: ", true_distance)
                # else:
                #     nb_contours_without_defects += 1
    print("nb_contours_without_defects: ", nb_contours_without_defects)


def findContoursSymbols(
    dst,
    tup_contours,
    **params
):
    # unpack
    contours, hierarchy = tup_contours
    # get params
    thickness = params.setdefault('thickness', 2)
    checkConvexivity = params.setdefault('checkConvexivity', True)
    exceptId = params.setdefault('exceptId', -1)
    use_rand_color = params.setdefault('use_rand_color', True)
    draw_contours = params.setdefault('draw_contours', False)
    draw_defects = params.setdefault('draw_defects', False)
    draw_convex_hull = params.setdefault('draw_convex_hull', False)
    ch_rand_color = params.setdefault('ch_rand_color', True)

    nb_contours_without_defects = 0
    print("exceptId: ", exceptId)
    for i, contour in enumerate(contours):
        if i is not exceptId:
            if not(checkConvexivity) or cv2.isContourConvex(contour):
                color_rand = np.random.randint(255, size=3)
                if draw_contours:
                    color_contours = color_rand if use_rand_color else (0, 0, 0)
                    cv2.drawContours(dst, contours, i, color_contours, thickness)

                # print("i, color_rand, arcLength: ", i, color_rand, cv2.arcLength(contour, True))

                if draw_defects or draw_convex_hull:
                    pts_hull = cv2.convexHull(contour, returnPoints=True)
                    if draw_convex_hull:
                        color_ch = color_rand if ch_rand_color else (0, 0, 0)
                        cv2.drawContours(dst, [pts_hull], 0, color_ch, 1)

                    if draw_defects:
                        # url:
                        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
                        hull = cv2.convexHull(contour, returnPoints=False)

                        # url:
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
                                min_distance = 1.5
                                color_circle = (color_rand) if (true_distance > min_distance) else (0, 0, 0)
                                radius_circle = 4 if (true_distance > min_distance) else 2
                                cv2.circle(dst, far, radius_circle, color_circle, -1)
                                # print("d: ", d, " - true_distance: ", true_distance)
                        else:
                            nb_contours_without_defects += 1
    # print("nb_contours_without_defects: ", nb_contours_without_defects)


def extract_horizontal(src, scale=30):
    """Summary

    Args:
        src (TYPE): Description
        scale (int, optional): play with this variable in order to increase/decrease the amount of lines to be detected

    Returns:
        TYPE: Description
    """
    # url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
    # horizontal = src.copy()
    horizontal = src

    # Specify size on horizontal axis
    height, width = horizontal.shape[:2]
    horizontalsize = width / scale

    # Create structure element for extracting horizontal lines through morphology operations
    # Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));
    # url:
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    # print horizontalStructure

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    return horizontal


def morpho_erode(src, kernelSize=2):
    """Summary

    Args:
        src (TYPE): Description
        kernelSize (int, optional): Description

    Returns:
        TYPE: Description
    """
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return cv2.erode(src, kernel, iterations=1)


def morpho_dilate(src, kernelSize=2):
    """Summary

    Args:
        src (TYPE): Description

    Returns:
        TYPE: Description
    """
    # url:
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return cv2.dilate(src, kernel, iterations=1)

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

    ################
    # Denoise image
    # url:
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # urls:
    # - http://docs.opencv.org/2.4/modules/photo/doc/denoising.html
    # - http://www.ipol.im/pub/art/2011/bcm_nlm/
    templateWindowSize = 7
    searchWindowSize = 21
    gray = cv2.fastNlMeansDenoising(gray, None, 10, templateWindowSize, searchWindowSize)
    cv2.imshow('Denoise', gray)
    ################

    minThreshold = 15
    maxThreshold = 255
    #
    # ret, gray = cv2.threshold(gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # #
    bitwise_gray = ~gray
    # bitwise_gray = cv2.medianBlur(bitwise_gray, 3)
    cv2.imshow('bitwise_gray', bitwise_gray)

    # gray = cv2.adaptiveThreshold(bitwise_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -2)
    ret, gray = cv2.threshold(bitwise_gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # gray = cv2.fastNlMeansDenoising(gray)
    cv2.imshow('gray', gray)

    h, w = img.shape[:2]

    params_findContours = {"mode": cv2.RETR_TREE, "method": cv2.CHAIN_APPROX_SIMPLE}
    tup_contours = findContours(gray, **params_findContours)

    img_symbols = np.zeros((h + 2, w + 2, 3), np.uint8)

    img_result = np.zeros((h + 2, w + 2), np.uint8)
    #
    idContourSheet = findContourSheet(gray, img_result, tup_contours, -1)
    # print("idContourSheet: ", idContourSheet)
    findContoursMusicSymbols(gray, img_result, tup_contours, -1, False, idContourSheet)
    ######
    # Reduce horizontals
    # findContoursMusicSymbols(gray, img_result, tup_contours, 3, False, idContourSheet)
    nb_erodation = 2
    for _ in range(nb_erodation):
        img_result = morpho_erode(img_result)
    #
    horizontals = extract_horizontal(img_result, 70)
    #
    img_result_contours = np.zeros((h, w, 3), np.uint8)
    img_result_contours[img_result == 255] = 255
    contours, hierarchy = findContours(img_result.copy(), **params_findContours)
    min_dimension = 5
    for i, contour in enumerate(contours):
        # (x, y), (width, height), angle = rect = cv2.minAreaRect(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if min(height, width) < min_dimension:
            # color_contour = np.random.randint(255, size=3)
            color_contour = (0, 0, 0)
            cv2.drawContours(img_result_contours, contours, i, color_contour, -1)
            cv2.drawContours(img_result, contours, i, color_contour, -1)
        else:
            color_contour = np.random.randint(255, size=3)
            cv2.drawContours(img_result_contours, contours, i, color_contour, 1)
            cv2.rectangle(img_result_contours, (x, y), (x + width, y + height), color_contour, 2)
            # cv2.rectangle(img, (x, y), (x + width, y + height), color_contour, 2)
    #
    cv2.imshow('horizontals', horizontals)
    cv2.imshow('img_result_contours', img_result_contours)
    ######
    #
    nb_dilatation = 3
    for _ in range(nb_dilatation):
        img_result = morpho_dilate(img_result, 3)
    contours, hierarchy = findContours(img_result.copy(), **params_findContours)
    min_dimension = 5
    for i, contour in enumerate(contours):
        x, y, width, height = cv2.boundingRect(contour)
        if min(height, width) >= min_dimension:
            color_contour = np.random.randint(255, size=3)
            cv2.rectangle(img, (x, y), (x + width, y + height), color_contour, 2)
            cv2.rectangle(img_result, (x, y), (x + width, y + height), color_contour, 2)
            #
            cv2.rectangle(img_symbols, (x, y), (x + width, y + height), color_contour, 4)

    cv2.imshow('img_result', img_result)

    # findContoursMusicSymbols(gray, img, tup_contours, -1, False)
    # findContourSheet(gray, img, tup_contours, 1)
    # findContourNotes(gray, img, 1)

    params_findContours = {
        'thickness': 1,
        'checkConvexivity': False,
        'exceptId': idContourSheet,
        'use_rand_color': True,
        'draw_contours': True,
        'draw_defects': False,
        'draw_convex_hull': True,
    }
    findContoursSymbols(img_symbols, tup_contours, **params_findContours)
    cv2.imshow('img_symbols', img_symbols)

    img_2 = img.copy()

    use_mask = True

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
