#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import cv2
import numpy as np
# import math


def showImage(
        img,
        namedWindow="",
        width=640, height=480
):
    cv2.namedWindow(namedWindow, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(namedWindow, width, height)
    cv2.imshow(namedWindow, img)


def morphological_skeleton(img, maxIter=1024):
    """Summary

    Args:
        img (TYPE): Description
        maxIter (int, optional): Description

    Returns:
        TYPE: Description
    """
    # url: http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    height, width = img.shape[:2]
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
        done = (cv2.countNonZero(img) == 0) or (nbIteration >= maxIter)

    return skel, nbIteration


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


def extract_vertical(src, scale=30):
    """Summary

    Args:
        src (TYPE): Description
        scale (int, optional): play with this variable in order to increase/decrease the amount of lines to be detected

    Returns:
        TYPE: Description
    """
    # url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
    vertical = src

    # Specify size on horizontal axis
    height, width = vertical.shape[:2]
    verticalsize = height / scale

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # print verticalStructure

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    return vertical


def morpho_dilate(src, kernelSize=2, iterations=1):
    """Summary

    Args:
        src (TYPE): Description

    Returns:
        TYPE: Description
    """
    # url:
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return cv2.dilate(src, kernel, iterations=iterations)


def morpho_erode(src, kernelSize=2, iterations=1):
    """Summary

    Args:
        src (TYPE): Description
        kernelSize (int, optional): Description

    Returns:
        TYPE: Description
    """
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return cv2.erode(src, kernel, iterations=iterations)


def fore_back_ground(img1, img2):
    """Summary
        Dessine img2 dans img1 en creant un mask (binaire) d'image lié à ces intensités de couleurs.
        Le noir (0, 0, 0) est considéré totalement transparent.
        Toute les autres couleurs (à partir d'une certaine intensite => threshold de 10 à 255) sont opaques.

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

# TODO: faire une version générique
# def renderContours(
#     img,
#     contours,
#     **params)


def drawContours_filterByPerimeter(img, contours, **params):
    """Summary

    Args:
        img (TYPE): Description
        contours (TYPE): Description
        **params (TYPE): Description

    Returns:
        TYPE: Description
    """
    #
    minPerimeter = params.get("minPerimeter", 0)
    maxPerimeter = params.get("maxPerimeter", 10000)
    thickness = params.get("thickness", 1)
    useRandomColor = params.get("useRandomColor", True)
    #
    if useRandomColor:
        for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            if (perimeter >= minPerimeter) and (perimeter <= maxPerimeter):
                color_rand = np.random.randint(255, size=3)
                cv2.drawContours(img, contours, i, color_rand, thickness)
    else:
        color = params.get("color", (255, 255, 255))
        for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            if (perimeter >= minPerimeter) and (perimeter <= maxPerimeter):
                cv2.drawContours(img, contours, i, color, thickness)


def fillContours_filterByPerimeter(img, contours, **params):
    """Summary

    Args:
        img (TYPE): Description
        contours (TYPE): Description
        **params (TYPE): Description

    Returns:
        TYPE: Description
    """
    #
    minPerimeter = params.get("minPerimeter", 0)
    maxPerimeter = params.get("maxPerimeter", 10000)
    color = params.get("color", (255, 255, 255))
    use_color_rand = params.get("use_color_rand", False)
    #
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        if (perimeter >= minPerimeter) and (perimeter <= maxPerimeter):
            if use_color_rand:
                color = np.random.randint(255, size=3)
                print "color_rand: ", color
            cv2.fillPoly(img, pts=contour, color=color)


def detectLine_fromContours_filterByPerimeter(_img, contours, **params):
    """Summary

    Args:
        img (TYPE): Description
        contours (TYPE): Description
        **params (TYPE): Description

    Returns:
        TYPE: Description
    """
    #
    minPerimeter = params.get("minPerimeter", 0)
    maxPerimeter = params.get("maxPerimeter", 10000)
    thickness = params.get("thickness", 1)
    color = params.get("color", (255, 255, 255))
    #
    width, height = _img.shape[:2]
    #
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        if (perimeter >= minPerimeter) and (perimeter <= maxPerimeter):
            # url: https://github.com/Itseez/opencv/blob/master/samples/python/fitline.py
            # url: http://stackoverflow.com/questions/14184147/detect-lines-opencv-in-object
            # then apply fitline() function
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            # [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_HUBER, 0, 0.01, 0.01)
            # Now find two extreme points on the line to draw line
            lefty = int((-x * vy / vx) + y)
            righty = int(((width - x) * vy / vx) + y)

            # Finally draw the line
            cv2.line(_img, (width - 1, righty), (0, lefty), color, thickness)


def detectLine_LSD(src, minLength2=-1):
    """Summary
        LSD: Line Segment Detector
    Args:
        src (TYPE): Description

    Returns:
        TYPE: Description
    """
    width, height = src.shape
    dst = np.zeros((width, height, 3), np.uint8)
    ls = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    tup_results = ls.detect(cv2.medianBlur(src, 5))
    lines, widths, _, _ = tup_results
    print widths
    if lines is not None:
        if minLength2 != -1:
            def length2(line):
                return (line[0][0] - line[0][2])**2 + (line[0][1] - line[0][3])**2
            lines = np.array(
                filter(
                    lambda line: length2(line) <= minLength2,
                    lines
                )
            )
        print "# Lines: ", len(lines)
        ls.drawSegments(dst, lines)
    return dst


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
    contours, hierarchy = tup_results[len(tup_results) == 3:]
    # url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
    # print hierarchy
    return contours, hierarchy


def binarize_img(img, **params):
    """Summary

    Args:
        img (TYPE): Description

    Returns:
        TYPE: Description
    """
    #
    minThreshold = params.get("minThreshold", 200)
    maxThreshold = params.get("maxThreshold", 255)
    #
    bitwise_gray = ~img

    # bitwise_gray = cv2.medianBlur(cv2.Scharr(bitwise_gray, cv2.CV_8U, 0, 1), 5)
    # bitwise_gray = cv2.medianBlur(bitwise_gray, 5)

    # bw = cv2.adaptiveThreshold(bitwise_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    ret, bw = cv2.threshold(bitwise_gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bw


def findContourSheet(dst, tup_contours, thickness=2):
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


def waitKey(idKeyForExit=27):
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == idKeyForExit:
            break


def test_unified_contours_morpho(contours, src, **params):
    #
    size_kernel = params.setdefault("size_kernel", 8 * 8)
    kernel = params.setdefault("kernel", cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_kernel, size_kernel)))
    # contour_color = params.setdefault("contour_color", (255, 255, 0))
    # contour_width = params.setdefault("contour_width", 2)
    # use_rand_color = params.setdefault("use_rand_color", False)
    #
    width, height = src.shape[:2]
    img_contours = np.zeros((width, height, 1), np.uint8)
    cv2.drawContours(img_contours, contours, -1, (255, 0, 0), -1)

    nb_times = 1
    pts_hull = []
    for i in range(nb_times):
        img_morpho = cv2.morphologyEx(img_contours, cv2.MORPH_CLOSE, kernel)

        tup_results = cv2.findContours(img_morpho, cv2.RETR_EXTERNAL, 2)  #
        cts, hier = tup_results[len(tup_results) == 3:]

        pts_hull = [cv2.convexHull(contour, returnPoints=True) for contour in cts]
        cv2.drawContours(img_contours, pts_hull, -1, (255, 255, 0), -1)

    # if use_rand_color:
    #     for contour in cts:
    #         rand_color = np.random.randint(255, size=3)
    #         cv2.drawContours(dst, [cv2.convexHull(contour, returnPoints=True)], -1, rand_color, contour_width)
    # else:
    #     cv2.drawContours(dst, [cv2.convexHull(contour, returnPoints=True)
    #                            for contour in cts], -1, contour_color, contour_width)

    return cts
