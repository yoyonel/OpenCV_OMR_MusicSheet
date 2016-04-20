#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

"""Summary

Attributes:
    bw (TYPE): Description
    filename (str): Description
    gray (TYPE): Description
    horizontal (TYPE): Description
    img_contours (TYPE): Description
    img_contours_2 (TYPE): Description
    img_contours_mskel (TYPE): Description
    params_detectLines (TYPE): Description
    params_drawContours (dict): Description
    params_fillContours (dict): Description
    params_findContours (TYPE): Description
    src (TYPE): Description
    vertical (TYPE): Description

Deleted Attributes:
    bitwise_gray (TYPE): Description
    minLineLength (int): Description
    minPerimeter (int): Description
    thickness (int): Description
    tup_results (TYPE): Description
"""
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
    useRandomColor = True
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
    #
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        if (perimeter >= minPerimeter) and (perimeter <= maxPerimeter):
            cv2.fillPoly(img, pts=contours, color=color)


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
    tup_results = cv2.findContours(horizontal, mode, method)  #
    global contours, hierarchy
    if len(tup_results) == 3:
        im2, contours, hierarchy = tup_results
    else:
        contours, hierarchy = tup_results
    # url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
    # print hierarchy
    return contours, hierarchy


def binarize_img(img):
    """Summary

    Args:
        img (TYPE): Description

    Returns:
        TYPE: Description
    """

    bitwise_gray = ~img

    # bitwise_gray = cv2.medianBlur(cv2.Scharr(bitwise_gray, cv2.CV_8U, 0, 1), 5)
    # bitwise_gray = cv2.medianBlur(bitwise_gray, 5)

    # bw = cv2.adaptiveThreshold(bitwise_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    minThreshold = 200
    maxThreshold = 255
    ret, bw = cv2.threshold(bitwise_gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bw

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
# filename = "Page_09_Pattern_24.png"
# filename = "Page_09_Pattern_25.png"
filename = "Page_09_Pattern_26.png"
#
# filename = "Page_09_Pattern_23_rot90.png"
# filename = "rotate_image.png"

src = cv2.imread(filename)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# LSD
line_length = 30
showImage(detectLine_LSD(gray, line_length ** 2), "LSD")

# Binarisation de l'image
bw = binarize_img(gray)
bin_img = bw.copy()
showImage(bw, "Black & White - adaptiveThreshold")

bw = morpho_dilate(bw)
#
showImage(bw, "Morpho - Dilatation")
cv2.imwrite("bw_after_dilate.png", bw)

# url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
horizontal = extract_horizontal(bw.copy())
#
showImage(horizontal, "Morpho - extract horizontal")
cv2.imwrite("extract_horizontal.png", horizontal)

remove_horizontal = bin_img.copy()
remove_horizontal[horizontal == 255] = 0
remove_horizontal = cv2.medianBlur(remove_horizontal, 5)
showImage(remove_horizontal, "remove_horizontal")

vertical = extract_vertical(bw.copy())
#
showImage(vertical, "Morpho - extract vertical")
cv2.imwrite("extract_vertical.png", vertical)

add_vertical = remove_horizontal.copy()
add_vertical[vertical == 255] = 255
add_vertical = cv2.GaussianBlur(morpho_dilate(add_vertical, 3), (5, 5), 0)
showImage(add_vertical, "add_vertical")

add_vertical_mskel, nbIter = morphological_skeleton(add_vertical.copy())
showImage(add_vertical_mskel, "add_vertical + MorphoSkel - nbIter={0}".format(nbIter))

params_findContours = {"mode": cv2.RETR_CCOMP, "method": cv2.CHAIN_APPROX_SIMPLE}
contours, hierarchy = findContours(horizontal, **params_findContours)

height, width = src.shape[:2]
img_contours = src.copy()
img_contours_2 = np.zeros((height, width, 1), np.uint8)

params_drawContours = {"minPerimeter": 2000, "useRandomColor": True, "thickness": 1}
params_fillContours = {"minPerimeter": 2000, "thickness": 1}
#
drawContours_filterByPerimeter(img_contours, contours, **params_drawContours)
# merge de dict python -> url: http://stackoverflow.com/a/39858
fillContours_filterByPerimeter(img_contours, contours, **dict(params_fillContours, **{'color': (0, 255, 255)}))
#
drawContours_filterByPerimeter(img_contours_2, contours, **params_drawContours)
fillContours_filterByPerimeter(img_contours_2, contours, **params_fillContours)

params_detectLines = dict(params_fillContours, **{'color': (255, 0, 0)})
detectLine_fromContours_filterByPerimeter(img_contours, contours, **params_detectLines)

img_contours_mskel, nbIter = morphological_skeleton(img_contours_2.copy())
showImage(img_contours_mskel, "img_contours_2 + MorphoSkel - nbIter={0}".format(nbIter))

img_contours_2 = cv2.Canny(img_contours_2, 150, 700, apertureSize=5)
showImage(img_contours_2, "Contours - Canny")

showImage(img_contours, "Contours - Couleurs")

img_contours_mskel = cv2.cvtColor(img_contours_mskel, cv2.COLOR_GRAY2BGR)
# url: stackoverflow.com/questions/11433604/opencv-setting-all-pixels-of-specific-bgr-value-to-another-bgr-value
img_contours_mskel[np.where((img_contours_mskel == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

# img_contours_mskel = cv2.addWeighted(img_contours, 0.5, img_contours_mskel, 0.5, 0)
img_contours_mskel = fore_back_ground(img_contours, img_contours_mskel)

showImage(img_contours_mskel, "Contours + MSkel - Couleurs")

_, contours, hierarchy = cv2.findContours(add_vertical.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)
    # discard areas that are too large
    # if h > 300 and w > 300:
    #     continue
    # discard areas that are too small
    # if h < 40 or w < 40:
    #     continue
    # draw rectangle around contour on original image
    cv2.rectangle(src, (x, y), (x + w, y + h), (200, 64, 255), 3)

src[img_contours == (255, 0, 0)] = img_contours[img_contours == (255, 0, 0)]

showImage(src, "Source")

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
