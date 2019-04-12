#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
"""
Floodfill sample.

Usage:
  floodfill.py [<image>] [<fix_range>] [<connectivity>] [<lo_init>] [<hi_init>]

  Click on the image to set seed point

Keys:
  f     - toggle floating range
  c     - toggle 4/8 connectivity
  ESC   - exit
"""
import ast
import sys

from omr_musicsheet.cv2_tools import *
from omr_musicsheet.datasets import get_image_path


def main():
    try:
        fn = sys.argv[1]
    except:
        fn = str(get_image_path('volt_sprite_sheet_by_kwelfury-d5hx008.png'))
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
    img_warping = img.copy()

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
    idContourSheet = findContourSheet(img_result, tup_contours, -1)
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
    img_result_contours = np.zeros(img_result.shape, np.uint8)
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
            color_contour = random_color()
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
            color_contour = random_color()
            cv2.rectangle(img, (x, y), (x + width, y + height), color_contour, 2)
            cv2.rectangle(img_result, (x, y), (x + width, y + height), color_contour, 2)
            #
            # cv2.rectangle(img_symbols, (x, y), (x + width, y + height), color_contour, 4)

    cv2.imshow('img_result', img_result)

    # findContoursMusicSymbols(gray, img, tup_contours, -1, False)
    # findContourSheet(img, tup_contours, 1)
    # findContourNotes(gray, img, 1)

    params_findContours = {
        'thickness': -1,
        'checkConvexivity': False,
        'exceptId': idContourSheet,
        'use_rand_color': True,
        'draw_contours': True,
        'draw_defects': False,
        'draw_convex_hull': True,
    }
    findContoursSymbols(img_symbols, tup_contours, **params_findContours)
    findContoursSymbols(img, tup_contours, **params_findContours)
    cv2.imshow('img_symbols', img_symbols)

    findContourSheet(img_warping, tup_contours)
    cv2.imshow('img_warping', img_warping)

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


if __name__ == '__main__':
    main()