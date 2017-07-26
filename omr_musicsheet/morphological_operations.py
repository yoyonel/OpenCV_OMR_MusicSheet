#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from omr_musicsheet.cv2_tools import *

if __name__ == "__main__":
    filename = "Page_09_Pattern_24.png"
    src = cv2.imread(filename)

    src = cv2.fastNlMeansDenoising(src, None, 10, 7, 21)
    # omr_musicsheet = cv2.medianBlur(omr_musicsheet, 5)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # LSD
    line_length = 30
    showImage(detectLine_LSD(gray, line_length ** 2), "LSD")

    # Binarisation de l'image
    bw = binarize_img(gray)
    bin_img = bw.copy()
    showImage(bw, "Black & White - adaptiveThreshold")

    # bw_erode = bw.copy()
    # bw_erode = cv2.medianBlur(bw_erode, 5)
    # # morpho: erode
    # nb_morpho_erode = 3
    # for _ in xrange(nb_morpho_erode):
    #     bw_erode = morpho_erode(bw_erode)
    # # morpho: dilate
    # nb_morpho_dilate = 2
    # for _ in xrange(2):
    #     bw_erode = morpho_dilate(bw_erode)
    # showImage(bw_erode, "Morpho - Erosion")
    # # cv2.imwrite("bw_after_erosion.png", bw_erode)

    bw = morpho_dilate(bw)
    #
    showImage(bw, "Morpho - Dilatation")
    cv2.imwrite("bw_after_dilate.png", bw)

    # url: http://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
    horizontal = extract_horizontal(bw.copy())
    #
    showImage(horizontal, "Morpho - extract horizsobelxyontal")
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

    minPerimeter = 100
    params_Contours = {
        "minPerimeter": minPerimeter,
        "thickness": 1
    }
    params_drawContours = params_Contours
    params_fillContours = dict(params_Contours, **{'use_color_rand': True})
    #
    drawContours_filterByPerimeter(img_contours, contours, **params_Contours)
    # merge de dict python -> url: http://stackoverflow.com/a/39858
    fillContours_filterByPerimeter(img_contours, contours, **params_fillContours)
    #
    drawContours_filterByPerimeter(img_contours_2, contours, **params_drawContours)
    fillContours_filterByPerimeter(img_contours_2, contours, **params_fillContours)

    params_detectLines = dict(params_fillContours, **{'color': (255, 0, 0)})
    # detectLine_fromContours_filterByPerimeter(img_contours, contours, **params_detectLines)

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

    _, contours, hierarchy = cv2.findContours(
        add_vertical.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        x, y, w, h = cv2.boundingRect(contour)
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

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
