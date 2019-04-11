from omr_musicsheet.cv2_tools import *

if __name__ == "__main__":
    filename = "Page_09_Pattern_26.png"
    src = cv2.imread(filename)

    src = cv2.fastNlMeansDenoising(src, None, 10, 7, 21)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bw = binarize_img(gray, minThreshold=15)

    cv2.imshow("bw", bw)

    #
    bw_erode = bw.copy()
    ##
    medianBlur_kernel = 3
    bw_erode = cv2.medianBlur(bw_erode, medianBlur_kernel)
    cv2.imshow("after medianBlur", bw_erode)

    # Morpho Operation : Opening
    # Creation d'un noyau ELLIPSE de rayon 5 (disque)
    size_kernel = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_kernel, size_kernel))
    # print "kernel: ", kernel
    bw_erode = cv2.morphologyEx(bw_erode, cv2.MORPH_OPEN, kernel)

    img_unified = bw_erode.copy()
    img_extract_contours = img_unified.copy()
    #
    tup_results = cv2.findContours(img_extract_contours, cv2.RETR_EXTERNAL, 2)  #
    contours, hier = tup_results[len(tup_results) == 3:]

    minPerimeter = 32
    contours = filter(lambda contour: cv2.arcLength(contour, True) > minPerimeter, contours)

    # unification de contours avec un noyau vertical de fusion
    verticalsize = 13
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    contours_unified = test_unified_contours_morpho(contours, src,
                                                    kernel=verticalStructure, contour_width=2, use_rand_color=True)
    horizontalSize = 7
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    contours_unified = test_unified_contours_morpho(contours_unified, src,
                                                    kernel=horizontalStructure, contour_width=2, use_rand_color=True)

    list_ch = [cv2.convexHull(contour, returnPoints=True) for contour in contours_unified]
    cv2.drawContours(src, list_ch, -1, (255, 255, 0), 1)
    cv2.drawContours(bw_erode, list_ch, -1, 255, 2)

    list_bb = [cv2.boundingRect(ch) for ch in list_ch]
    map(lambda bb: cv2.rectangle(src, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 255), 1), list_bb)
    map(lambda bb: cv2.rectangle(src, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 255), 1), list_bb)
    map(lambda bb: cv2.circle(src, (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2),
                              min(bb[2], bb[3]) / 2, (255, 0, 255), 1), list_bb)
    map(lambda bb: cv2.circle(src, (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2),
                              3, (255, 0, 255), -11), list_bb)

    showImage(bw_erode, "Morpho - Erosion")
    showImage(src, "Src + Unified Contours")
    # cv2.imwrite("bw_after_erosion.png", bw_erode)

    waitKey()
