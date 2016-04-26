# url: http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them

import cv2
import numpy as np
from shapely.geometry import Polygon


def find_if_close(cnt1, cnt2, dist_min=50):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < dist_min:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def unified_contours_brute_force(contours, dist_min=50):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i

        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2, dist_min)
                if dist:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in xrange(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    return unified


def test_unified_contours_brute_force(contours, img):
    unified = unified_contours_brute_force(contours)

    cv2.drawContours(img, unified, -1, (0, 255, 0), 2)
    # cv2.drawContours(thresh, unified, -1, 255, -1)

    # cv2.imshow('img', img)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)


def test_unified_contours_morpho(contours, img, thresh):
    img_contours = thresh.copy()
    cv2.drawContours(img_contours, contours, -1, (255, 0, 0), -1)

    img_morpho = np.zeros((width, height, 1), np.uint8)

    nb_times = 1
    pts_hull = []
    for i in range(nb_times):
        scale = 8
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * scale, 8 * scale))
        img_morpho = cv2.morphologyEx(img_contours, cv2.MORPH_CLOSE, kernel)

        tup_results = cv2.findContours(img_morpho, cv2.RETR_EXTERNAL, 2)  #
        cts, hier = tup_results[len(tup_results) == 3:]

        pts_hull = [cv2.convexHull(contour, returnPoints=True) for contour in cts]
        cv2.drawContours(img_contours, pts_hull, -1, (255, 255, 0), -1)

    pts_hull = [cv2.convexHull(contour, returnPoints=True) for contour in cts]
    cv2.drawContours(img, pts_hull, -1, (255, 255, 0), 2)


if __name__ == '__main__':
    img = cv2.imread('RoKEh.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    tup_results = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)  #
    contours, hier = tup_results[len(tup_results) == 3:]

    b_use_morpho = True
    if b_use_morpho:
        width, height = thresh.shape[:2]
        # test_unified_contours_morpho(contours, img, thresh)
        import timeit
        print(timeit.timeit("test_unified_contours_morpho(contours, img, thresh)",
                            setup="from __main__ import test_unified_contours_morpho, contours, img, thresh",
                            number=1)
              )
        cv2.imshow('img', img)
        cv2.waitKey(0)

    b_use_brute_force = False
    if b_use_brute_force:
        # test_unified_contours_brute_force(contours, img)
        import timeit
        print(timeit.timeit("test_unified_contours_brute_force(contours, img)",
                            setup="from __main__ import test_unified_contours_brute_force, contours, img",
                            number=1)
              )
        cv2.imshow('image', img)
        # cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
