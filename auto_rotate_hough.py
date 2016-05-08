import cv2
import numpy as np
import math
import sys

from cv2_tools import *

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
filename = "Page_09_Pattern_24.png"
# filename = "Page_09_Pattern_25.png"
# filename = "Page_09_Pattern_26.png"
#
# filename = "Page_09_Pattern_23_rot90.png"
# filename = "Page_09_Pattern_24_rot.png"
# filename = "Page_09_Pattern_26_rot_crop.png"
#
# filename = "rotate_image.png"


def rotate_image_2(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
    return rotated_mat


if __name__ == '__main__':
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)

    src = cv2.imread(filename)
    src2 = src.copy()

    #############################
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    ####
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    ####

    #
    # minThreshold = 0  #
    # maxThreshold = 255
    # ret, bw = cv2.threshold(~gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # edges = cv2.Canny(gray, 150, 700, apertureSize=5)
    # edges = ~edges
    edges = gray
    cv2.imshow("window", edges)
    cv2.waitKey(0)

    bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    cv2.imwrite("bw.png", bw)
    #############################

    minLineLength = 0
    lines = cv2.HoughLinesP(bw, rho=1,
                            theta=math.pi / 180, threshold=70,
                            minLineLength=50, maxLineGap=5
                            )
    # url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    # print lines
    angles = []
    weights = []
    for line in lines:
        # print line
        x1, y1, x2, y2 = line[0]
        # url: http://stackoverflow.com/questions/9528421/value-for-epsilon-in-python
        if abs(x2 - x1) >= sys.float_info.epsilon:
            # url: http://www.pdnotebook.com/2012/07/measuring-angles-in-opencv/
            dy = float(y2 - y1)
            dx = float(x2 - x1)
            tangente = -dy / dx
            angle = math.atan(tangente)
            angle *= (180.0 / math.pi)

            angles.append(angle)
            cv2.line(bw, (x1, y1), (x2, y2), (0, 255, 0) if angle > 0 else (255, 0, 0), 1)
    cv2.imshow("bw + lines", bw)
    # waitKey()

    # tri aleatoire du tableau des angles
    # on s'assure que les angles ne soient pas triees (comme de par "hasard" :p)
    angles = sorted(angles, key=lambda _: np.random.rand())
    # angles = sorted(angles)   # le tri des angles casse l'algo ... a etudier !

    m = min(angles)
    M = max(angles)
    axis = [0, len(angles), m, M]
    # print("axis: ", axis)

    w = len(angles)
    h = w
    img_angles = np.zeros((h + 1, w + 1, 1), np.uint8)
    for i, angle in enumerate(angles):
        angle = (angle - m) / (M - m) * w
        img_angles[i, angle] = 255
    cv2.imshow("img_angles", img_angles)

    img_hough = np.zeros((h + 1, w + 1, 3), np.uint8)
    minLineLength = w * 0.90
    maxLineGap = w
    lines = cv2.HoughLinesP(img_angles, rho=1,
                            theta=math.pi / 180, threshold=70,
                            minLineLength=minLineLength, maxLineGap=maxLineGap
                            )
    if lines is not None:
        # url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        print lines
        results_rotations = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            print "(x1, y1), (x2, y2): ", (x1, y1), (x2, y2)
            angle = (x1 / (float)(w)) * (M - m) + m
            result = (angle, abs(y2 - y1))
            print "=> angle & length_line: ", result
            cv2.line(img_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
            results_rotations.append(result)
        cv2.imshow("houghlines3", img_hough)
        print "results_rotations: ", results_rotations
        angle_rotation = max(lambda tup: tup[1], results_rotations)[0][0]
        print "-> angle_rotation: ", angle_rotation
        dst = rotate_image_2(src2, (360 - angle_rotation))
        cv2.imshow("window", dst)

    ####
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    ####
    edges = ~gray
    bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horizontals = extract_horizontal(bw, 40)
    contours, hier = findContours(horizontals.copy())
    # ratio_approx = 0.005 * 0
    # contours_approx = map(lambda cnt: cv2.approxPolyDP(cnt, ratio_approx * cv2.arcLength(cnt, True), True), contours)
    img_contours = dst.copy()
    drawContours_filterByPerimeter(img_contours, contours, **{'minPerimeter': 500})
    # drawContours_filterByPerimeter(img_contours, contours_approx, **{'minPerimeter': 500})

    list_ch = [cv2.convexHull(contour, returnPoints=True) for contour in contours]
    cv2.drawContours(img_contours, list_ch, -1, (255, 255, 0), 1)

    cv2.imshow("horizontals", horizontals)
    cv2.imshow("bw", bw)
    cv2.imshow("img_contours", img_contours)

    waitKey()
