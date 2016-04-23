import cv2
import numpy as np
import math
import sys
# import pylab as P
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
# filename = "Page_09_Pattern_23.png"
# filename = "Page_09_Pattern_24.png"
# filename = "Page_09_Pattern_25.png"
# filename = "Page_09_Pattern_26.png"
#
# filename = "Page_09_Pattern_23_rot90.png"
filename = "Page_09_Pattern_24_rot.png"
# filename = "Page_09_Pattern_26_rot_crop.png"
#
# filename = "rotate_image.png"


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# url: http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
def rotate_image(image, angle):
    '''Rotate image "angle" degrees.

    How it works:
      - Creates a blank image that fits any rotation of the image. To achieve
        this, set the height and width to be the image's diagonal.
      - Copy the original image to the center of this blank image
      - Rotate using warpAffine, using the newly created image's center
        (the enlarged blank image center)
      - Translate the four corners of the source image in the enlarged image
        using homogenous multiplication of the rotation matrix.
      - Crop the image according to these transformed corners
    '''

    diagonal = int(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))
    offset_x = (diagonal - image.shape[0]) / 2
    offset_y = (diagonal - image.shape[1]) / 2
    dst_image = np.zeros((diagonal, diagonal, 3), dtype='uint8')
    image_center = (diagonal / 2, diagonal / 2)

    R = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    dst_image[offset_x:(offset_x + image.shape[0]),
              offset_y:(offset_y + image.shape[1]),
              :] = image
    dst_image = cv2.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv2.INTER_LINEAR)

    # Calculate the rotated bounding rect
    x0 = offset_x
    x1 = offset_x + image.shape[0]
    x2 = offset_x
    x3 = offset_x + image.shape[0]

    y0 = offset_y
    y1 = offset_y
    y2 = offset_y + image.shape[1]
    y3 = offset_y + image.shape[1]

    corners = np.zeros((3, 4))
    corners[0, 0] = x0
    corners[0, 1] = x1
    corners[0, 2] = x2
    corners[0, 3] = x3
    corners[1, 0] = y0
    corners[1, 1] = y1
    corners[1, 2] = y2
    corners[1, 3] = y3
    corners[2:] = 1

    c = np.dot(R, corners)

    x = int(c[0, 0])
    y = int(c[1, 0])
    left = x
    right = x
    up = y
    down = y

    for i in range(4):
        x = int(c[0, i])
        y = int(c[1, i])
        if (x < left):
            left = x
        if (x > right):
            right = x
        if (y < up):
            up = y
        if (y > down):
            down = y
    h = down - up
    w = right - left

    cropped = np.zeros((w, h, 3), dtype='uint8')
    cropped[:, :, :] = dst_image[left:(left + w), up:(up + h), :]
    return cropped


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
    # cv2.imshow('window', gray)
    # cv2.waitKey(0)

    # minThreshold = 15
    # maxThreshold = 255
    # #
    # ret, gray = cv2.threshold(gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # #
    # bitwise_gray = ~gray
    # bitwise_gray = cv2.medianBlur(bitwise_gray, 3)
    # cv2.imshow('bitwise_gray medianBlur', bitwise_gray)

    # gray = cv2.adaptiveThreshold(bitwise_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -2)
    # ret, gray = cv2.threshold(bitwise_gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # gray = cv2.fastNlMeansDenoising(gray)
    # cv2.imshow('gray', gray)
    ####

    #
    minThreshold = 0  #
    maxThreshold = 255
    ret, bw = cv2.threshold(~gray, minThreshold, maxThreshold, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    edges = cv2.Canny(gray, 150, 700, apertureSize=5)
    edges = ~edges
    cv2.imshow("window", edges)
    cv2.waitKey(0)

    # bw = edges
    bw = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # cv2.imshow("bw", bw)
    # cv2.waitKey(0)
    # url: http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
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
            # angle = abs(angle)

            # print x1, y1, x2, y2
            # print angle, dx, dy, tangente, x1, y1, x2, y2

            angles.append(angle)
            # print "angle: ", angle
            cv2.line(src, (x1, y1), (x2, y2), (0, 255, 0) if angle > 0 else (255, 0, 0), 2)

            # weight = math.sqrt(dy**2 + dx**2)
            # weights.append(weight)
            # print weight

    # print "angles: ", angles

    angles = sorted(angles, key=lambda _: np.random.rand())

    # url:
    # - http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html#numpy.histogram
    # - http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html#gsc.tab=0
    # hist, bin_edges = np.histogram(angles, density=False, normed=False, weights=weights)
    nb_bins = 1000  # un peu bourrrin l'histogramme :p
    hist, bin_edges = np.histogram(angles, nb_bins, density=True)
    hist *= np.diff(bin_edges)
    #
    print "hist: ", hist
    print "bin_edges: ", bin_edges
    print "angles:", angles

    hist = hist.tolist()
    max_value = max(hist)
    print 'max value: ', max_value
    index_of_max = hist.index(max_value)
    angle_for_max = bin_edges[index_of_max]
    print "(hist) angle for max: ", angle_for_max
    print "index of max in hist: ", index_of_max

    angles_around_angle_max = filter(lambda angle: abs(angle - angle_for_max) < 0.1, angles)
    if angles_around_angle_max:
        angle_for_max = max(angles_around_angle_max)
        print("angles_around_angle_max: ", angles_around_angle_max)
        #
        print "angle for max: ", angle_for_max

    cv2.imshow("window", src)
    cv2.imwrite("houghlines3.png", src)
    cv2.waitKey(0)

    # dst = rotate(src, angle_for_max)
    # dst = rotate_image(src2, angle_for_max)
    dst = rotate_image_2(src2, (360 - angle_for_max))
    # dst = rotate_image_2(src2, angle_for_max)

    cv2.imshow("window", dst)
    cv2.imwrite("rotate_image.png", dst)
    cv2.waitKey(0)

    # mu, sigma = 200, 25
    # x = angles
    # the histogram of the data with histtype='step'
    # n, bins, patches = P.hist(x, 5000, normed=1, histtype='stepfilled')
    # P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    # P.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # mu, sigma = 100, 15
    # x = angles
    # the histogram of the data
    # n, bins, patches = ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    # hist uses np.histogram under the hood to create 'n' and 'bins'.
    # np.histogram returns the bin edges, so there will be 50 probability
    # density values in n, 51 bin edges in bins and 50 patches.  To get
    # everything lined up, we'll compute the bin centers
    # bincenters = 0.5 * (bins[1:] + bins[:-1])
    # print("bincenters: ", bincenters)
    # print("average(bincenters): ", np.average(bincenters))
    # add a 'best fit' line for the normal PDF
    # y = mlab.normpdf(bincenters, mu, sigma)
    # l = ax.plot(bincenters, y, 'r--', linewidth=2)

    # ax.set_xlabel('Smarts')
    # ax.set_ylabel('Probability')
    # ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # ax.set_xlim(-90, 90)
    # ax.set_ylim(0, 0.03)
    # ax.grid(True)

    # plt.show()

    m = min(angles)
    M = max(angles)
    axis = [0, len(angles), m, M]
    print("axis: ", axis)

    # import matplotlib.pyplot as plt
    # plt.plot(xrange(len(angles)), angles, 'ro')
    # plt.axis(axis)
    # plt.show()

    w = len(angles)
    h = w
    img_angles = np.zeros((h + 1, w + 1, 1), np.uint8)
    for i, angle in enumerate(angles):
        angle = (angle - m) / (M - m) * w
        img_angles[i, angle] = 255
    cv2.imshow("img_angles", img_angles)

    img_hough = np.zeros((h + 1, w + 1, 3), np.uint8)
    minLineLength = w * 0.90
    # minLineLength = 10
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
        cv2.imshow("image rotated with Hough method", dst)

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
