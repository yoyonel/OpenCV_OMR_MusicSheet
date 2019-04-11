"""
"""
import cv2
import logging
import numpy as np
from pathlib import Path
import pprint
from random import randint
from typing import Tuple
from sklearn.cluster import MeanShift, estimate_bandwidth

from omr_musicsheet.tools.logger import init_logger
from omr_musicsheet.datasets import get_module_path_datasets

logger = logging.getLogger(__name__)


def find_contours(in_img):
    """Summary

    Args:
        in_img (TYPE): Description

    Returns:
        TYPE: Description
    """
    tup_results = cv2.findContours(in_img, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_TC89_L1)

    contours, hierarchy = tup_results[len(tup_results) == 3:]

    # # url: http://opencvpython.blogspot.fr/2013/01/contours-5-hierarchy.html
    logger.info(f"hierarchy: {pprint.pformat(hierarchy)}")

    return contours, hierarchy


def random_color() -> Tuple[int, int, int]:
    return randint(0, 255), randint(0, 255), randint(0, 255)


def draw_contours(out_img, contours, thickness=3):
    """Summary

    Args:
        out_img (TYPE): Description
        contours (TYPE): Description
        thickness (int, optional): Description

    Returns:
        TYPE: Description
    """
    # url: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.randint.html
    for i, contour in enumerate(contours):
        cv2.drawContours(out_img, contours, i,
                         color=random_color(), thickness=thickness)


def find_aabbox_from_contours(contours, offset: int = 0):
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        top_left = bbox[0] - offset, bbox[1] - offset
        bottom_right = (
            bbox[0] + bbox[2] - 1 + offset,
            bbox[1] + bbox[3] - 1 + offset
        )
        yield top_left, bottom_right


def draw_aabbox(
        out_img,
        bbox,
        thickness=3,
        color=None,
):
    """

    :param out_img:
    :param bbox:
    :param thickness:
    :param color:
    """
    for top_left, bottom_right in bbox:
        cv2.rectangle(out_img, top_left, bottom_right,
                      color if color else random_color(), thickness)


def bbox_perimeter(bb):
    return (abs(bb[1][0] - bb[0][0]) + abs(bb[1][1] - bb[0][1])) * 2


def bbox_center(bb):
    return (bb[1][0] + bb[0][0]) * 0.5, (bb[1][1] + bb[0][1]) * 0.5


def cluster_bbox(bbox):
    # https://stackoverflow.com/questions/18364026/clustering-values-by-their-proximity-in-python-machine-learning

    # x = [1, 1, 5, 6, 1, 5, 10, 22, 23, 23, 50, 51, 51, 52, 100, 112, 130, 500,
    #      512, 600, 12000, 12230]

    # X = np.array(list(zip(x, np.zeros(len(x)))), dtype=np.int)
    X = np.array(
        list(zip([bb[0][1] for bb in bbox], np.zeros(len(bbox)))),
        dtype=np.int
    )
    bandwidth = estimate_bandwidth(X, quantile=1.0 / 15.0)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    # cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    for k in range(n_clusters_):
        my_members = labels == k
        print("cluster {0}: {1}".format(k, set(X[my_members, 0])))


def hline_intersect_bbox(y: float, bb) -> bool:
    return bb[0][1] <= y <= bb[1][1]


def cluster_bbox_with_rectilines(bbox):
    id_cluster = 0
    clusters = []
    it_bbox = iter(bbox)
    try:
        bb = next(it_bbox)
        while True:
            nb_elements_in_cluster = 1
            bb_center = bbox_center(bb)

            next_bb = next(it_bbox)
            while hline_intersect_bbox(bb_center[1], next_bb):
                nb_elements_in_cluster += 1
                next_bb = next(it_bbox)

            clusters += [id_cluster] * nb_elements_in_cluster
            id_cluster += 1

            bb = next_bb
    except StopIteration:
        pass

    return clusters


def compute_aabbox_from_img_and_mask(img_fn: str):
    img_path = Path(img_fn)
    if not img_path.exists():
        raise IOError(f"{img_path} does'nt exist !")

    img_mask_path = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")
    if not img_mask_path.exists():
        raise IOError(f"{img_mask_path} does'nt exist !")

    im = cv2.imread(str(img_mask_path))
    im_mss = cv2.imread(str(img_path))

    im_result = im.copy()

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html#histograms-getting-started
    # https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
    # hist = cv2.calcHist(im, [0], None, [256], [0, 256])
    # bg_color = im[0]
    # bg_color = [113, 105, 76, 0]

    # Binarization
    # using background color
    # http://benjamintan.io/blog/2018/05/24/making-transparent-backgrounds-with-numpy-and-opencv-in-python/
    # im_result[np.all(im == bg_color, axis=2)] = [0, 0, 0, 0]
    # im_result[np.all(im != bg_color, axis=2)] = [255, 255, 255, 255]
    # thresh = im_result
    #
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 254, 255, cv2.THRESH_BINARY)

    #
    img_for_contours = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    #
    contours, hierarchy = find_contours(img_for_contours)
    draw_contours(im_result, contours, thickness=2)

    #
    bbox = list(find_aabbox_from_contours(contours, offset=2))
    # filter bbox list
    min_perimeter = 15 * 4
    max_perimeter = 80 * 4
    bbox = list(filter(
        lambda bb: min_perimeter <= bbox_perimeter(bb) <= max_perimeter,
        bbox
    ))
    bbox = sorted(bbox, key=lambda bb: (bb[0][1], bb[0][0]))
    logger.info(f"bbox =\n{pprint.pformat(bbox)}")
    draw_aabbox(im_result, bbox, thickness=1, color=(0, 255, 0))
    draw_aabbox(im, bbox, thickness=1, color=(255, 0, 0))
    draw_aabbox(im_mss, bbox, thickness=1, color=(255, 0, 0))

    #
    # cv2.imshow("img_for_contours", img_for_contours)
    # cv2.imshow('Sprite Sheets - Mask', im)
    # cv2.imshow('Sprite Sheets - Results', im_result)
    cv2.imshow(f'Sprite Sheets on {img_fn}', im_mss)

    # cluster_bbox(bbox)
    print(cluster_bbox_with_rectilines(bbox))


def compute_and_render(fn_img: str, wait_for_escape=True):
    # Mask created with GIMP: image>mode>rgb tools>color_tools>colorize
    # https://docs.gimp.org/2.10/fr/gimp-tool-threshold.html
    fn = str(Path(get_module_path_datasets()) / fn_img)
    compute_aabbox_from_img_and_mask(fn)
    while True and wait_for_escape:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def main():
    compute_and_render('mercedesspritesheets.png', wait_for_escape=False)
    compute_and_render('trump_run.png')


if __name__ == '__main__':
    init_logger(logger)
    main()
