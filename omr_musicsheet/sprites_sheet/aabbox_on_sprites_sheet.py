"""
"""
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import logging
import numpy as np
from pathlib import Path
import pprint
from random import randint
from typing import Tuple, List, Iterable
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

from omr_musicsheet.tools.logger import init_logger
from omr_musicsheet.datasets import get_module_path_datasets

logger = logging.getLogger(__name__)


@dataclass
class Point2D:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))


@dataclass
class AABBox:
    top_left: Point2D
    bottom_right: Point2D

    perimeter: int = field(init=False)

    def compute_perimeter(self) -> int:
        self.perimeter = (abs(self.bottom_right.x - self.top_left.x) +
                          abs(self.bottom_right.y - self.top_left.y)) * 2
        return self.perimeter

    def compute_center(self) -> Point2D:
        return Point2D(
            (self.bottom_right.x + self.top_left.x) // 2,
            (self.bottom_right.y + self.top_left.y) // 2,
        )

    def __iter__(self):
        return self.top_left, self.bottom_right


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


def find_aabbox_from_contours(contours, offset: int = 0) -> Iterable[AABBox]:
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        top_left = Point2D(bbox[0] - offset, bbox[1] - offset)
        bottom_right = Point2D(
            bbox[0] + bbox[2] - 1 + offset,
            bbox[1] + bbox[3] - 1 + offset
        )
        yield AABBox(top_left, bottom_right)


def draw_aabbox(
        out_img,
        list_bbox,
        thickness=3,
        color=None,
):
    """

    :param out_img:
    :param list_bbox:
    :param thickness:
    :param color:
    """
    for bbox in list_bbox:
        cv2.rectangle(out_img, tuple(bbox.top_left), tuple(bbox.bottom_right),
                      color if color else random_color(), thickness)


# def bbox_perimeter(bb):
#     return (abs(bb[1][0] - bb[0][0]) + abs(bb[1][1] - bb[0][1])) * 2


# def bbox_center(bb):
#     return (bb[1][0] + bb[0][0]) * 0.5, (bb[1][1] + bb[0][1]) * 0.5


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


def hline_intersect_bbox(y: float, bb: AABBox) -> bool:
    return bb.top_left.y <= y <= bb.bottom_right.y


def cluster_bbox_with_rectilines(bbox):
    id_cluster = 0
    clusters = []
    it_bbox = iter(bbox)
    try:
        bb = next(it_bbox)
        while True:
            nb_elements_in_cluster = 1
            bb_center = bb.compute_center()

            next_bb = next(it_bbox)
            while hline_intersect_bbox(bb_center.y, next_bb):
                nb_elements_in_cluster += 1
                next_bb = next(it_bbox)

            clusters += [id_cluster] * nb_elements_in_cluster
            id_cluster += 1

            bb = next_bb
    except StopIteration:
        pass

    return clusters


@dataclass
class AABBoxParams:
    img_fn: str
    filter_min_perimeter: int
    filter_max_perimeter: int

    path_img: Path = field(init=False)

    def __post_init__(self):
        self.path_img = Path(get_module_path_datasets()) / self.img_fn


def compute_aabbox_from_img_and_mask(params: AABBoxParams):
    img_path = Path(params.path_img)
    if not img_path.exists():
        raise IOError(f"{img_path} does'nt exist !")

    # Mask created with GIMP: image>mode>rgb tools>color_tools>colorize
    # https://docs.gimp.org/2.10/fr/gimp-tool-threshold.html
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

    # def cluster_on_bbox_perimeters(list_bbox, n_clusters=3):
    #     X = np.array(list(zip(
    #         [
    #             bbox.compute_perimeter()
    #             for bbox in list_bbox
    #         ],
    #         np.zeros(len(list_bbox)))
    #     ))
    #     kmeans = KMeans(n_clusters=n_clusters).fit(X)
    #     labels = kmeans.predict(X)
    #     return labels
    # labels = cluster_on_bbox_perimeters(bbox)
    # labels_bbox = [[] for _ in range(3)]
    # for label, bb in zip(labels, bbox):
    #     labels_bbox[label].append(bb)

    bbox = list(filter(
        lambda bb: params.filter_min_perimeter <= bb.compute_perimeter() <=
                   params.filter_max_perimeter,
        bbox
    ))
    bbox = sorted(bbox, key=lambda bb: (bb.top_left.y, bb.top_left.x))
    logger.info(f"bbox =\n{pprint.pformat(bbox)}")
    logger.info(f"perimeter: min={min(bbox, key=lambda bb: bb.perimeter)}")
    logger.info(f"perimeter: max={max(bbox, key=lambda bb: bb.perimeter)}")
    draw_aabbox(im_result, bbox, thickness=1, color=(0, 255, 0))
    draw_aabbox(im, bbox, thickness=1, color=(255, 0, 0))
    draw_aabbox(im_mss, bbox, thickness=1, color=(255, 0, 0))

    #
    cv2.imshow("img_for_contours", img_for_contours)
    cv2.imshow('Sprite Sheets - Mask', im)
    cv2.imshow('Sprite Sheets - Results', im_result)
    cv2.imshow(f'Sprite Sheets on {params.img_fn}', im_mss)

    # cluster_bbox(bbox)
    logger.info(cluster_bbox_with_rectilines(bbox))


def compute_and_render(
        list_params_aabbox_searching: List[AABBoxParams],
        wait_for_escape: bool = True
):
    for params_aabbox_searching in list_params_aabbox_searching:
        compute_aabbox_from_img_and_mask(params_aabbox_searching)
    #
    while True and wait_for_escape:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def main():
    compute_and_render([
        AABBoxParams('mercedesspritesheets.png', 132, 236),
        AABBoxParams('trump_run.png', 230, 290),
        AABBoxParams('volt_sprite_sheet_by_kwelfury-d5hx008.png', 632, 676)
    ])


if __name__ == '__main__':
    init_logger(logger)
    main()
