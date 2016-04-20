import cv2
import numpy as np
import sys
import os

import cPickle as pickle


TUP_RANGE_PARAMS_RATIO = (0.0, (0, 1), 1000)


def get_attribs(params):
    return [attr for attr in dir(params) if not callable(attr) and not attr.startswith("__")]


def print_params(params):
    attribs = get_attribs(params)
    for attr in attribs:
        print attr, getattr(params, attr)


class params_for_pickle:
    """Summary
    """

    def __init__(self, params):
        self._params = params
        self._attr = get_attribs(params)
        self._dict_pickle = dict()

    def __getstate__(self):
        """Return state values to be pickled."""
        return (dict(zip(self._attr, [getattr(params, attr) for attr in self._attr])))

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self._dict_pickle = state

    def create_params(self):
        params = cv2.SimpleBlobDetector_Params()
        for (attr, value) in self._dict_pickle.iteritems():
            setattr(params, attr, value)
        return params


# url: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
def onChange(x):
    global update_window
    update_window = True


def update_TrackBars_From_Params(params, nameWindow, dict_range_params):
    list_attr = set(get_attribs(params))
    # filters
    list_filters = set([attr for attr in list_attr if 'filterBy' in attr])
    # min/max parameters
    list_min_params = set([attr[3:] for attr in list_attr if 'min' in attr])
    list_max_params = set([attr[3:] for attr in list_attr if 'max' in attr])
    # minmax parameters
    list_minmax_params = list_min_params.intersection(list_max_params)
    # min (only) parameters
    list_min_only_params = list_min_params.difference(list_max_params)

    for filter in list_filters:
        # url: http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
        cv2.setTrackbarPos(filter, nameWindow, getattr(params, filter))

    for minmax_param in list_minmax_params:
        trackbarname_param_min = 'min' + minmax_param
        trackbarname_param_max = 'max' + minmax_param

        current_value, (param_min, param_max), param_remap = dict_range_params.get(minmax_param, TUP_RANGE_PARAMS_RATIO)
        cv2.setTrackbarPos(
            ('%' if param_remap != 1 else '') + trackbarname_param_min,
            nameWindow, int(getattr(params, trackbarname_param_min) * param_remap)
        )

        cv2.setTrackbarPos(
            ('%' if param_remap != 1 else '') + trackbarname_param_max,
            nameWindow, int(getattr(params, trackbarname_param_max) * param_remap)
        )

    for min_param in list_min_only_params:
        trackbarname_param_min = 'min' + minmax_param

        current_value, (param_min, param_max), param_remap = dict_range_params.get(min_param, TUP_RANGE_PARAMS_RATIO)
        cv2.setTrackbarPos(
            ('%' if param_remap != 1 else '') + trackbarname_param_min,
            nameWindow, int(getattr(params, trackbarname_param_min) * param_remap)
        )


def createTrackBar(params, nameWindow, callback, dict_range_params={}):
    list_attr = set(get_attribs(params))
    # filters
    list_filters = set([attr for attr in list_attr if 'filterBy' in attr])
    # min/max parameters
    list_min_params = set([attr[3:] for attr in list_attr if 'min' in attr])
    list_max_params = set([attr[3:] for attr in list_attr if 'max' in attr])
    # minmax parameters
    list_minmax_params = list_min_params.intersection(list_max_params)
    # min (only) parameters
    list_min_only_params = list_min_params.difference(list_max_params)

    print "list_min_only_params:", list_min_only_params

    for filter in list_filters:
        cv2.createTrackbar(filter, nameWindow, getattr(params, filter), 1, callback)

    for minmax_param in list_minmax_params:
        trackbarname_param_min = 'min' + minmax_param
        trackbarname_param_max = 'max' + minmax_param

        current_value, (param_min, param_max), param_remap = dict_range_params.get(minmax_param, TUP_RANGE_PARAMS_RATIO)
        # REMAP
        current_value *= param_remap
        param_min *= param_remap
        param_max *= param_remap

        cv2.createTrackbar(
            ('%' if param_remap != 1 else '') + trackbarname_param_min,
            nameWindow, int(current_value), int(param_max), onChange)

        cv2.createTrackbar(
            ('%' if param_remap != 1 else '') + trackbarname_param_max,
            nameWindow, int(param_max), int(param_max), onChange)

    for min_param in list_min_only_params:
        trackbarname_param_min = 'min' + min_param

        current_value, (param_min, param_max), param_remap = dict_range_params.get(min_param, TUP_RANGE_PARAMS_RATIO)
        # REMAP
        current_value *= param_remap
        param_min *= param_remap
        param_max *= param_remap

        cv2.createTrackbar(
            ('%' if param_remap != 1 else '') + trackbarname_param_min,
            nameWindow, int(current_value), int(param_max), onChange)


def updateParams(params, nameWindow, dict_range_params={}):
    list_attr = set([attr for attr in dir(params) if not callable(attr) and not attr.startswith("__")])
    # filters
    list_filters = set([attr for attr in list_attr if 'filterBy' in attr])
    # min/max parameters
    list_min_params = set([attr[3:] for attr in list_attr if 'min' in attr])
    list_max_params = set([attr[3:] for attr in list_attr if 'max' in attr])
    # minmax parameters
    list_minmax_params = list_min_params.intersection(list_max_params)
    # min (only) parameters
    list_min_only_params = list_min_params.difference(list_max_params)

    for filter in list_filters:
        # print filter, bool(cv2.getTrackbarPos(filter, nameWindow))
        setattr(params, filter, bool(cv2.getTrackbarPos(filter, nameWindow)))

    for minmax_param in list_minmax_params:
        current_value, (param_min, param_max), param_remap = dict_range_params.get(minmax_param, (0.5, (0, 1), 1000))
        min_param = 'min' + minmax_param
        max_param = 'max' + minmax_param
        trackbarname_param_min = ('%' if param_remap != 1 else '') + min_param
        trackbarname_param_max = ('%' if param_remap != 1 else '') + max_param
        # print trackbarname_param_min, cv2.getTrackbarPos(trackbarname_param_min, nameWindow)
        setattr(params, min_param, float(cv2.getTrackbarPos(trackbarname_param_min, nameWindow)) / float(param_remap))
        setattr(params, max_param, float(cv2.getTrackbarPos(trackbarname_param_max, nameWindow)) / float(param_remap))

    for min_only_param in list_min_only_params:
        current_value, (param_min, param_max), param_remap = dict_range_params.get(min_only_param, (0.5, (0, 1), 1000))
        min_param = 'min' + min_only_param
        trackbarname_param_min = ('%' if param_remap != 1 else '') + min_param
        # print min_param, float(cv2.getTrackbarPos(trackbarname_param_min, nameWindow)) / float(param_remap)
        setattr(params, min_param, int((cv2.getTrackbarPos(trackbarname_param_min, nameWindow)) / float(param_remap)))

#
# filename = "Page_09_HD.jpg"
# filename = "Page_09.jpg"
#
filename = "Page_09_Pattern_23.png"
# filename = "Page_09_Pattern_24.png"
# filename = "Page_09_Pattern_25.png"
# filename = "Page_09_Pattern_26.png"

# url: https://github.com/spmallick/learnopencv/blob/master/BlobDetector/blob.py
# Read image
im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
im = cv2.fastNlMeansDenoising(im, None, 10, 7, 21)
im = cv2.medianBlur(im, 5)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200

# Filters
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = True
params.filterByInertia = True

dict_ranges_params = {
    'Threshold': (10, (10, 200), 1),
    'Area': (256, (0, 1000), 1),
    'Circularity': (0.75, (0, 1), 1000),
    'DistBetweenBlobs': (10, (0, 256), 1),
    'Repeatability': (2, (0, 256), 1),
    'Threshold': (10, (0, 255), 1),
    'Convexity': (0.920, (0, 1), 1000),
    'InertiaRatio': (0.01, (0, 1), 1000)
}
nameWindow_Parameters = 'Parameters for SimpleBlobDetector'
cv2.namedWindow(nameWindow_Parameters, cv2.WINDOW_NORMAL)
createTrackBar(params, nameWindow_Parameters, onChange, dict_ranges_params)

nameWindow_Results = 'Results'
cv2.namedWindow(nameWindow_Results, cv2.WINDOW_NORMAL)
cv2.resizeWindow(nameWindow_Results, 640, 480)

update_window = True


def exit():
    # url: http://stackoverflow.com/questions/285289/exit-codes-in-python
    sys.exit(os.EX_OK)


def nothing():
    pass


def save():
    global params
    pickle_filename = filename[:-4] + 'blob_detection.p'
    print "Write params dict into: ", pickle_filename
    pickle.dump(params_for_pickle(params), open(pickle_filename, "wb"))


def load():
    # global params
    pickle_filename = filename[:-4] + 'blob_detection.p'
    print "Load params dict from: ", pickle_filename
    # url: https://docs.python.org/2/tutorial/errors.html
    try:
        params_from_pickle = pickle.load(open(pickle_filename, "rb"))
    except IOError:
        print 'cannot open', pickle_filename
    else:
        params = params_from_pickle.create_params()
        update_TrackBars_From_Params(params, nameWindow_Parameters, dict_ranges_params)
        onChange(None)

dict_key_event = {
    27: exit,
    ord('s'): save,
    ord('l'): load
}

while(1):
    k = cv2.waitKey(1) & 0xFF
    # url: http://www.tutorialspoint.com/python/dictionary_setdefault.htm
    dict_key_event.setdefault(k, nothing)()
    # if k == 27:
    #     break

    if update_window:
        updateParams(params, nameWindow_Parameters, dict_ranges_params)

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')

        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(im)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(
            im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show blobs
        cv2.imshow(nameWindow_Results, im_with_keypoints)

        update_window = False
