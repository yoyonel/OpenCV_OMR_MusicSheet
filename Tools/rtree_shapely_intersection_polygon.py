# sudo apt-get install libspatialindex-dev
# sudo pip install rtree
# url: http://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely

from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
from rtree import index
# import cv2
import numpy as np


def drawPolygons_matplotlib(list_polygons):
    # url: http://stackoverflow.com/questions/26935701/ploting-filled-polygons-in-python
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import sys

    fig, ax = plt.subplots()

    patches = []
    # url: http://stackoverflow.com/questions/3477283/maximum-float-in-python
    xmin, xmax, ymin, ymax = (sys.float_info.max, -sys.float_info.max, sys.float_info.max, -sys.float_info.max)
    for polygon in list_polygons:
        # url: http://stackoverflow.com/questions/20474549/extract-points-coordinates-from-python-shapely-polygon
        poly_x, poly_y = poly_xy = polygon.exterior.coords.xy
        # print poly_x, poly_y, min(poly_x)
        xmin = min(xmin, min(poly_x))
        xmax = max(xmax, max(poly_x))
        ymin = min(ymin, min(poly_y))
        ymax = max(ymax, max(poly_y))

        pts_polygon = np.dstack(np.array(poly_xy))[0]
        # print "pts_polygon: ", pts_polygon
        polygon = Polygon(pts_polygon, True)
        patches.append(polygon)

    # url: http://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    colors = 100 * np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)

    plt.show()


# Example polygon
xy = [[130.21001, 27.200001], [129.52, 27.34], [129.45, 27.1], [130.13, 26.950001]]
polygon_shape = Polygon(xy)
polygons = [polygon_shape]

# w, h = (512, 512)
# image = np.zeros((h + 2, w + 2), np.uint8)
# url: http://stackoverflow.com/questions/17960441/in-numpy-how-to-zip-two-2-d-arrays
# pts_polygon = np.array(polygon_shape.exterior.coords.xy)
# url: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ndarray.astype.html
# pts_polygon = np.dstack(pts_polygon).astype(int)
# print pts_polygon
# url: http://opencv-python-tutroals.readthedocs.or/gen/latest/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
# cv2.polylines(image, pts_polygon, True, (255, 0, 0), 1)
# cv2.imshow('image', image)
# cv2.waitKey(0)

# Example grid cell
gridcell_shape = box(129.5, -27.0, 129.75, 27.25)

grid_cells = [gridcell_shape]

# The intersection
# polygon_shape.intersection(gridcell_shape).area

idx = index.Index()

# Populate R-tree index with bounds of grid cells
for pos, cell in enumerate(grid_cells):
    # assuming cell is a shapely object
    idx.insert(pos, cell.bounds)

# Loop through each Shapely polygon
polygons_intersections = []
for poly in polygons:
    # Merge cells that have overlapping bounding boxes
    merged_cells = cascaded_union([grid_cells[pos] for pos in idx.intersection(poly.bounds)])
    # print merged_cells
    # Now do actual intersection
    poly_intersection = poly.intersection(merged_cells)
    print "Resultat ! Aire d'intersection: ", poly_intersection.area
    polygons_intersections.append(poly_intersection)

drawPolygons_matplotlib(polygons + grid_cells + polygons_intersections)
