from scipy import spatial
import numpy as np
import heapq
import math


def euclideanDistance(x, y):
    # print "x: ", x
    # print "y: ", y
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(x, y)]))

x, y, z = np.mgrid[0:5, 2:8, 2:3]
data = zip(x.ravel(), y.ravel(), z.ravel())
print data
tree = spatial.KDTree(data)
print 'ball', [data[i] for i in tree.query_ball_point(np.array([1, 2, 2]), 1)]
distance, index = tree.query(np.array([[2, 2, 2.2]]))
print 'query', distance, index, data[index[0]]
pts = np.array([[2, 2, 2.2]])

tree.query(pts)

closestPoints = heapq.nsmallest(
    1,
    enumerate(pts),
    key=lambda y: euclideanDistance(x[0], y[1])
)
print closestPoints
