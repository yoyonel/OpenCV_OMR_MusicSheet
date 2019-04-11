from scipy import spatial
import numpy as np
import heapq
import math


def euclidean_distance(x, y):
    # print "x: ", x
    # print "y: ", y
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(x, y)]))


def main():
    x, y, z = np.mgrid[0:5, 2:8, 2:3]
    data = list(zip(x.ravel(), y.ravel(), z.ravel()))
    print(data)
    tree = spatial.KDTree(data)
    print('ball',
          [data[i] for i in tree.query_ball_point(np.array([1, 2, 2]), 1)])
    distance, index = tree.query(np.array([[2, 2, 2.2]]))
    print('query', distance, index, data[index[0]])
    pts = np.array([[2, 2, 2.2]])

    tree.query(pts)

    closest_points = heapq.nsmallest(
        1, enumerate(pts),
        key=lambda y: euclidean_distance(x[0], y[1])
    )
    print(closest_points)


if __name__ == '__main__':
    main()
