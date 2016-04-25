import numpy as np
import matplotlib.pyplot as plt


def fit2DCurve(points, degree):
    # get x and y vectors
    x = points[:, 0]
    y = points[:, 1]

    z = np.polyfit(x, y, degree)

    f = np.poly1d(z)

    return f


def resample2DCurve(points, nb_samples):
    x = points[:, 0]
    x_new = np.linspace(x[0], x[-1], nb_samples)
    y_new = f(x_new)

    return (x_new, y_new)


def drawCurves(points, x_new, y_new):
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y, 'o', x_new, y_new)
    plt.xlim([x[0] - 1, x[-1] + 1])
    plt.show()


if __name__ == "__main__":
    points = np.array([(1, 1), (2, 4), (3, 1), (9, 3)])
    f = fit2DCurve(points, 3)
    print f
    
    # calculate new x's and y's
    nb_points = 50
    x_new, y_new = resample2DCurve(points, nb_points)

    drawCurves(points, x_new, y_new)
