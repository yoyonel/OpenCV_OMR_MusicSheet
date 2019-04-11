"""
"""
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np


def main():
    fig, ax = plt.subplots()
    patches = []
    N = 5

    for i in range(N):
        polygon = Polygon(np.random.rand(N, 2), True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=get_cmap("jet"), alpha=0.4)

    colors = 100 * np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)

    plt.show()


if __name__ == '__main__':
    main()
