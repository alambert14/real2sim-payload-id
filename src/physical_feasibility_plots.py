import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import SpatialInertia, UnitInertia


def create_mesh_grid(width, height):
    x = np.linspace(-width, width, 101)
    z = np.linspace(-height, 0, 101)

    cxs, czs = np.meshgrid(x, z)

    y = np.zeros(cxs.shape)
    for i in range(cxs.shape[0]):
        for j in range(cxs.shape[1]):
            # Now I can't seem to make it physically inconsistent, even with crazy center of masses
            # Was there an issue with my calculations in the beginning?
            valid_inertia = SpatialInertia(
                0.6, [0, 0, 0], UnitInertia(0.05 / 0.6, 0.05 / 0.6, 7.5e-8 / 0.6),
                skip_validity_check=False)
            new_inertia = valid_inertia.Shift([cxs[i, j], 0, czs[i, j]])
            if new_inertia.IsPhysicallyValid():
                y[i, j] = 1
            else:
                y[i, j] = 0

    print(np.where(y > 0))
    print(y)
    # tl;dr its physically valid everywhere when I actually construct the inertia properly
    plt.pcolormesh(cxs, czs, y)
    plt.show()
