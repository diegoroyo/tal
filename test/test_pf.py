import tal
from tal.reconstruct.pf import solve
from tal.reconstruct import get_volume
from tal.plot import plot_zxy_interactive

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print('Start read')
    data_path = 'D:/Documentos/nlos_dataset/rotated_mesh_cube_256x256/rotation[0]/data.hdf5'
    data = tal.io.read_capture(data_path)
    print('Finish read')

    tal.plot.txy_interactive(data)

    row_x = np.linspace(-0.99609375, 0.99609375, 256, dtype = float)
    row_a = np.stack([row_x, np.full(256, -0.99609375)], axis = -1)
    row_b = np.stack([row_x, np.full(256, 0.99609375)], axis = -1)
    plane_xy = np.linspace(row_a, row_b, 256)
    plane_a = np.stack([plane_xy[:,:,0], plane_xy[:,:,1], np.full((256,256), 0.5)], axis = -1)
    plane_b = np.stack([plane_xy[:,:,0], plane_xy[:,:,1], np.full((256,256), 1.5)], axis = -1)
    V = np.linspace(plane_a, plane_b, 256)

    # reconstruction = solve(data, 6, 20, get_volume(), 1, 2)
    reconstruction = solve(data, 6, 20, V, 3, 8)

    plot_zxy_interactive(np.abs(reconstruction))
    plot_zxy_interactive(np.angle(reconstruction), cmap = 'bwr')