import tal
from tal.reconstruct.pf import solve
from tal.reconstruct import get_volume

import matplotlib.pyplot as plt
import numpy as np

print('Start read')
data_path = 'D:/Documentos/nlos_dataset/rotated_mesh_cube_256x256/rotation[0]/data.hdf5'
data = tal.io.read_capture(data_path)
print('Finish read')

tal.plot.txy_interactive(data)

# a_corner = [-1, -1, 0.5]; b_corner = [1, 1, 1.5]
row_x = np.linspace(-1, 1, 256, dtype = float)
row_a = np.stack([row_x, np.full(256, -1)], axis = -1)
row_b = np.stack([row_x, np.full(256, 1)], axis = -1)
plane_xy = np.linspace(row_a, row_b, 256)
plane_a = np.stack([plane_xy[:,:,0], plane_xy[:,:,1], np.full((256,256), 0.5)], axis = -1)
plane_b = np.stack([plane_xy[:,:,0], plane_xy[:,:,1], np.full((256,256), 1.5)], axis = -1)
V = np.linspace(plane_a, plane_b, 32)
print(V.shape)

# reconstruction = solve(data, 6, 20, get_volume(), 1, 2)
reconstruction = solve(data, 6, 20, V, 2, 1)

image_rec = reconstruction.reshape(256,256)

print(image_rec.dtype)

plt.figure()
plt.imshow(np.abs(image_rec), cmap = 'hot')

plt.figure()
plt.imshow(np.angle(image_rec), cmap = 'bwr')

plt.show()