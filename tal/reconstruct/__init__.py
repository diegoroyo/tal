from tal.reconstruct.pf import *


# def get_volume(center, rotation, size, resolution):
#     #TODO: Volume type up to Diego
#     raise NotImplemented('get_volume to be implemented')

def get_volume():
    nx = 256
    ny = 256
    center = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
    dx = 1 / np.sqrt(2)
    dx /= nx
    dy = 1
    dy /= ny
    dz = 1 / np.sqrt(2)
    dz /= nx

    x = np.linspace(center[0] - dx * nx // 2, center[0] + dx * nx // 2, nx)
    y = np.linspace(center[1] - dy * ny // 2, center[1] + dy * ny // 2, ny)
    z = np.linspace(center[2] - dz * nx // 2, center[2] + dz * nx // 2, nx)
    xyz = np.stack((
        np.repeat(x.reshape((nx, 1, 1, 1)), ny, axis=1),
        np.repeat(y.reshape((1, ny, 1, 1)), nx, axis=0),
        np.repeat(z.reshape((nx, 1, 1, 1)), ny, axis=1),
    ), axis=-1)

    return xyz.reshape((-1, 3))
