from tal.enums import HFormat, GridFormat
import numpy as np
from tqdm import tqdm


def backproject(data, volume_xyz, camera_system, progress=False):
    assert data.H_format == HFormat.T_Sx_Sy, \
        'Backprojection solve only supports H in (T_Sx_Sy) format yet'
    assert data.sensor_grid_format == GridFormat.X_Y_3, \
        'Backprojection solve only supports H in (T_Sx_Sy) format yet'
    H = data.H
    nt, nsx, nsy = H.shape
    nv, _ = volume_xyz.shape

    if camera_system.is_transient():
        f = np.zeros((nt, nv), dtype=H.dtype)
    else:
        f = np.zeros(nv, dtype=H.dtype)

    # d_1: laser origin to laser illuminated point
    # d_2: laser illuminated point to x_v
    # d_3: x_v to sensor imaged point
    # d_4: sensor imaged point to sensor origin
    # TODO(diego) outer loops, nlx, nly, etc.
    x_l = data.laser_grid_xyz.reshape(3)
    if data.t_accounts_first_and_last_bounces:
        d_1 = np.linalg.norm(data.laser_xyz - x_l)
    else:
        d_1 = 0.0
    range_sx = range(nsx)
    range_sy = range(nsy)
    if progress:
        range_sx = tqdm(range_sx)
        range_sy = tqdm(range_sy, leave=False)
    for sx in range_sx:
        for sy in range_sy:
            x_s = data.sensor_grid_xyz[sx, sy, :]
            if data.t_accounts_first_and_last_bounces:
                d_4 = np.linalg.norm(x_s - data.sensor_xyz)
            else:
                d_4 = 0.0
            for i_v, x_v in enumerate(volume_xyz):
                if camera_system.bp_accounts_for_d_2():
                    d_2 = 0.0
                else:
                    d_2 = np.linalg.norm(x_l - x_v)
                d_3 = np.linalg.norm(x_v - x_s)
                t_i = int((d_1 + d_2 + d_3 + d_4 - data.t_start) / data.delta_t)
                if camera_system.is_transient():
                    p = np.copy(H[:, sx, sy])
                    p[t_i+nt-1:] = 0.0
                    p[:t_i] = 0.0
                    f[:, i_v] += np.roll(p, -t_i)
                else:
                    if t_i >= 0 and t_i < nt:
                        f[i_v] += H[t_i, sx, sy]

    return f
