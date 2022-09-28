from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import HFormat, VolumeFormat, GridFormat
import numpy as np
from tqdm import tqdm


def backproject(data, volume_xyz, progress=False):
    assert data.H_format == HFormat.T_Sx_Sy, \
        'Backprojection solve only supports H in (T_Sx_Sy) format yet'
    assert data.sensor_grid_format == GridFormat.X_Y_3, \
        'Backprojection solve only supports H in (T_Sx_Sy) format yet'
    H = data.H

    f = np.zeros(volume_xyz.shape[0], dtype=H.dtype)
    nt, nsx, nsy = H.shape

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
    range_sx = range(0, nsx)
    range_sy = range(0, nsy)
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
                d_2 = np.linalg.norm(x_l - x_v)
                d_3 = np.linalg.norm(x_v - x_s)
                t_i = int((d_1 + d_2 + d_3 + d_4 - data.t_start) / data.delta_t)
                if t_i >= 0 and t_i < nt:
                    f[i_v] += H[t_i, sx, sy]

    return f
