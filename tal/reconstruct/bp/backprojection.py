from tal.enums import HFormat, GridFormat
from tal.config import get_memory_usage, get_resources
import numpy as np
from tqdm import tqdm


def backproject(H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz,
                camera_system, t_accounts_first_and_last_bounces,
                t_start, delta_t,
                laser_xyz=None, sensor_xyz=None, progress=False):
    # TODO(diego): extend for multiple laser points
    # TODO(diego): drjit?
    assert H_0.ndim == 2 and laser_grid_xyz.size == 3, \
        'backproject only supports one laser point'

    nt, ns = H_0.shape
    assert sensor_grid_xyz.shape[0] == ns, 'H does not match with sensor_grid_xyz'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'
    ns, _ = sensor_grid_xyz.shape
    nv, _ = volume_xyz.shape

    if camera_system.is_transient():
        H_1 = np.zeros((nt, nv), dtype=H_0.dtype)
    else:
        H_1 = np.zeros(nv, dtype=H_0.dtype)

    # d_1: laser origin to laser illuminated point
    # d_2: laser illuminated point to x_v
    # d_3: x_v to sensor imaged point
    # d_4: sensor imaged point to sensor origin
    x_l = laser_grid_xyz.reshape(3)
    if t_accounts_first_and_last_bounces:
        d_1 = np.linalg.norm(laser_xyz - x_l)
    else:
        d_1 = 0.0

    def work(subrange_s):
        if camera_system.is_transient():
            H_1 = np.zeros((nt, nv), dtype=H_0.dtype)
        else:
            H_1 = np.zeros(nv, dtype=H_0.dtype)

        if progress:
            subrange_s = tqdm(subrange_s, leave=False)

        for s_i in subrange_s:
            x_s = sensor_grid_xyz[s_i, :]
            if t_accounts_first_and_last_bounces:
                d_4 = np.linalg.norm(x_s - sensor_xyz)
            else:
                d_4 = 0.0
            for i_v, x_v in enumerate(volume_xyz):
                if camera_system.bp_accounts_for_d_2():
                    d_2 = np.linalg.norm(x_l - x_v)
                else:
                    d_2 = 0.0
                d_3 = np.linalg.norm(x_v - x_s)
                t_i = int((d_1 + d_2 + d_3 + d_4 - t_start) / delta_t)
                if camera_system.is_transient():
                    p = np.copy(H_0[:, s_i])
                    if t_i > 0:
                        p[:t_i] = 0.0
                    elif t_i < 0:
                        p[t_i+nt-1:] = 0.0
                    H_1[:, i_v] += np.roll(p, -t_i)
                else:
                    if t_i >= 0 and t_i < nt:
                        H_1[i_v] += H_0[t_i, s_i]
        return H_1

    h = H_0.dtype.itemsize
    s = sensor_grid_xyz.dtype.itemsize

    range_s = np.arange(ns, dtype=np.int32)

    get_resources().split_work(
        work,
        data_in=range_s,
        data_out=H_1,
        f_mem_usage=lambda dc: (
            lambda _, cpus:
            get_memory_usage(
                (sensor_grid_xyz.shape, s * cpus), (H_0.shape, (1 + h) * cpus), (H_1.shape, (1 + h) * cpus))
        )(*dc),
        slice_dims=(0, None),
    )

    return H_1
