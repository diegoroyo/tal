from tal.enums import HFormat, GridFormat
from tal.config import get_memory_usage, get_resources
import numpy as np
from tqdm import tqdm
from numba import njit


def backproject(H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz,
                camera_system, t_accounts_first_and_last_bounces,
                t_start, delta_t,
                projector_focus=None,
                laser_xyz=None, sensor_xyz=None, progress=False):

    nt, nl, ns = H_0.shape
    nv, _ = volume_xyz.shape
    assert laser_grid_xyz.shape[0] == nl, 'H does not match with laser_grid_xyz'
    assert sensor_grid_xyz.shape[0] == ns, 'H does not match with sensor_grid_xyz'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'

    # reshape everything into (nl, nv, ns, 3)
    laser_xyz = laser_xyz.reshape((1, 1, 1, 3)).astype(np.float32)
    sensor_xyz = sensor_xyz.reshape((1, 1, 1, 3)).astype(np.float32)
    laser_grid_xyz = laser_grid_xyz.reshape((nl, 1, 1, 3)).astype(np.float32)
    sensor_grid_xyz = sensor_grid_xyz.reshape((1, 1, ns, 3)).astype(np.float32)
    volume_xyz = volume_xyz.reshape((1, nv, 1, 3)).astype(np.float32)

    if camera_system.implements_projector():
        assert projector_focus is not None and len(projector_focus) == 3, \
            'projector_focus is required for this camera system, should be a 3D point'
        projector_focus = np.array(projector_focus).reshape(
            (1, 1, 1, 3)).repeat(nv, axis=1)
    else:
        assert projector_focus is None, \
            'projector_focus must not be set for this camera system'
        projector_focus = volume_xyz.reshape((1, nv, 1, 3))
    projector_focus = projector_focus.astype(np.float32)

    def distance(a, b):
        return np.linalg.norm(b - a, axis=-1)

    if camera_system.is_transient():
        H_1 = np.zeros((nt, nv), dtype=H_0.dtype)
    else:
        H_1 = np.zeros(nv, dtype=H_0.dtype)

    if t_accounts_first_and_last_bounces:
        d_1 = distance(laser_xyz, laser_grid_xyz)
        d_4 = distance(sensor_grid_xyz, sensor_xyz)
    else:
        d_1 = np.float32(0.0)
        d_4 = np.float32(0.0)

    if camera_system.bp_accounts_for_d_2():
        d_2 = distance(laser_grid_xyz, projector_focus)
    else:
        d_2 = np.float32(0.0)

    def backproject_i(subrange_s):
        nsi = len(subrange_s)
        sensor_grid_xyz_i = sensor_grid_xyz[:, :, subrange_s, :]
        d_3 = distance(volume_xyz, sensor_grid_xyz_i)

        d = t_start + d_1 + d_2 + d_3 + d_4
        d /= delta_t
        d = d.astype(np.int32)
        idx = d

        if camera_system.is_transient():
            @njit(fastmath=True, parallel=True)
            def jit(nt, nv, nl, nsi, idx, H_0, subrange_s):
                range_t = np.arange(nt, dtype=np.int32)
                H_1 = np.zeros((nt, nv), dtype=H_0.dtype)
                for ti in range_t:
                    for li in range(nl):
                        for nv in range(nv):
                            for ls in range(nsi):
                                if idx[ti, li, ls] < 0 or idx[ti, li, ls] >= nt:
                                    continue
                                H_1[ti, nv] += H_0[idx[ti, li, ls],
                                                   li, subrange_s[ls]]
                return H_1
            return jit(nt, nv, nl, nsi, idx, H_0, subrange_s)
        else:
            i_l, i_v, i_s = np.ogrid[:nl, :nv, :nsi]
            good = np.logical_and(0 <= idx, idx < nt)

            def gather_idx(idx):
                H_1_i = H_0[idx[i_l, i_v, i_s], i_l, i_s]
                H_1_i[~good] = 0.0
                return H_1_i

            return gather_idx(idx).sum(axis=2).sum(axis=0)

        # range_t = np.arange(nt, dtype=np.int32)

        # if progress:
        #     range_t = tqdm(range_t, leave=False)

        # H_1 = np.zeros((nt, nv), dtype=H_0.dtype)
        # for t_i in range_t:
        #     H_1[t_i, :] = gather_idx(idx + t_i).sum(axis=2).sum(axis=0)
        # return H_1

    range_s = np.arange(ns, dtype=np.int32)
    h = H_0.dtype.itemsize
    s = sensor_grid_xyz.dtype.itemsize

    get_resources().split_work(
        backproject_i,
        data_in=range_s,
        data_out=H_1,
        f_mem_usage=lambda dc: (
            lambda _, cpus:
            get_memory_usage((H_0.shape, h * (4.75 * cpus)),
                             (H_1.shape, h * (2 * cpus)))
        )(*dc),
        slice_dims=(0, None),
    )

    return H_1


def backproject_old(H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz,
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
