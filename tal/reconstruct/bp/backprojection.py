from tal.enums import HFormat, GridFormat
from tal.config import get_resources
from tal.log import log, LogLevel, TQDMLogRedirect
import numpy as np
from tqdm import tqdm


def backproject(H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz, volume_xyz_shape,
                camera_system, t_accounts_first_and_last_bounces,
                t_start, delta_t, is_laser_paired_to_sensor,
                projector_focus=None,
                laser_xyz=None, sensor_xyz=None,
                compensate_invsq=False, progress=False):

    if camera_system.is_transient():
        log(LogLevel.WARNING, 'tal.reconstruct.bp: You have specified a time-resolved camera_system. '
            'The tal.reconstruct.bp implementation is better suited for time-gated systems. '
            'This will work, but you may want to check out tal.reconstruct.pf_dev for time-resolved reconstructions.')

    nt, nl, ns = H_0.shape
    nv, _ = volume_xyz.shape
    if is_laser_paired_to_sensor:
        assert laser_grid_xyz.shape[0] == ns, 'H does not match with laser_grid_xyz'
    else:
        assert laser_grid_xyz.shape[0] == nl, 'H does not match with laser_grid_xyz'
    assert sensor_grid_xyz.shape[0] == ns, 'H does not match with sensor_grid_xyz'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'

    # reshape everything into (nl, nv, ns, 3)
    if laser_xyz is not None:
        laser_xyz = laser_xyz.reshape((1, 1, 1, 3)).astype(np.float32)
    if sensor_xyz is not None:
        sensor_xyz = sensor_xyz.reshape((1, 1, 1, 3)).astype(np.float32)
    if is_laser_paired_to_sensor:
        laser_grid_xyz = laser_grid_xyz.reshape(
            (1, 1, ns, 3)).astype(np.float32)
    else:
        laser_grid_xyz = laser_grid_xyz.reshape(
            (nl, 1, 1, 3)).astype(np.float32)
    sensor_grid_xyz = sensor_grid_xyz.reshape((1, 1, ns, 3)).astype(np.float32)
    volume_xyz = volume_xyz.reshape((1, nv, 1, 3)).astype(np.float32)

    if camera_system.implements_projector():
        assert projector_focus is not None, 'projector_focus is required for this camera system'
        assert projector_focus.size == 3, \
            'When using tal.reconstruct.bp, projector_focus must be a single 3D point. ' \
            'If you want to focus the illumination aperture at multiple points, ' \
            'please use tal.reconstruct.pf_dev instead or call tal.reconstruct.bp once per projector_focus.'
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
        H_1 = np.zeros((nt, *volume_xyz_shape), dtype=H_0.dtype)
    else:
        H_1 = np.zeros(volume_xyz_shape, dtype=H_0.dtype)

    # d_1: laser origin to laser illuminated point
    # d_2: laser illuminated point to projector_focus
    # d_3: x_v (camera_focus) to sensor imaged point
    # d_4: sensor imaged point to sensor origin
    if t_accounts_first_and_last_bounces:
        d_1 = distance(laser_xyz, laser_grid_xyz)
        d_4 = distance(sensor_grid_xyz, sensor_xyz)
    else:
        d_1 = np.zeros((nl, 1, 1))
        d_4 = np.zeros((1, 1, ns))

    if camera_system.bp_accounts_for_d_2():
        d_2 = distance(laser_grid_xyz, projector_focus)
    else:
        d_2 = np.float32(0.0)

    def backproject_i(subrange_s):
        nsi = len(subrange_s)
        H_0_i = H_0[:, :, subrange_s]
        if is_laser_paired_to_sensor:
            d_2_i = d_2[:, :, subrange_s]
        else:
            d_2_i = d_2
        sensor_grid_xyz_i = sensor_grid_xyz[:, :, subrange_s, :]
        d_3 = 0
        if camera_system.bp_accounts_for_d_3():
            d_3 = distance(volume_xyz, sensor_grid_xyz_i)
        d_4_i = d_4[:, :, subrange_s]

        invsq = 1
        if compensate_invsq:
            def c(d):
                if isinstance(d, int) and d == 0:
                    return 1
                term = np.ones_like(d)
                epsilon = 1e-4
                term[d > epsilon] = d[d > epsilon] ** 2
                return term

            invsq = c(d_1) * c(d_2_i) * c(d_3) * c(d_4_i)

        idx = d_1 + d_2_i + d_3 + d_4_i - t_start
        idx /= delta_t
        idx = idx.astype(np.int32)

        t_range = nt if camera_system.is_transient() else 1
        t_range = np.arange(t_range, dtype=np.int32)
        if progress and len(t_range) > 1:
            t_range = tqdm(t_range, file=TQDMLogRedirect(),
                           desc='tal.reconstruct.bp time bins',
                           position=0,
                           leave=True)

        H_1_i = np.zeros((len(t_range), nv), dtype=H_0.dtype)
        i_v, i_s = np.ogrid[:nv, :nsi]
        for i_t in t_range:
            for i_l in range(nl):
                idx_i = idx[i_l, ...] + i_t
                good = np.logical_and(idx_i >= 0, idx_i < nt)
                idx_i[~good] = 0
                H_1_raw = H_0_i[idx_i[i_v, i_s], i_l, i_s]
                if compensate_invsq:
                    H_1_raw *= invsq[i_l]
                H_1_raw[~good] = 0.0
                H_1_i[i_t, :] += H_1_raw.sum(axis=1)

        if camera_system.is_transient():
            return H_1_i.reshape((nt, *volume_xyz_shape))
        else:
            return H_1_i[0].reshape(volume_xyz_shape)

    range_s = np.arange(ns, dtype=np.int32)

    get_resources().split_work(
        backproject_i,
        data_in=range_s,
        data_out=H_1,
        slice_dims=(0, None),
    )

    return H_1
