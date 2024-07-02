import numpy as np
import tal
from tal.config import get_resources
from tal.log import log, LogLevel, TQDMLogRedirect
from tqdm import tqdm


def backproject_pf_multi_frequency(
        H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz, volume_xyz_shape,
        camera_system, t_accounts_first_and_last_bounces,
        t_start, delta_t, is_laser_paired_to_sensor,
        projector_focus,
        frequencies, weights, nt,
        optimize_projector_convolutions, optimize_camera_convolutions,
        laser_xyz=None, sensor_xyz=None, progress=False):

    assert not is_laser_paired_to_sensor, 'tal.reconstruct.upf_dev does not support confocal or custom captures. ' \
        'Please use tal.reconstruct.bp or tal.reconstruct.fbp instead.'

    if not optimize_projector_convolutions and not optimize_camera_convolutions and not camera_system.is_transient():
        log(LogLevel.WARNING, 'tal.reconstruct.upf_dev: You have specified a time-gated camera system '
            'with an arbitrary reconstruction volume (that is not parallel to the relay wall). '
            'This will work, but the tal.reconstruct.bp or tal.reconstruct.fbp implementations '
            'are better suited for these cases.')

    nw, nl, ns = H_0.shape
    nv = np.prod(volume_xyz.shape[:-1])  # N or X * Y or X * Y * Z
    if optimize_projector_convolutions or optimize_camera_convolutions:
        if volume_xyz.ndim == 3:
            nvx, nvy, _ = volume_xyz.shape
            volume_xyz = volume_xyz.reshape((nvx, nvy, 1, 3))
        assert volume_xyz.ndim == 4, 'Expecting X_Y_Z_3 format'
        nvx, nvy, nvz, _ = volume_xyz.shape
    if optimize_projector_convolutions:
        assert laser_grid_xyz.ndim == 3, 'Expecting X_Y_3 format'
        nlx, nly, _ = laser_grid_xyz.shape
    else:
        nlx, nly = 1, 1
    if optimize_camera_convolutions:
        assert sensor_grid_xyz.ndim == 3, 'Expecting X_Y_3 format'
        nsx, nsy, _ = sensor_grid_xyz.shape
    else:
        nsx, nsy = 1, 1

    """ Phasor fields filter """

    if weights is None:
        weights = np.ones(nw, dtype=np.float32, dtype=np.float32)
    else:
        weights = weights.astype(np.complex64)
    freqs = frequencies.astype(np.float32)
    assert len(freqs) == len(weights) == nw, \
        'len(freqs), len(weights), and H_0.shape[0] are not the same.'

    t = np.arange(nt, dtype=np.float32) * delta_t
    padding = 0
    # min_freq = freqs[~np.isclose(freqs, 0)].min()
    # nt = int(np.ceil((1 / min_freq) / delta_t))
    # log(LogLevel.INFO, f'tal.reconstruct.upf_dev: computed nt is {nt}')

    """ Distance calculation and propagation """

    if camera_system.implements_projector():
        assert projector_focus is not None, \
            'projector_focus is required for this camera system'
        if projector_focus.size == 3:
            projector_focus = np.array(projector_focus).reshape(
                (1, 1, 1, 3))
            projector_focus_mode = 'single'
            if optimize_projector_convolutions:
                log(LogLevel.INFO, 'tal.reconstruct.upf_dev: When projector_focus is a 3D point, the projector convolution optimization is not implemented. '
                    'Falling back to default method.')
            optimize_projector_convolutions = False
        else:
            assert np.allclose(projector_focus.flatten(), volume_xyz.flatten()), \
                'projector_focus must be a single 3D point, ' \
                'or you should pass the same value as volume_xyz.'
            assert nvz == 1, \
                'When projector_focus=volume_xyz, the volume must be a single Z slice. ' \
                f'In your case, your volume has {nvz} Z slices. ' \
                'You can call this same function for each individual point in the volume.'
            projector_focus = projector_focus.reshape((1, nv, 1, 3))
            projector_focus_mode = 'exhaustive'
    else:
        assert projector_focus is None, \
            'projector_focus must not be set for this camera system'
        projector_focus = volume_xyz.reshape((1, nv, 1, 3))
        projector_focus_mode = 'confocal'
        if optimize_projector_convolutions:
            log(LogLevel.INFO, 'tal.reconstruct.upf_dev: When projector_focus is not set, the projector convolution optimization is not implemented. '
                'Falling back to default method.')
        optimize_projector_convolutions = False

    log(LogLevel.INFO, 'tal.reconstruct.upf_dev: '
        f'projector_focus_mode={projector_focus_mode}')
    projector_focus = projector_focus.astype(np.float32)

    # reshape everything into (nl, nv, ns, 3)
    if laser_xyz is not None:
        laser_xyz = laser_xyz.reshape((1, 1, 1, 3)).astype(np.float32)
    if sensor_xyz is not None:
        sensor_xyz = sensor_xyz.reshape((1, 1, 1, 3)).astype(np.float32)
    laser_grid_xyz = laser_grid_xyz.reshape((nl, 1, 1, 3)).astype(np.float32)
    sensor_grid_xyz = sensor_grid_xyz.reshape((1, 1, ns, 3)).astype(np.float32)
    volume_xyz = volume_xyz.reshape((1, nv, 1, 3)).astype(np.float32)

    def distance(a, b):
        return np.linalg.norm(b - a, axis=-1)

    # d_0: t_start (moment the sensor starts capturing w.r.t. pulse emission)
    # d_1: laser origin to laser illuminated point
    # d_2: laser illuminated point to projector_focus
    # d_3: x_v (camera_focus) to sensor imaged point
    # d_4: sensor imaged point to sensor origin
    d_0 = np.float32(t_start * -1)

    if t_accounts_first_and_last_bounces:
        d_1 = distance(laser_xyz, laser_grid_xyz).reshape((nl, 1))
        d_4 = distance(sensor_grid_xyz, sensor_xyz).reshape((1, ns))
    else:
        d_1 = np.float32(0.0)
        d_4 = np.float32(0.0)

    if projector_focus_mode == 'exhaustive':
        n_projector_points = nv
    else:
        n_projector_points = 1

    d_014 = d_0 + d_1 + d_4

    if optimize_projector_convolutions or optimize_camera_convolutions:
        nvi = nvx * nvy
        range_z = range(nvz)
    else:
        nvi = nv
        range_z = [0]
        nvz = 1
    range_z = enumerate(range_z)
    if nvz > 1 and progress:
        range_z = tqdm(
            range_z,
            desc='tal.reconstruct.pf_dev Z slices',
            total=nvz,
            file=TQDMLogRedirect(),
            position=0,
            leave=True)

    if camera_system.is_transient():
        H_1 = np.zeros(
            (nt, n_projector_points, nvi, nvz),
            dtype=np.complex64)
    else:
        H_1 = np.zeros(
            (n_projector_points, nvi, nvz),
            dtype=np.complex64)

    for i_z, nvzi in range_z:
        if optimize_projector_convolutions:
            projector_focus_i = projector_focus.reshape(
                (nvx, nvy, nvz, 3))[..., nvzi, :]
        else:
            projector_focus_i = projector_focus
        if optimize_camera_convolutions:
            volume_xyz_i = volume_xyz.reshape(
                (nvx, nvy, nvz, 3))[..., nvzi, :]
        else:
            volume_xyz_i = volume_xyz

        if camera_system.bp_accounts_for_d_2():
            if optimize_projector_convolutions:
                assert projector_focus_mode in ['exhaustive', 'confocal']
                rlx = nvx + nlx - 1
                rly = nvy + nly - 1
                laser_grid_xyz = laser_grid_xyz.reshape((nlx, nly, 3))
                l_dx = laser_grid_xyz[1, 0, ...] - \
                    laser_grid_xyz[0, 0, ...]
                l_dy = laser_grid_xyz[0, 1, ...] - \
                    laser_grid_xyz[0, 0, ...]

                p0 = projector_focus_i[0, 0, ...] - \
                    laser_grid_xyz[-1, -1, ...]

                i_vals, j_vals = np.meshgrid(
                    np.arange(rlx), np.arange(rly), indexing='ij')
                i_vals = i_vals.reshape((rlx, rly, 1))
                j_vals = j_vals.reshape((rlx, rly, 1))
                p0 = p0.reshape((1, 1, 3))
                l_dx = l_dx.reshape((1, 1, 3))
                l_dy = l_dy.reshape((1, 1, 3))
                d_2 = np.linalg.norm(
                    p0 + l_dx * i_vals + l_dy * j_vals, axis=-1).astype(np.float32)
                d_2 = np.fft.ifftshift(d_2)
            elif projector_focus_mode == 'confocal':
                d_2 = distance(
                    laser_grid_xyz, projector_focus_i.reshape((1, nvi, 1, 3)))
            else:
                assert projector_focus_mode == 'single'
                d_2 = distance(
                    laser_grid_xyz, projector_focus_i.reshape((1, 1, 1, 3)).repeat(nvi, axis=1))
        else:
            d_2 = np.float32(0.0)

        if optimize_camera_convolutions:
            rsx = nvx + nsx - 1
            rsy = nvy + nsy - 1
            sensor_grid_xyz = sensor_grid_xyz.reshape((nsx, nsy, 3))
            s_dx = sensor_grid_xyz[1, 0, ...] - \
                sensor_grid_xyz[0, 0, ...]
            s_dy = sensor_grid_xyz[0, 1, ...] - \
                sensor_grid_xyz[0, 0, ...]

            p0 = volume_xyz_i[0, 0, ...] - \
                sensor_grid_xyz[-1, -1, ...]

            i_vals, j_vals = np.meshgrid(
                np.arange(rsx), np.arange(rsy), indexing='ij')
            i_vals = i_vals.reshape((rsx, rsy, 1))
            j_vals = j_vals.reshape((rsx, rsy, 1))
            p0 = p0.reshape((1, 1, 3))
            s_dx = s_dx.reshape((1, 1, 3))
            s_dy = s_dy.reshape((1, 1, 3))
            d_3 = np.linalg.norm(
                p0 + s_dx * i_vals + s_dy * j_vals, axis=-1).astype(np.float32)
            d_3 = np.fft.ifftshift(d_3)
        else:
            d_3 = distance(volume_xyz_i, sensor_grid_xyz)

        nw_pow2 = 2 ** np.ceil(np.log2(nw)).astype(np.int32)
        freqs_pad = np.pad(freqs, (0, nw_pow2 - nw), 'constant')
        weights_pad = np.pad(weights, (0, nw_pow2 - nw), 'constant')

        def work_zslice_freq(range_w):
            nwi = len(range_w)
            freqs_i = freqs_pad[range_w]
            weights_i = weights_pad[range_w]
            H_1_w = np.zeros((nwi, n_projector_points, nvi),
                             dtype=np.complex64)

            fw_iterator = enumerate(zip(freqs_i, weights_i))
            if progress:
                fw_iterator = tqdm(fw_iterator,
                                   desc='tal.reconstruct.upf_dev propagation (1/2)',
                                   file=TQDMLogRedirect(),
                                   total=min(len(freqs_i), nw),
                                   position=0,
                                   leave=True)

            for i_w, (frequency, weight) in fw_iterator:
                if weight == 0.0:
                    # H_1_w[i_w, ...] = 0
                    continue

                H_0_w = H_0[range_w[i_w], ...].reshape((nl, ns))

                rsd_014 = np.exp(np.complex64(2j * np.pi) * d_014 * frequency)
                H_0_w *= rsd_014
                del rsd_014

                rsd_3 = np.exp(np.complex64(2j * np.pi) * d_3 * frequency)
                if optimize_camera_convolutions:
                    H_0_w = H_0_w.reshape((nlx, nly, nsx, nsy))
                    rsd_3 = rsd_3.reshape((1, 1, rsx, rsy))
                    H_0_w_fft = np.fft.fft2(
                        H_0_w, axes=(2, 3), s=(rsx, rsy))
                    rsd_3_fft = np.fft.fft2(
                        rsd_3, axes=(2, 3), s=(rsx, rsy))
                    H_0_w_fft *= rsd_3_fft
                    H_0_w = np.fft.ifft2(H_0_w_fft, axes=(2, 3))
                    H_0_w = H_0_w[:, :, :nvx, :nvy]
                    H_0_w = H_0_w.reshape((nl, nvi))
                    del rsd_3
                else:
                    H_0_w = H_0_w.reshape((nl, 1, ns))
                    H_0_w = H_0_w * rsd_3.reshape((1, nvi, ns))
                    H_0_w = H_0_w.sum(axis=2)

                if camera_system.bp_accounts_for_d_2():
                    rsd_2 = np.exp(np.complex64(2j * np.pi) * d_2 * frequency)
                    if projector_focus_mode == 'exhaustive':
                        assert optimize_projector_convolutions, \
                            'You must use the convolutions optimization when projector_focus=volume_xyz. ' \
                            'Check the documentation for tal.reconstruct.upf_dev for more information.'
                        H_0_w = H_0_w.reshape((nlx, nly, nvx, nvy))
                        rsd_2 = rsd_2.reshape((rlx, rly, 1, 1))
                        H_0_w_fft = np.fft.fft2(
                            H_0_w, axes=(0, 1), s=(rlx, rly))
                        rsd_2_fft = np.fft.fft2(
                            rsd_2, axes=(0, 1), s=(rlx, rly))
                        H_0_w_fft *= rsd_2_fft
                        H_0_w = np.fft.ifft2(H_0_w_fft, axes=(0, 1))
                        H_0_w = H_0_w[:nvx, :nvy, :, :]
                    else:
                        assert projector_focus_mode in ['single', 'confocal']
                        H_0_w = H_0_w.reshape((nl, nvi))
                        H_0_w *= rsd_2.reshape((nl, nvi))
                        H_0_w = H_0_w.sum(axis=0)
                    H_0_w = H_0_w.reshape((n_projector_points, nvi))
                    del rsd_2

                H_1_w[i_w, ...] = weight * H_0_w
                del H_0_w

            return H_1_w

        H_1_w = np.zeros((nw_pow2, n_projector_points, nvi),
                         dtype=np.complex64)
        range_w = np.arange(nw_pow2, dtype=np.int32)

        get_resources().split_work(
            work_zslice_freq,
            data_in=range_w,
            data_out=H_1_w,
            slice_dims=(0, 0),
        )

        # H_1 has shape (n_projector_points, nvi, nvz) or (nt, n_projector_points, nvi, nvz)
        # H_1_w has shape (nw_pow2, n_projector_points, nvi)
        f_iterator = enumerate(freqs)
        if progress:
            f_iterator = tqdm(f_iterator,
                              desc='tal.reconstruct.upf_dev ifft (2/2)',
                              file=TQDMLogRedirect(),
                              total=nw,
                              position=0,
                              leave=True)
        for i_w, frequency in f_iterator:
            e = np.exp(np.complex64(2j * np.pi) * t * frequency) / nt
            H_1_t = (
                e.reshape((nt, 1, 1))
                *
                H_1_w[i_w, ...].reshape((1, n_projector_points, nvi))
            )

            if camera_system.is_transient():
                H_1[..., i_z] += H_1_t[padding:-padding, ...]
            else:
                H_1[..., i_z] += H_1_t[padding, ...]

    if n_projector_points == 1:
        # squeeze n_projector_points axis
        H_1 = H_1.squeeze(axis=-3)

    return H_1
