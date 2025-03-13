import numpy as np
import tal
from tal.config import get_resources
from tal.log import log, LogLevel, TQDMLogRedirect
from tqdm import tqdm


def _get_padding(wl_sigma, delta_t):
    t_6sigma = int(np.ceil(6 * wl_sigma / delta_t))
    padding = 2 * t_6sigma
    return padding


def backproject_pf_multi_frequency(
        H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz, volume_xyz_shape,
        camera_system, t_accounts_first_and_last_bounces,
        t_start, delta_t, is_laser_paired_to_sensor,
        projector_focus,
        wl_mean, wl_sigma, border,
        optimize_projector_convolutions, optimize_camera_convolutions,
        laser_xyz=None, sensor_xyz=None,
        compensate_invsq=False,
        skip_H_fft=False, skip_H_padding=False, nt=None,
        progress=False):

    assert not is_laser_paired_to_sensor, 'tal.reconstruct.pf_dev does not support confocal or custom captures. ' \
        'Please use tal.reconstruct.bp or tal.reconstruct.fbp instead.'

    if skip_H_padding:
        assert skip_H_fft, 'skip_H_fft should be also set when skip_H_padding is set'
        assert nt is not None, 'nt is required when skip_H_padding is set'

    if not optimize_projector_convolutions and not optimize_camera_convolutions and not camera_system.is_transient():
        log(LogLevel.WARNING, 'tal.reconstruct.pf_dev: You have specified a time-gated camera system '
            'with an arbitrary reconstruction volume (that is not parallel to the relay wall). '
            'This will work, but the tal.reconstruct.bp or tal.reconstruct.fbp implementations '
            'are better suited for these cases.')

    nt_, nl, ns = H_0.shape
    nt_ = nt or nt_
    nv = np.prod(volume_xyz.shape[:-1])  # N or X * Y or X * Y * Z
    if projector_focus is None:
        npf = 0
    elif projector_focus.size == 3:
        npf = 1
    else:
        npf = np.prod(projector_focus.shape[:-1])
    if optimize_projector_convolutions:
        assert laser_grid_xyz.ndim == 3, 'Expecting X_Y_3 format'
        nlx, nly, _ = laser_grid_xyz.shape

        if projector_focus.ndim == 3:
            npfx, npfy, _ = projector_focus.shape
            projector_focus = projector_focus.reshape((npfx, npfy, 1, 3))
        assert projector_focus.ndim == 4, 'Expecting X_Y_Z_3 format'
        npfx, npfy, npfz, _ = projector_focus.shape
    else:
        nlx, nly = 1, 1
        npfz = 1
    if optimize_camera_convolutions:
        assert sensor_grid_xyz.ndim == 3, 'Expecting X_Y_3 format'
        nsx, nsy, _ = sensor_grid_xyz.shape

        if volume_xyz.ndim == 3:
            nvx, nvy, _ = volume_xyz.shape
            volume_xyz = volume_xyz.reshape((nvx, nvy, 1, 3))
        assert volume_xyz.ndim == 4, 'Expecting X_Y_Z_3 format'
        nvx, nvy, nvz, _ = volume_xyz.shape
    else:
        nsx, nsy = 1, 1
        nvz = 1

    """ Phasor fields filter """

    if skip_H_padding:
        padding = 0
    else:
        padding = _get_padding(wl_sigma, delta_t)
    # FIXME(diego) if we want to convert a circular convolution to linear,
    # this should be nt + t_6sigma - 1 instead of nt + 4 * t_6sigma or even nt + 2 * t_6sigma
    # I have found cases where it fails even with nt + 2 * t_6sigma (Z, 0th pixel)
    nf = nt + 2 * padding

    t_max = delta_t * (nf - 1)
    t = np.linspace(start=0, stop=t_max, num=nf, dtype=np.float32)

    gaussian_envelope = np.exp(-((t - t_max / 2) / wl_sigma) ** 2 / 2)
    K = gaussian_envelope / np.sum(gaussian_envelope) * \
        np.exp(2j * np.pi * t / wl_mean)
    K = np.fft.fftshift(K)  # center at zero

    mean_idx = (nf * delta_t) / wl_mean
    sigma_idx = (nf * delta_t) / (wl_sigma * 6)
    # shift to center at zero, easier for low negative frequencies
    freq_min_idx = np.maximum(0, int(np.floor(mean_idx - 3 * sigma_idx)))
    freq_max_idx = np.minimum(nf // 2, int(np.ceil(mean_idx + 3 * sigma_idx)))

    weights = np.fft.fft(K)[freq_min_idx:freq_max_idx+1].astype(np.complex64)
    freqs = np.fft.fftfreq(nf, d=delta_t)[
        freq_min_idx:freq_max_idx+1].astype(np.float32)
    freq_idxs = np.arange(freq_min_idx, freq_max_idx+1, dtype=np.int32)
    if skip_H_padding:
        freq_idxs -= freq_min_idx  # convert to (0, nw-1) range
    log(LogLevel.INFO, 'tal.reconstruct.pf_dev: '
        f'Using {len(freqs)} wavelengths from {1 / freqs[-1]:.4f}m to {1 / freqs[0]:.4f}m')
    nw = len(weights)

    if skip_H_fft:
        # NOTE: this is kind of a hack. When pre-computing the FFT you need to pad the signal
        # before. But if you pass a padded H then it messes up the nt and nf variables.
        # So the precompute_fft function removes the last values (they are not used anyway)
        # and it's re-padded here.
        if not skip_H_padding:
            H_0 = np.pad(H_0,
                         ((0, 2 * padding),) + ((0, 0),) * (H_0.ndim - 1), 'constant')
    else:
        if border == 'zero':
            # only pad temporal dimension
            H_0 = np.pad(H_0,
                         ((padding, padding),) + ((0, 0),) * (H_0.ndim - 1), 'constant')
        else:
            raise AssertionError('Implemented only for border="zero"')

    """ Distance calculation and propagation """

    if camera_system.implements_projector():
        assert projector_focus is not None, \
            'projector_focus is required for this camera system'
        if projector_focus.size == 3:
            projector_focus = np.array(projector_focus).reshape(
                (1, 1, 1, 3))
            projector_focus_mode = 'single'
            if optimize_projector_convolutions:
                log(LogLevel.INFO, 'tal.reconstruct.pf_dev: When projector_focus is a 3D point, the projector convolution optimization is not implemented. '
                    'Falling back to default method.')
            optimize_projector_convolutions = False
        else:
            assert npfz == 1, \
                'When projector_focus is a volume, it must be a single Z slice. ' \
                f'In your case, your volume has {nvz} Z slices. ' \
                'You can call this same function for each individual point in the volume.'
            projector_focus = projector_focus.reshape((1, npf, 1, 3))
            projector_focus_mode = 'exhaustive'
    else:
        assert projector_focus is None, \
            'projector_focus must not be set for this camera system'
        projector_focus = volume_xyz.reshape((1, nv, 1, 3))
        projector_focus_mode = 'confocal'
        if optimize_projector_convolutions:
            log(LogLevel.INFO, 'tal.reconstruct.pf_dev: When projector_focus is not set, the projector convolution optimization is not implemented. '
                'Falling back to default method.')
        optimize_projector_convolutions = False

    log(LogLevel.INFO, 'tal.reconstruct.pf_dev: '
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

    def invsq(d):
        term = np.ones_like(d)
        epsilon = 1e-4
        term[d > epsilon] = d[d > epsilon] ** 2
        return term

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
        n_projector_points = npf
    else:
        n_projector_points = 1 if camera_system.bp_accounts_for_d_2() else nl

    d_014 = d_0 + d_1 + d_4
    invsq_14 = 1
    if compensate_invsq:
        invsq_14 = invsq(d_1) * invsq(d_4)

    if optimize_projector_convolutions or optimize_camera_convolutions:
        nva = nvx * nvy
        range_z = range(nvz)
    else:
        nva = nv
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
            (nt, n_projector_points, nva, nvz),
            dtype=np.complex64)
    else:
        H_1 = np.zeros(
            (n_projector_points, nva, nvz),
            dtype=np.complex64)

    if skip_H_fft:
        assert np.isclose(d_014, 0.0) and np.isclose(invsq_14, 1), \
            'skip_H_fft is only supported when d_014=0 and invsq_14=1'
        H_0_w = H_0
    else:
        H_0_w = np.zeros_like(H_0, dtype=np.complex64)
        range_s = np.arange(ns)

        def fft_and_compensate_014(subrange_s):
            H_0_w = np.fft.fft(H_0[:, :, subrange_s], axis=0)
            if not np.isclose(d_014, 0.0):
                if d_014.size == 1:
                    d_014_reshaped = np.repeat(
                        np.repeat(d_014.reshape(1, 1, 1),
                                  repeats=nl, axis=1),
                        repeats=ns, axis=2)
                    invsq_14_reshaped = np.repeat(
                        np.repeat(invsq_14.reshape(1, 1, 1),
                                  repeats=nl, axis=1),
                        repeats=ns, axis=2)
                else:
                    d_014_reshaped = d_014.reshape(1, nl, ns)
                    invsq_14_reshaped = invsq_14.reshape(1, nl, ns)

                frequencies = np.fft.fftfreq(nf, d=delta_t).astype(np.float32)
                frequencies[:freq_min_idx] = 0
                frequencies[freq_max_idx+1:] = 0
                rsd_014 = np.exp(np.complex64(2j * np.pi) *
                                 d_014_reshaped[..., subrange_s] *
                                 frequencies.reshape(nf, 1, 1))
                rsd_014 *= invsq_14_reshaped[..., subrange_s]
                H_0_w *= rsd_014
                del rsd_014
            elif not np.isclose(invsq_14, 1):
                if invsq_14.size == 1:
                    invsq_14_reshaped = np.repeat(
                        np.repeat(invsq_14.reshape(1, 1, 1),
                                  repeats=nl, axis=1),
                        repeats=ns, axis=2)
                else:
                    invsq_14_reshaped = invsq_14.reshape(1, nl, ns)
                H_0_w *= invsq_14_reshaped[..., subrange_s]
            return H_0_w

        get_resources().split_work(
            fft_and_compensate_014,
            data_in=range_s,
            data_out=H_0_w,
            slice_dims=(0, 2),
        )

    propagation_mode = 'divide_frequencies' if optimize_projector_convolutions or optimize_camera_convolutions else 'divide_volume'
    if propagation_mode == 'divide_frequencies':
        log(LogLevel.INFO,
            'tal.reconstruct.pf_dev: Will use divide_frequencies implementation')
    elif propagation_mode == 'divide_volume':
        log(LogLevel.INFO,
            'tal.reconstruct.pf_dev: Will use divide_volume implementation')
    else:
        raise AssertionError('Unknown propagation mode')

    for i_z, nvzi in range_z:
        if optimize_projector_convolutions:
            projector_focus_i = projector_focus.reshape((npfx, npfy, 3))
        else:
            # FIXME does not work for multiple Z slices
            projector_focus_i = projector_focus
        if optimize_camera_convolutions:
            volume_xyz_i = volume_xyz.reshape(
                (nvx, nvy, nvz, 3))[..., nvzi, :]
        else:
            volume_xyz_i = volume_xyz

        if camera_system.bp_accounts_for_d_2():
            if optimize_projector_convolutions:
                assert projector_focus_mode in ['exhaustive', 'confocal']
                rlx = npfx + nlx - 1
                rly = npfy + nly - 1
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
                    laser_grid_xyz, projector_focus_i.reshape((1, nva, 1, 3)))
            else:
                assert projector_focus_mode == 'single'
                d_2 = distance(
                    laser_grid_xyz, projector_focus_i.reshape((1, 1, 1, 3)).repeat(nva, axis=1))
        else:
            d_2 = np.float32(0.0)

        if camera_system.bp_accounts_for_d_3():
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
        else:
            d_3 = np.float32(0.0)

        invsq_2 = 1
        invsq_3 = 1
        if compensate_invsq:
            invsq_2 = invsq(d_2)
            invsq_3 = invsq(d_3)

        if propagation_mode == 'divide_frequencies':

            def work_dividing_frequencies(subrange_w):
                nwi = len(subrange_w)
                freq_idxs_i = freq_idxs[subrange_w]
                freqs_i = freqs[subrange_w]
                weights_i = weights[subrange_w]

                H_1_w = np.zeros(
                    (nwi, n_projector_points, nva), dtype=np.complex64)

                fw_iterator = enumerate(zip(freq_idxs_i, freqs_i, weights_i))
                if progress:
                    fw_iterator = tqdm(fw_iterator,
                                       desc='tal.reconstruct.pf_dev divide-frequency',
                                       file=TQDMLogRedirect(),
                                       total=nwi,
                                       position=0,
                                       leave=True)

                for i_w, (f_idx, frequency, weight) in fw_iterator:
                    if np.isclose(weight, 0.0):
                        # H_1_w[f_idx, ...] = 0
                        continue

                    H_p = H_0_w[f_idx]

                    if camera_system.bp_accounts_for_d_3():
                        rsd_3 = np.exp(
                            np.complex64(2j * np.pi) * d_3 * frequency)
                        rsd_3 *= invsq_3
                        if optimize_camera_convolutions:
                            H_p = H_p.reshape((nl, nsx, nsy))
                            rsd_3 = rsd_3.reshape((1, rsx, rsy))
                            H_p_fft = np.fft.fft2(
                                H_p, axes=(1, 2), s=(rsx, rsy))
                            rsd_3_fft = np.fft.fft2(
                                rsd_3, axes=(1, 2), s=(rsx, rsy))
                            H_p_fft *= rsd_3_fft
                            H_p = np.fft.ifft2(H_p_fft, axes=(1, 2))
                            H_p = H_p[:, :nvx, :nvy]
                            H_p = H_p.reshape((nl, nva))
                        else:
                            H_p = H_p.reshape((nl, 1, ns))
                            H_p = H_p * rsd_3.reshape((1, nva, ns))
                            H_p = H_p.sum(axis=2)
                        del rsd_3

                    if camera_system.bp_accounts_for_d_2():
                        rsd_2 = np.exp(
                            np.complex64(2j * np.pi) * d_2 * frequency)
                        rsd_2 *= invsq_2
                        if projector_focus_mode == 'exhaustive':
                            assert optimize_projector_convolutions, \
                                'You must use the convolutions optimization when projector_focus=volume_xyz. ' \
                                'Check the documentation for tal.reconstruct.pf_dev for more information.'
                            H_p = H_p.reshape((nlx, nly, nsx, nsy))
                            rsd_2 = rsd_2.reshape((rlx, rly, 1, 1))
                            H_p_fft = np.fft.fft2(
                                H_p, axes=(0, 1), s=(rlx, rly))
                            rsd_2_fft = np.fft.fft2(
                                rsd_2, axes=(0, 1), s=(rlx, rly))
                            H_p_fft *= rsd_2_fft
                            H_p = np.fft.ifft2(H_p_fft, axes=(0, 1))
                            H_p = H_p[:npfx, :npfy, :, :]
                        else:
                            assert projector_focus_mode in ['single',
                                                            'confocal']
                            H_p = H_p.reshape((nl, nva))
                            H_p *= rsd_2.reshape((nl, nva))
                            H_p = H_p.sum(axis=0)
                        H_p = H_p.reshape((n_projector_points, nva))
                        del rsd_2

                    H_1_w[i_w, ...] = H_p * weight

                return H_1_w

            H_1_w = np.zeros((nw, n_projector_points, nva), dtype=np.complex64)
            range_w = np.arange(nw, dtype=np.int32)

            get_resources().split_work(
                work_dividing_frequencies,
                data_in=range_w,
                data_out=H_1_w,
                slice_dims=(0, 0),
            )

        elif propagation_mode == 'divide_volume':
            assert not camera_system.bp_accounts_for_d_2() or \
                projector_focus_mode in ['single', 'confocal']

            def work_dividing_volume(subrange_v):
                nvi = len(subrange_v)
                if camera_system.bp_accounts_for_d_2():
                    d_2_i = d_2[:, subrange_v, :]
                if camera_system.bp_accounts_for_d_3():
                    d_3_i = d_3[:, subrange_v, :]

                H_1_w = np.zeros((nw, n_projector_points, nvi),
                                 dtype=np.complex64)

                fw_iterator = zip(freq_idxs, freqs, weights)
                if progress:
                    fw_iterator = tqdm(fw_iterator,
                                       desc='tal.reconstruct.pf_dev divide-volume',
                                       file=TQDMLogRedirect(),
                                       total=nw,
                                       position=0,
                                       leave=True)

                for f_idx, frequency, weight in fw_iterator:
                    if np.isclose(weight, 0.0):
                        # H_1_w[f_idx, ...] = 0
                        continue

                    H_p = H_0_w[f_idx]

                    if camera_system.bp_accounts_for_d_3():
                        rsd_3 = np.exp(
                            np.complex64(2j * np.pi) * d_3_i * frequency)
                        rsd_3 *= invsq_3
                        H_p = H_p.reshape((nl, 1, ns))
                        H_p = H_p * rsd_3.reshape((1, nvi, ns))
                        H_p = H_p.sum(axis=2)
                        del rsd_3

                    if camera_system.bp_accounts_for_d_2():
                        rsd_2 = np.exp(
                            np.complex64(2j * np.pi) * d_2_i * frequency)
                        rsd_2 *= invsq_2
                        H_p = H_p.reshape((nl, nvi))
                        H_p *= rsd_2.reshape((nl, nvi))
                        H_p = H_p.sum(axis=0)
                        H_p = H_p.reshape((n_projector_points, nvi))
                        del rsd_2

                    H_1_w[f_idx - freq_min_idx, ...] = H_p * weight

                return H_1_w

            H_1_w = np.zeros((nw, n_projector_points, nva), dtype=np.complex64)
            range_v = np.arange(nva, dtype=np.int32)

            get_resources().split_work(
                work_dividing_volume,
                data_in=range_v,
                data_out=H_1_w,
                slice_dims=(0, 2),
            )

        else:
            raise AssertionError('Unknown propagation mode')

        del d_2, d_3
        if not skip_H_fft and i_z == nvz - 1:
            del H_0_w

        def ifft_slice(subrange_v):
            H_1_w_i = H_1_w[:, :, subrange_v]
            H_1_w_i = np.pad(H_1_w_i,
                             ((freq_min_idx, nf - freq_max_idx - 1), (0, 0), (0, 0)), 'constant')

            if camera_system.is_transient():
                return np.fft.ifft(H_1_w_i, axis=0)[padding:-padding, ...]
            else:
                return np.fft.ifft(H_1_w_i, axis=0)[padding, ...]

        range_v = np.arange(nva, dtype=np.int32)
        get_resources().split_work(
            ifft_slice,
            data_in=range_v,
            data_out=H_1[..., i_z],
            slice_dims=(0, 2 if camera_system.is_transient() else 1),
        )

        del H_1_w

    if n_projector_points == 1:
        # squeeze n_projector_points axis
        H_1 = H_1.squeeze(axis=-3)

    return H_1
