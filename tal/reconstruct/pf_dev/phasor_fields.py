import numpy as np
import tal
from tal.config import get_resources, get_memory_usage
from tqdm import tqdm


def backproject_pf_single_frequency_naive(H_0, d, t, frequency):
    """
    H_0: (nt, ns)
    d: (ns, nv)
    t: (nt)
    frequency: scalar, in meters
    """
    assert H_0.ndim == 2, 'Incorrect H format'

    ns, nv = d.shape
    nt, ns_ = H_0.shape
    assert ns == ns_, 'Incorrect shape'

    propagator = np.exp(2j * np.pi * d * frequency)

    e = np.exp(-2j * np.pi * t * frequency)
    H_0_w = np.sum(H_0 * e.reshape((nt, 1)), axis=0).reshape((ns, 1))
    H_1_w = np.sum(H_0_w * propagator, axis=0)

    return H_1_w


def backproject_pf_single_frequency_conv(H_0, d_014, d_2, d_3, t, frequency):
    """
    Special case where the volume is a slice parallel to the relay wall
    with the same resolution and XY positions (only Z varies)

    The convolution can be implemented in frequency domain and can be much faster

    H_0: (nt, nsx, nsy)
    d_014: scalar, in meters
    d_2: (nsx, nsy)
    d_3: (rx, ry) where rx = 2 * nsx - 1 and ry = 2 * nsy - 1
    t: (nt)
    frequency: scalar, in meters
    """
    assert H_0.ndim == 3, 'Incorrect H format'

    rx, ry = d_3.shape
    nt, nsx, nsy = H_0.shape
    assert rx == 2 * nsx - 1 and ry == 2 * nsy - 1, 'Incorrect shape'

    rsd_014 = np.exp(2j * np.pi * d_014 * frequency)
    rsd_3 = np.exp(2j * np.pi * d_3 * frequency)

    e = np.exp(-2j * np.pi * t * frequency)
    H_0_w = np.sum(H_0 * e.reshape((nt, 1, 1)), axis=0).reshape((nsx, nsy))

    H_0_w *= rsd_014

    H_0_w_fft = np.fft.fft2(H_0_w, s=(rx, ry))
    rsd_3_fft = np.fft.fft2(np.fft.ifftshift(rsd_3), s=(rx, ry))
    H_1_w_full_fft = H_0_w_fft * rsd_3_fft
    H_1_w_full = np.fft.ifft2(H_1_w_full_fft)
    H_1_w = H_1_w_full[:nsx, :nsy]

    if d_2 is not None:
        rsd_2 = np.exp(2j * np.pi * d_2 * frequency)
        H_1_w *= rsd_2

    return H_1_w


def backproject_pf_multi_frequency(
        H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz,
        camera_system, t_accounts_first_and_last_bounces,
        t_start, delta_t,
        wl_mean, wl_sigma, border,
        laser_xyz=None, sensor_xyz=None, progress=False):
    assert laser_grid_xyz.size == 3, 'Only supports one laser position'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'

    nt = H_0.shape[0]
    nv = np.prod(volume_xyz.shape[:-1])  # N or X * Y or X * Y * Z
    ns = np.prod(sensor_grid_xyz.shape[:-1])  # N or X * Y

    try:
        assert H_0.ndim == 3
        assert sensor_grid_xyz.ndim == 3
        assert volume_xyz.ndim == 3 or volume_xyz.ndim == 4
        assert not t_accounts_first_and_last_bounces

        nt, nsx, nsy = H_0.shape
        nsx_, nsy_ = sensor_grid_xyz.shape[:2]
        assert nsx == nsx_ and nsy == nsy_
        nvx, nvy = volume_xyz.shape[:2]
        assert nvx == nsx and nvy == nsy

        def check_parallel(xyz_a, xyz_b):
            return np.allclose(xyz_a[..., 0:2], xyz_b[..., 0:2])

        if volume_xyz.ndim == 3:
            assert check_parallel(sensor_grid_xyz, volume_xyz)
        elif volume_xyz.ndim == 4:
            nvx, nvy, nvz = volume_xyz.shape[:3]
            for iz in range(nvz):
                depths = volume_xyz[:, :, iz, 2].reshape(-1)
                assert np.all(depths == depths[0])
                assert check_parallel(sensor_grid_xyz, volume_xyz[:, :, iz])

        print('tal.reconstruct.pf_dev: Using convolutions optimization')
        optimize_slices = True
    except AssertionError:
        assert np.prod(H_0.shape[1:]) == ns, 'Incorrect shape'
        H_0 = H_0.reshape((nt, ns))
        sensor_grid_xyz = sensor_grid_xyz.reshape((ns, 3))
        volume_xyz = volume_xyz.reshape((nv, 3))
        optimize_slices = False

    """ Phasor fields filter """

    # FIXME(diego): obtain filter parameters directly in frequency
    # domain (avoid conversion from time domain)

    t_6sigma = int(np.ceil(6 * wl_sigma / delta_t))
    padding = 2 * t_6sigma
    # FIXME(diego) if we want to convert a circular convolution to linear,
    # this should be nt + t_6sigma - 1 instead of nt + 4 * t_6sigma or even nt + 2 * t_6sigma
    # I have found cases where it fails even with nt + 2 * t_6sigma (Z, 0th pixel)
    nf = nt + 2 * padding

    t_max = delta_t * (nf - 1)
    t = np.linspace(start=0, stop=t_max, num=nf)

    gaussian_envelope = np.exp(-((t - t_max / 2) / wl_sigma) ** 2 / 2)
    K = gaussian_envelope / np.sum(gaussian_envelope) * \
        np.exp(2j * np.pi * t / wl_mean)
    K = np.fft.fftshift(K)  # center at zero

    mean_idx = (nf * delta_t) / wl_mean
    sigma_idx = (nf * delta_t) / (wl_sigma * 6)
    # shift to center at zero, easier for low negative frequencies
    freq_min_idx = nf // 2 + int(np.floor(mean_idx - 3 * sigma_idx))
    freq_max_idx = nf // 2 + int(np.ceil(mean_idx + 3 * sigma_idx))
    K_fftshift = np.fft.fftshift(np.fft.fft(K))
    K_fftfreq = np.fft.fftshift(np.fft.fftfreq(nf, d=delta_t))

    weights = K_fftshift[freq_min_idx:freq_max_idx+1]
    freqs = K_fftfreq[freq_min_idx:freq_max_idx+1]
    print('tal.reconstruct.pf_dev: '
          f'Using wavelengths from {1 / freqs[-1]:.4f}m to {1 / freqs[0]:.4f}m')
    nw = len(weights)

    if border == 'zero':
        # only pad temporal dimension
        H_0 = np.pad(H_0,
                     ((padding, padding),) + ((0, 0),) * (H_0.ndim - 1), 'constant')
    else:
        raise AssertionError('Implemented only for border="zero"')

    """ Propagation of specific frequencies """

    if camera_system.is_transient():
        H_1 = np.zeros((nt, nv), dtype=np.complex64)
    else:
        H_1 = np.zeros(nv, dtype=np.complex64)

    d_014 = t_start * -1
    if t_accounts_first_and_last_bounces:
        # d_1
        d_014 += np.linalg.norm(laser_xyz.reshape(3) -
                                laser_grid_xyz.reshape(3))
        # d_4
        d_014 += np.linalg.norm(sensor_xyz.reshape((1, 1, 3)) -
                                sensor_grid_xyz.reshape((ns, 1, 3)), axis=2)

    t = delta_t * np.linspace(start=0, stop=nf - 1, num=nf)

    h0 = H_0.dtype.itemsize
    h1 = H_1.dtype.itemsize
    s = sensor_grid_xyz.dtype.itemsize

    if optimize_slices:
        # optimized for when the volume is formed
        # with XY slices parallel to the relay wall
        if volume_xyz.ndim == 3:
            nvx, nvy, _ = volume_xyz.shape
            volume_xyz = volume_xyz.reshape((nvx, nvy, 1, 3))
        nvx, nvy, nvz, _ = volume_xyz.shape

        old_H_1_shape = H_1.shape
        if camera_system.is_transient():
            H_1 = H_1.reshape((nt, nvx, nvy, nvz))
        else:
            H_1 = H_1.reshape((nvx, nvy, nvz))

        range_z = np.arange(nvz, dtype=np.int32)

        dx = volume_xyz[1, 0, 0, 0] - volume_xyz[0, 0, 0, 0]
        dy = volume_xyz[0, 1, 0, 1] - volume_xyz[0, 0, 0, 1]
        rx = 2 * nvx - 1
        ry = 2 * nvy - 1
        px = np.linspace(start=-dx * (nvx - 1), stop=dx *
                         (nvx - 1), num=rx, dtype=np.float32)
        px = np.squeeze(np.stack((px,)*ry, axis=1))
        py = np.linspace(start=-dy * (nvy - 1), stop=dy *
                         (nvy - 1), num=ry, dtype=np.float32)
        py = np.squeeze(np.stack((py,)*rx, axis=0))

        def work_conv(subrange_z):
            volume_xyz_d = volume_xyz[..., subrange_z, :]
            nvx, nvy, nvzd, _ = volume_xyz_d.shape

            if camera_system.is_transient():
                H_1 = np.zeros((nt, nvx, nvy, nvzd), dtype=np.complex64)
            else:
                H_1 = np.zeros((nvx, nvy, nvzd), dtype=np.complex64)

            z_list = volume_xyz_d[0, 0, :, 2]
            if progress:
                progress_bar = tqdm(leave=False,
                                    total=len(z_list) * len(freqs))

            for iz, pz in enumerate(z_list):
                progress_bar.set_description(f'Z = {pz:.2f}m')
                d_2 = None
                if camera_system.bp_accounts_for_d_2():
                    d_2 = np.linalg.norm(
                        volume_xyz_d[:, :, iz, :] -
                        laser_grid_xyz.reshape((1, 1, 3)),
                        axis=2)
                d_3 = np.linalg.norm(
                    np.stack(
                        (px, py, np.ones((rx, ry), dtype=np.float32) * pz),
                        axis=2),
                    axis=2)

                fw_iterator = zip(freqs, weights)

                for frequency, weight in fw_iterator:
                    H_1_w = weight * \
                        backproject_pf_single_frequency_conv(
                            H_0, d_014, d_2, d_3, t, frequency)
                    e = np.exp(2j * np.pi * t * frequency) / nf
                    H_1_i = H_1_w.reshape((1, nvx, nvy)) * \
                        e.reshape((nf, 1, 1))

                    if camera_system.is_transient():
                        H_1[..., iz] += H_1_i[padding:-padding, ...]
                    else:
                        H_1[..., iz] += H_1_i[padding, ...]

                    if progress:
                        progress_bar.update(1)

            return H_1

        get_resources().split_work(
            work_conv,
            data_in=range_z,
            data_out=H_1,
            f_mem_usage=lambda dc: (
                lambda downscale, cpus:
                get_memory_usage(
                    ((ns, 3), s * (1 + cpus)), ((4 * ns, 3), s * (1 + cpus)),
                    (H_0.shape, (1 + h0) * cpus), (H_1.shape, 2 * h1))
            )(*dc),
            slice_dims=(0, H_1.ndim - 1),
        )

        H_1 = H_1.reshape(old_H_1_shape)
    else:
        # arbitrary volume
        range_v = np.arange(nv, dtype=np.int32)

        def work_n3(subrange_v):
            nvd = len(subrange_v)
            volume_xyz_d = volume_xyz[subrange_v]

            if camera_system.is_transient():
                H_1 = np.zeros((nt, nvd), dtype=np.complex64)
            else:
                H_1 = np.zeros(nvd, dtype=np.complex64)

            # d_3
            d = np.linalg.norm(
                sensor_grid_xyz.reshape((ns, 1, 3)) -
                volume_xyz_d.reshape((1, nvd, 3)),
                axis=2)
            if camera_system.bp_accounts_for_d_2():
                # d_2
                d += np.linalg.norm(
                    laser_grid_xyz.reshape((1, 1, 3)) -
                    volume_xyz_d.reshape((1, nvd, 3)),
                    axis=2)
            d += d_014  # add t_start, d_1 and d_4 terms

            iterator = zip(freqs, weights)
            if progress:
                iterator = tqdm(iterator, total=nw, leave=False)

            for frequency, weight in iterator:
                H_1_w = weight * \
                    backproject_pf_single_frequency_naive(
                        H_0, d, t, frequency)
                e = np.exp(2j * np.pi * t * frequency) / nf
                H_1_i = H_1_w.reshape((1, nvd)) * e.reshape((nf, 1))

                if camera_system.is_transient():
                    H_1 += H_1_i[padding: -padding, ...]
                else:
                    H_1 += H_1_i[padding]

            return H_1

        get_resources().split_work(
            work_n3,
            data_in=range_v,
            data_out=H_1,
            f_mem_usage=lambda dc: (
                lambda downscale, cpus:
                get_memory_usage(
                    ((ns, nv, 3), s * (1 + cpus / downscale)), (H_0.shape, (1 + h0) * cpus), (H_1.shape, 2 * h1))
            )(*dc),
            slice_dims=(0, H_1.ndim - 1),
        )

    return H_1
