import numpy as np
import matplotlib.pyplot as plt
import tal
from tal.enums import HFormat
from tqdm import tqdm


def backproject_pf_single_frequency(H_0, delta_t, d, wavelength, progress=True):
    """
    H_0: (nt, ns)
    delta_t: scalar, in meters
    d: (ns, nv)
    wavelength: scalar, in meters
    """
    assert H_0.ndim == 2, 'Incorrect H format'

    ns, nv = d.shape
    nt, ns_ = H_0.shape
    assert ns == ns_, 'Incorrect shape'

    propagator = np.exp(2j * np.pi * d / wavelength)

    t = delta_t * np.linspace(start=0, stop=nt - 1, num=nt)
    e = np.exp(-2j * np.pi * t / wavelength)
    H_0_w = np.sum(H_0 * e.reshape((nt, 1)), axis=0).reshape((ns, 1))
    # FIXME(diego): implement convolution in frequency domain
    H_1_w = np.sum(H_0_w * propagator, axis=0)

    return H_1_w


def backproject_pf_multi_frequency(
        H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3,
        camera_system, t_accounts_first_and_last_bounces,
        t_start, delta_t,
        wl_mean, wl_sigma, edges,
        laser_xyz=None, sensor_xyz=None, progress=False):
    assert H_0.ndim == 2, 'Incorrect H format'
    assert volume_xyz_n3.ndim == 2 and volume_xyz_n3.shape[1] == 3, \
        'Incorrect volume_xyz format, should be N_3'
    assert laser_grid_xyz.size == 3, 'Only supports one laser position'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'

    nt, ns = H_0.shape
    nv, _ = volume_xyz_n3.shape

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
    nw = len(weights)

    if edges == 'zero':
        H_0 = np.pad(H_0, ((padding, padding), (0, 0)), 'constant')
    else:
        raise AssertionError('Implemented only for edges="zero"')

    """ Propagation of specific frequencies """

    # d_3
    d = np.linalg.norm(
        sensor_grid_xyz.reshape((ns, 1, 3)) -
        volume_xyz_n3.reshape((1, nv, 3)),
        axis=2)
    d -= t_start
    if t_accounts_first_and_last_bounces:
        # d_1
        d += np.linalg.norm(laser_xyz.reshape(3) -
                            laser_grid_xyz.reshape(3))
        # d_4
        d += np.linalg.norm(sensor_xyz.reshape((1, 1, 3)) -
                            sensor_grid_xyz.reshape((ns, 1, 3)), axis=2)
    if camera_system.bp_accounts_for_d_2():
        # d_2
        d += np.linalg.norm(
            laser_grid_xyz.reshape((1, 1, 3)) -
            volume_xyz_n3.reshape((1, nv, 3)),
            axis=2)

    if camera_system.is_transient():
        H_1 = np.zeros((nt, nv), dtype=np.complex64)
    else:
        H_1 = np.zeros(nv, dtype=np.complex64)

    t = delta_t * np.linspace(start=0, stop=nf - 1, num=nf)

    iterator = zip(freqs, weights)
    if progress:
        iterator = tqdm(iterator, total=nw)

    for frequency, weight in iterator:
        wavelength = np.inf if np.isclose(frequency, 0) else 1 / frequency
        H_1_w = weight * \
            backproject_pf_single_frequency(
                H_0, delta_t, d, wavelength, progress=progress)
        e = np.exp(2 * np.pi * 1j * t * frequency) / nf
        H_1_i = H_1_w.reshape((1, nv)) * e.reshape((nf, 1))

        if camera_system.is_transient():
            H_1 += H_1_i[padding: -padding, ...]
        else:
            H_1 += H_1_i[padding]

    return H_1
