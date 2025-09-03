import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats.sampling import DiscreteGuideTable
import time
import h5py


c = 3e8 # Speed of light in m/s

def gaussian(mean: float, std: float, n_timebins: int):
    """

    """
    t_range = np.linspace(0, n_timebins, n_timebins)
    return np.exp(-((t_range - mean) ** 2) / (2 * std ** 2))


def exponenitally_modified_gaussian(mean:float , std:float, decay_rate:float, n_timebins:int):
    """

    """
    t_range = np.linspace(0, n_timebins, n_timebins)
    exgaussian = (decay_rate / 2) * np.exp((decay_rate / 2) * (2 * mean + decay_rate * std * std - 2 * t_range)) \
                  * erfc((mean + decay_rate * std * std - t_range) / (np.sqrt(2) * std))
    return exgaussian


def generate_parametric_jitter(SPAD_FWHM:float, SPAD_tail:float, gaussian_laser_FWHM:float, timebin_width_ps:float, n_timebins:int):
    """

    """
    # Convert parameters in picoseconds to number of timebins
    SPAD_FWHM_scaled = SPAD_FWHM / timebin_width_ps
    SPAD_tail_scaled = SPAD_tail / timebin_width_ps
    SPAD_std = SPAD_FWHM_scaled / (2 * np.sqrt(2 * np.log(2)))
    gaussian_laser_FWHM_scaled = gaussian_laser_FWHM / timebin_width_ps
    laser_std = gaussian_laser_FWHM_scaled / (2 * np.sqrt(2 * np.log(2)))

    # Compute the time jitter caused by the SPAD (exponentially modified gaussian)
    mean = n_timebins * 0.2 # TODO: start value, should be computed automatically
    SPAD_jitter = exponenitally_modified_gaussian(mean, SPAD_std, 1 / SPAD_tail_scaled, n_timebins)

    # Compute time jitter caused by a laser pulse (gaussian)
    mean = n_timebins * 0.5
    laser_jitter = gaussian(mean, laser_std, n_timebins)
    laser_jitter_centered = np.roll(laser_jitter, shift=-int(mean))

    # Complete time jitter (SPAD and laser convolved)
    jitter = np.real(np.fft.ifft(np.fft.fft(SPAD_jitter) * np.fft.fft(laser_jitter_centered)))
    jitter = jitter / np.max(jitter) + 1e-8
    return jitter


def load_jitter_from_file(path:str):
    """

    """
    jitter_file = h5py.File(path, 'r')
    jitter = np.array(jitter_file['counts'])[:, 0]
    jitter_n_timebins = np.array(jitter_file['n_timebins']).item()
    jitter_timebin_width_ps = np.array(jitter_file['timebin_width_ps']).item()
    jitter_file.close()
    return jitter, jitter_n_timebins, jitter_timebin_width_ps


def get_indices_from_linear(index:int, capture_dimensionality:int, shape):
    if capture_dimensionality == 2:
        i = index % shape[0]
        j = index // shape[0]
        return i, j # TODO: TUPLE
    elif capture_dimensionality == 4:
        i = index % shape[0]
        j = (index // shape[0]) % shape[1]
        k = (index // (shape[0] * shape[1])) % shape[2]
        l = (index // (shape[0] * shape[1] * shape[2]))
        return i, j, k, l
    else:
        print('Error, capture is neither single, confocal nor exhaustive')
        exit(1)


def access_transient_data(transient_data, index_tuple, capture_dimensionality):
    if capture_dimensionality == 2:
        return transient_data[:, index_tuple[0], index_tuple[1]]
    elif capture_dimensionality == 4:
        return transient_data[:, index_tuple[0], index_tuple[1], index_tuple[2], index_tuple[3]]
    else:
        print('Error, capture is neither single, confocal nor exhaustive')
        exit(1)


def store_transient_data(transient_data, transient_data_i, index_tuple, capture_dimensionality):
    if capture_dimensionality == 2:
        transient_data[:, index_tuple[0], index_tuple[1]] = transient_data_i
    elif capture_dimensionality == 4:
        transient_data[:, index_tuple[0], index_tuple[1], index_tuple[2], index_tuple[3]] = transient_data_i
    else:
        print('Error, capture is neither single, confocal nor exhaustive')
        exit(1)


# TODO: Given simulation_data (loaded hdf5 file) & configuration, apply noise to the capture
def simulate_noise(capture_data_path:str, config_path:str, args):
    """

    """
    # Load capture data from file
    from tal.io import read_capture
    capture_data = read_capture(capture_data_path)

    start_time = time.time()

    # Load transient data and histogram properties from the captured data
    H = capture_data.H
    timebin_width_opl = capture_data.delta_t
    timebin_width_ps = timebin_width_opl / c * 1e12
    start_opl = capture_data.t_start
    start_ps = start_opl / c * 1e12 # TODO: this is unused. Check if we need it
    n_timebins = H.shape[0]
    n_measurements = H[0].size
    capture_dimensionality = H.ndim - 1
    assert capture_dimensionality == 4 or capture_dimensionality == 2, \
        'Transient data does not match with single, confocal or exhaustive capture data'

    # Load noise simulation configuration from YAML file
    noise_config = None
    assert os.path.exists(config_path), f'{config_path} does not exist'
    assert os.path.isfile(config_path), f'{config_path} is not a TAL config file'
    try:
        noise_config = yaml.safe_load(open(config_path, 'r')) or dict()
    except yaml.YAMLError as exc:
        raise AssertionError(f'Invalid YAML format in noise simulation configuration file: {exc}') from exc

    # Generate or load from file the time jitter function of SPAD and laser
    jitter = None
    jitter_n_timebins = 0
    jitter_timebin_width_ps = 0
    if noise_config['time_jitter_path'] == '' or noise_config['time_jitter_path'] is None:
        # Analytical time jitter function
        print('Generate parametric')
        jitter_n_timebins = noise_config['time_jitter_n_timebins']
        jitter_timebin_width_ps = noise_config['time_jitter_timebin_width']
        jitter = generate_parametric_jitter(noise_config['time_jitter_FWHM'], noise_config['time_jitter_tail'],
                                            noise_config['laser_jitter_FWHM'], jitter_timebin_width_ps, jitter_n_timebins)
    else:
        # Recorded time jitter function from file
        print('Generate from file')
        jitter, jitter_n_timebins, jitter_timebin_width_ps = load_jitter_from_file(noise_config['time_jitter_path'])

    exposure_time = noise_config['exposure_time'] # Exposure time per measurement
    laser_frequency = noise_config['frequency']   # Laser frequency in MHz
    photon_detection_ratio = 0.0
    if noise_config['photon_detection_ratio'] > 0.0:
        photon_detection_ratio = noise_config['photon_detection_ratio']
    else:
        # TODO: compute from the excess voltage of the SPAD. Is this needed?
        photon_detection_ratio = 1.0

    # Expected number of samples per measurement, taking into account the photon detection rate
    n_samples = exposure_time * laser_frequency * 1e6 * photon_detection_ratio
    H_noise = np.zeros(shape=H.shape)
    jitter_sampler = DiscreteGuideTable(jitter, random_state=np.random.RandomState())
    jitter_peak_idx = np.argmax(jitter)

    for i in range(n_measurements):
        index = get_indices_from_linear(i, capture_dimensionality, H[0].shape)
        H_original = access_transient_data(H, index, capture_dimensionality)

        H_sampler = DiscreteGuideTable(H_original, random_state=np.random.RandomState())
        H_sampled = H_sampler.rvs(int(n_samples))
        jitter_sampled = jitter_sampler.rvs(int(n_samples))
        jitter_sampled_scaled = jitter_sampled * jitter_timebin_width_ps / timebin_width_ps # Transform to the timebin width of the transient data

        H_sampled_convolved = H_sampled + jitter_sampled_scaled
        H_histogram = np.histogram(H_sampled_convolved, bins=n_timebins)[0] # TODO: np.histogram generates discontinuities (eg some values go to 0, even if every timebin has actual samples)
        store_transient_data(H_noise, H_histogram, index, capture_dimensionality)

        # TODO: other noise sources. Dark counts and ambient/external noise

        # TODO: afterpulsing
        #       1. Is it needed? (Only if the temporal sequence is very long)
        #       2. How do we do it? Quercus and JSolan did it photon by photon

    print(f'Noise simulation took {time.time() - start_time} seconds (SO FAR)')

    plt.plot(H[:, 16, 17] / np.max(H[:, 16, 17]), label='H')
    plt.plot(H_noise[:, 16, 17] / np.max(H_noise[:, 16, 17]), label='H noisy'); plt.legend(); plt.show()
    # plt.plot(jitter / np.max(jitter), label='jitter')
    # plt.plot(jitter_sampled_histogram / np.max(jitter_sampled_histogram)); plt.show()

    return H_noise
