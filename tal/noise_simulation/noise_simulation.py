import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats.sampling import DiscreteGuideTable
import time
import h5py
from tqdm import tqdm


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
    mean = n_timebins * 0.3
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
        return i, j
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
    start_ps = start_opl / c * 1e12
    n_timebins = H.shape[0]
    sequence_time_ps = n_timebins * timebin_width_ps # Total time of the temporal sequence in picoseconds
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
        jitter_n_timebins = noise_config['time_jitter_n_timebins']
        jitter_timebin_width_ps = noise_config['time_jitter_timebin_width']
        jitter = generate_parametric_jitter(noise_config['time_jitter_FWHM'], noise_config['time_jitter_tail'],
                                            noise_config['laser_jitter_FWHM'], jitter_timebin_width_ps, jitter_n_timebins)
    else:
        # Recorded time jitter function from file
        jitter, jitter_n_timebins, jitter_timebin_width_ps = load_jitter_from_file(noise_config['time_jitter_path'])

    exposure_time = noise_config['exposure_time']               # Exposure time per measurement
    laser_frequency = noise_config['frequency']                 # Laser frequency in MHz
    dead_time_ps = noise_config['dead_time'] # SPAD deadtime after capturing a photon (in picoseconds)
    max_afterpulses = int(sequence_time_ps // dead_time_ps)

    photon_detection_ratio = 0.0
    if noise_config['photon_detection_ratio'] > 0.0:
        photon_detection_ratio = noise_config['photon_detection_ratio']
    else:
        # TODO: compute from the excess voltage of the SPAD. Is this needed?
        photon_detection_ratio = 1.0

    # Expected number of samples per measurement, taking into account the photon detection rate
    n_samples = 0
    if noise_config['number_of_samples'] == 0 or noise_config['number_of_samples'] == None:
        n_samples = int(exposure_time * laser_frequency * 1e6 * photon_detection_ratio)
    else:
        n_samples = int(noise_config['number_of_samples'])

    # Expected number of false positive samples (caused by dark counts or external noise)
    n_false_samples = 0
    if noise_config['number_of_false_counts'] == 0 or noise_config['number_of_false_counts'] == None:
        n_false_samples = int(exposure_time * int(noise_config['dark_count_rate'] + noise_config['external_noise_rate']))
    else:
        n_false_samples = int(noise_config['number_of_false_counts'])

    H_noise = np.zeros(shape=H.shape)
    jitter_sampler = DiscreteGuideTable(jitter, random_state=np.random.RandomState())
    jitter_peak_idx = np.argmax(jitter)
    false_count_sampler = DiscreteGuideTable(np.ones(shape=(n_timebins), dtype=float), random_state=np.random.RandomState())

    for i in tqdm(range(n_measurements), total=n_measurements, desc=f'Simulating noise ({n_samples} samples per measurement)...'):
        index = get_indices_from_linear(i, capture_dimensionality, H[0].shape)
        H_original = access_transient_data(H, index, capture_dimensionality)

        H_sampler = DiscreteGuideTable(H_original, random_state=np.random.RandomState())
        H_sampled = H_sampler.rvs(n_samples)
        jitter_sampled = jitter_sampler.rvs(n_samples) - jitter_peak_idx
        jitter_sampled_scaled = jitter_sampled * jitter_timebin_width_ps / timebin_width_ps # Transform to the timebin width of the transient data

        H_sampled_convolved = H_sampled + jitter_sampled_scaled
        H_histogram = np.histogram(H_sampled_convolved, bins=n_timebins, range=(0, n_timebins-1))[0]

        # After pulse simulation
        H_afterpulses_histogram = None
        if noise_config['simulate_afterpulses']:
            previous_afterpulse_mask = np.ones(n_samples, dtype=bool)
            for afterpulse_index in range(max_afterpulses):
                # Generate a mask for all the measurements that cause an afterpulse
                afterpulse_samples = np.random.rand(n_samples)
                afterpulse_mask = afterpulse_samples <= noise_config['afterpulse_probability']

                # Only measurements that cause a previous afterpulse could cause another one
                afterpulse_mask = afterpulse_mask & previous_afterpulse_mask
                previous_afterpulse_mask = afterpulse_mask

                # Accumulate the afterpulsed samples
                afterpulse_time_offset = dead_time_ps * (afterpulse_index + 1) / timebin_width_ps
                H_afterpulses = (H_sampled_convolved + afterpulse_time_offset)[afterpulse_mask]
                H_afterpulses_histogram = np.histogram(H_afterpulses, bins=n_timebins, range=(0.0, n_timebins-1))[0]
                H_histogram = H_histogram + H_afterpulses_histogram

        # Add other noise sources: dark counts and external noise
        # NOTE: Assumes the same number of false positive counts in all measurements
        false_count_sampled = false_count_sampler.rvs(n_false_samples)
        false_count_histogram = np.histogram(false_count_sampled, bins=n_timebins, range=(0, n_timebins-1))[0]
        H_histogram = H_histogram + false_count_histogram

        store_transient_data(H_noise, H_histogram, index, capture_dimensionality)

    print('DONE. Noise simulation took {0:.3f} seconds'.format(time.time() - start_time))
    plt.plot(H[:, 16, 16] / np.max(H[16, 16]), label='H')
    plt.plot(H_noise[:, 16, 16] / np.max(H_noise[:, 16, 16]), label='H noisy'); plt.legend()
    plt.show()

    capture_data_noisy = capture_data
    capture_data_noisy.H = H_noise
    capture_data_noisy.noise_info = noise_config
    return capture_data_noisy
