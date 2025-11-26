import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats.sampling import DiscreteGuideTable
import time
from math import floor
import h5py
from tqdm import tqdm

c = 3e8 # Speed of light in m/s

def simulate_noise(capture_data_path:str, config_path:str, args):
    """
        Simulates the noise caused by a transient capture process using a SPAD and a pulsed laser, to a transient
        capture file previously generated using 'tal render'.
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

    print(f'Simulating noise for capture data {capture_data_path}.')

    # Generate or load from file the time jitter function of SPAD and laser
    jitter = None
    jitter_n_timebins = 0
    jitter_timebin_width_ps = 0
    if 'time_jitter_path' in noise_config and noise_config['time_jitter_path'] != '':
        # Recorded time jitter function from file
        jitter, jitter_n_timebins, jitter_timebin_width_ps = load_jitter_from_file(noise_config['time_jitter_path'])
        print(f' - Jitter function loaded from {noise_config["time_jitter_path"]}.')
    else:
        # Analytical time jitter function
        jitter_n_timebins = noise_config['time_jitter_n_timebins']
        jitter_timebin_width_ps = float(noise_config['time_jitter_timebin_width'])
        jitter = generate_parametric_jitter(noise_config['time_jitter_FWHM'], noise_config['time_jitter_tail'],
                                            noise_config['laser_jitter_FWHM'], jitter_timebin_width_ps, jitter_n_timebins)
        print(f' - Jitter function:')
        print(f'   - SPAD FWHM = {noise_config["time_jitter_FWHM"]} ps')
        print(f'   - SPAD tail = {noise_config["time_jitter_tail"]} ps')
        print(f'   - Laser FWHM = {noise_config["laser_jitter_FWHM"]} ps')


    exposure_time = float(noise_config['exposure_time']) # Exposure time per measurement
    laser_frequency = float(noise_config['frequency'])   # Laser frequency in MHz

    # Load the photon detection ratio
    photon_detection_ratio = noise_config['photon_detection_ratio']
    print(f' - Photon detection ratio = {100 * photon_detection_ratio:.2f} %.')

    # Compute the expected number of samples per measurement, taking into account the photon detection rate
    # NOTE: n_samples is the theoretical maximum number of photons detected by the sensor, assuming at most one
    #       photon can be detected per laser pulse (in the case of event based) or per frame

    n_samples = 0
    if 'number_of_samples' in noise_config and noise_config['number_of_samples'] != 0:
        # Load explicit number of detected samples if defined in configuration file
        n_samples = int(noise_config['number_of_samples'])
    elif noise_config['sensor_type'] == 'frame':
        # Frame based SPAD
        frame_exposure_time = float(noise_config['frame_exposure_time']) # Exposure time per frame in microseconds

        n_frames = 0
        if 'n_frames' in noise_config and noise_config['n_frames'] != '':
            # Load number of frames from configuration if defined
            n_frames = int(noise_config['n_frames'])
        else:
            # Compute the number of frames given the total exposure time and the per frame exposure time
            n_frames = floor(exposure_time / (frame_exposure_time * 1e-6))

        # The sensor can only detect at most a single photon per frame
        n_samples = int(n_frames * photon_detection_ratio)

    elif noise_config['sensor_type'] == 'event':
        # Event based SPAD
        n_samples = int(exposure_time * laser_frequency * 1e6 * photon_detection_ratio)
        print(f' - Simulated exposure time = {exposure_time:.3f} seconds.')
        print(f' - Laser frequency = {laser_frequency:.2f} MHz.')
    else:
        raise AssertionError('sensor_type must be one of ("frame", "event")')

    # Expected number of false positive samples (caused by dark counts or external noise)
    n_false_samples = 0
    if 'number_of_false_counts' in noise_config and noise_config['number_of_false_counts'] != 0:
        n_false_samples = int(noise_config['number_of_false_counts'])
    else:
        n_false_samples = int(exposure_time * int(noise_config['dark_count_rate'] + noise_config['external_noise_rate']))
    print(f'{n_samples=}')

    # Afterpulse configuration
    simulate_afterpulses = noise_config['simulate_afterpulses']
    afterpulse_probability = noise_config['afterpulse_probability']
    dead_time_ps = noise_config['dead_time'] # SPAD deadtime after capturing a photon (in picoseconds)
    max_afterpulses = int(sequence_time_ps // dead_time_ps)
    if simulate_afterpulses:
        print(f' - SPAD dead time = {dead_time_ps} ps')
        print(f' - Afterpulse probability = {100 * afterpulse_probability:.2f} %')


    H_noise = np.zeros(shape=H.shape)
    # plt.plot(jitter); plt.title('Jitter function'); plt.show()
    jitter_peak_idx = np.argmax(jitter) # To center the jitter function and avoid offseting the signal
    jitter_sampler = DiscreteGuideTable(jitter, random_state=np.random.RandomState())
    false_count_sampler = DiscreteGuideTable(np.ones(shape=(n_timebins), dtype=float), random_state=np.random.RandomState()) # Uniform Guide Table

    print(f' - Number of photons sampled = {n_samples}')
    print(f' - Number of false positive samples = {n_false_samples}')

    H_maximum = np.max(H, axis=None) # Highest signal intensity

    # For every transient sequence in the capture
    for i in tqdm(range(n_measurements), total=n_measurements, desc=f'Simulating noise ({n_samples} samples per measurement)...'):
        index = get_indices_from_linear(i, capture_dimensionality, H[0].shape)
        H_original = access_transient_data(H, index, capture_dimensionality)
        H_histogram = access_transient_data(H_noise, index, capture_dimensionality) # Array to store the noised transient data

        # If desired, scale n_samples given the ratio of the maximum of the current measurement and the maximum of the highest measurement
        if noise_config['intensity_scaling']:
            n_samples_i = int(n_samples * np.max(H_original, axis=None) / H_maximum)
        else:
            n_samples_i = n_samples

        # Apply jitter and afterpulsing. If the transient signal is empty, only dark count and ambient noise will be added
        if not (H_original == 0.0).all():
            # Sample n_samples photons arrival timestamps from the original transient data, as well as n_samples jitter values
            H_sampler = DiscreteGuideTable(H_original, random_state=np.random.RandomState())
            H_sampled = H_sampler.rvs(n_samples_i)
            jitter_sampled = jitter_sampler.rvs(n_samples_i) - jitter_peak_idx
            jitter_sampled_scaled = jitter_sampled * jitter_timebin_width_ps / timebin_width_ps # Transform to the timebin width of the transient data

            # Sum the sampled timestamps
            H_sampled_convolved = H_sampled + jitter_sampled_scaled
            H_histogram += np.histogram(H_sampled_convolved, bins=n_timebins, range=(0, n_timebins-1))[0]

            # Afterpulse simulation
            H_afterpulses_histogram = None
            if simulate_afterpulses:
                previous_afterpulse_mask = np.ones(n_samples_i, dtype=bool)
                for afterpulse_index in range(max_afterpulses):
                    # Generate a mask for all the measurements that cause an afterpulse
                    afterpulse_samples = np.random.rand(n_samples_i)
                    afterpulse_mask = afterpulse_samples <= afterpulse_probability

                    # Only measurements that caused a previous afterpulse could cause another one
                    afterpulse_mask = afterpulse_mask & previous_afterpulse_mask
                    previous_afterpulse_mask = afterpulse_mask

                    # Accumulate the afterpulsed samples
                    afterpulse_time_offset = dead_time_ps * (afterpulse_index + 1) / timebin_width_ps
                    H_afterpulses = (H_sampled_convolved + afterpulse_time_offset)[afterpulse_mask]
                    H_afterpulses_histogram = np.histogram(H_afterpulses, bins=n_timebins, range=(0.0, n_timebins-1))[0]
                    H_histogram = H_histogram + H_afterpulses_histogram

            # Crosstalk simulation
            if noise_config['is_spad_array'] and noise_config['crosstalk_probability'] > 0.0:
                # Add the crosstalk to the valid neighbors

                # Left neighbor
                if index[0] != 0:
                    H_neighbor = access_transient_data(H_noise, (index[0] - 1, index[1]), capture_dimensionality)
                    H_crosstalk_histogram = compute_crosstalk_histogram(n_samples_i, noise_config['crosstalk_probability'], H_sampled_convolved, n_timebins)
                    H_neighbor += H_crosstalk_histogram
                    store_transient_data(H_noise, H_neighbor, (index[0] - 1, index[1]), capture_dimensionality)

                # Upper neighbor
                if index[1] != 0:
                    H_neighbor = access_transient_data(H_noise, (index[0], index[1] - 1), capture_dimensionality)
                    H_crosstalk_histogram = compute_crosstalk_histogram(n_samples_i, noise_config['crosstalk_probability'], H_sampled_convolved, n_timebins)
                    H_neighbor += H_crosstalk_histogram
                    store_transient_data(H_noise, H_neighbor, (index[0], index[1] - 1), capture_dimensionality)

                # Right neighbor
                if index[0] != H[0].shape[0] - 1:
                    H_neighbor = access_transient_data(H_noise, (index[0] + 1, index[1]), capture_dimensionality)
                    H_crosstalk_histogram = compute_crosstalk_histogram(n_samples_i, noise_config['crosstalk_probability'], H_sampled_convolved, n_timebins)
                    H_neighbor += H_crosstalk_histogram
                    store_transient_data(H_noise, H_neighbor, (index[0] + 1, index[1]), capture_dimensionality)

                # Downward neighbor
                if index[1] != H[0].shape[1] - 1:
                    H_neighbor = access_transient_data(H_noise, (index[0], index[1] + 1), capture_dimensionality)
                    H_crosstalk_histogram = compute_crosstalk_histogram(n_samples_i, noise_config['crosstalk_probability'], H_sampled_convolved, n_timebins)
                    H_neighbor += H_crosstalk_histogram
                    store_transient_data(H_noise, H_neighbor, (index[0], index[1] + 1), capture_dimensionality)

        # Add other noise sources: dark counts and external noise
        # NOTE: Assumes the same number of false positive counts in all measurements
        false_count_sampled = false_count_sampler.rvs(n_false_samples)
        false_count_histogram = np.histogram(false_count_sampled, bins=n_timebins, range=(0, n_timebins-1))[0]
        H_histogram = H_histogram + false_count_histogram

        store_transient_data(H_noise, H_histogram, index, capture_dimensionality)

    print('DONE. Noise simulation took {0:.3f} seconds'.format(time.time() - start_time))

    # Store the transient data with the simulated nosie, as well as the configuration used for the nosie and the jitter function
    capture_data_noisy = capture_data
    capture_data_noisy.H = H_noise
    capture_data_noisy.noise_info = noise_config

    capture_data_noisy.jitter = dict()
    capture_data_noisy.jitter['counts'] = jitter
    capture_data_noisy.jitter['n_timebins'] = jitter_n_timebins
    capture_data_noisy.jitter['timebin_widht_ps'] = jitter_timebin_width_ps
    return capture_data_noisy


def gaussian(mean: float, std: float, n_timebins: int):
    """
        Generates a gaussian distribution given its mean and standard deviation
    """
    t_range = np.linspace(0, n_timebins, n_timebins)
    return np.exp(-((t_range - mean) ** 2) / (2 * std ** 2))


def exponenitally_modified_gaussian(mean:float , std:float, decay_rate:float, n_timebins:int):
    """
        Generates an exponentially modified gaussian distribution, given its mean, standard deviation, and the
        decay rate of the exponential tail
    """
    t_range = np.linspace(0, n_timebins, n_timebins)
    exgaussian = (decay_rate / 2) * np.exp((decay_rate / 2) * (2 * mean + decay_rate * std * std - 2 * t_range)) \
                 * erfc((mean + decay_rate * std * std - t_range) / (np.sqrt(2) * std))
    return exgaussian


def generate_parametric_jitter(SPAD_FWHM:float, SPAD_tail:float, gaussian_laser_FWHM:float, timebin_width_ps:float, n_timebins:int):
    """
        Generates the jitter function of a SPAD and laser given:
         - FWHM of the SPAD
         - Exponential tail of the SPAD
         - FWHM of the laser
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
        Loads a jitter function from an hdf5 file. The file must contain the following datasets:
         - counts: recorded data of the SPAD's jitter
         - n_timebins: number of timebins of the temporal data
         - timebin_width_ps: the temporal width of each timebin, in picoseconds
    """
    jitter_file = h5py.File(path, 'r')
    jitter = np.array(jitter_file['counts'])[:, 0]
    jitter_n_timebins = np.array(jitter_file['n_timebins']).item()
    jitter_timebin_width_ps = np.array(jitter_file['timebin_width_ps']).item()
    jitter_file.close()
    return jitter, jitter_n_timebins, jitter_timebin_width_ps


def get_indices_from_linear(index:int, capture_dimensionality:int, shape):
    """
        Given a linear index, the dimensionality of the transient data and its shape, returns a tuple of indices
        to access the transient data tensor.
    """
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


def compute_crosstalk_histogram(n_samples, crosstalk_probability, H_sampled, n_timebins):
    crosstalk_samples = np.random.rand(n_samples)
    crosstalk_mask = crosstalk_samples <= crosstalk_probability
    H_crosstalk = H_sampled[crosstalk_mask]

    H_crosstalk_histogram = np.histogram(H_crosstalk, bins=n_timebins, range=(0.0, n_timebins-1))[0]
    print(f'{np.count_nonzero(crosstalk_mask)=}, {H_crosstalk.shape=}')
    return H_crosstalk_histogram


def access_transient_data(transient_data, index_tuple, capture_dimensionality):
    """
        Access a single transient measurement from the complete tensor using a tuple of indices.
    """
    if capture_dimensionality == 2:
        return transient_data[:, index_tuple[0], index_tuple[1]]
    elif capture_dimensionality == 4:
        return transient_data[:, index_tuple[0], index_tuple[1], index_tuple[2], index_tuple[3]]
    else:
        print('Error, capture is neither single, confocal nor exhaustive')
        exit(1)


def store_transient_data(transient_data, transient_data_i, index_tuple, capture_dimensionality):
    """
        Store a transient sequence into the complete tensor using a tuple of indices.
    """
    if capture_dimensionality == 2:
        transient_data[:, index_tuple[0], index_tuple[1]] = transient_data_i
    elif capture_dimensionality == 4:
        transient_data[:, index_tuple[0], index_tuple[1], index_tuple[2], index_tuple[3]] = transient_data_i
    else:
        print('Error, capture is neither single, confocal nor exhaustive')
        exit(1)
