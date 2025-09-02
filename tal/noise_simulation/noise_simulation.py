import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats.sampling import DiscreteGuideTable
import time


c = 3e8 # Speed of light in m/s

def gaussian(mean, std, n_timebins):
    """

    """
    t_range = np.linspace(0, n_timebins, n_timebins)
    return np.exp(-((t_range - mean) ** 2) / (2 * std ** 2))


def exponenitally_modified_gaussian(mean, std, decay_rate, n_timebins):
    """

    """
    t_range = np.linspace(0, n_timebins, n_timebins)
    exgaussian = (decay_rate / 2) * np.exp((decay_rate / 2) * (2 * mean + decay_rate * std * std - 2 * t_range)) \
                  * erfc((mean + decay_rate * std * std - t_range) / (np.sqrt(2) * std))
    return exgaussian


def generate_parametric_jitter(SPAD_FWHM, SPAD_tail, gaussian_laser_FWHM, timebin_width_ps, n_timebins):
    """

    """
    SPAD_FWHM_scaled = SPAD_FWHM / timebin_width_ps
    SPAD_tail_scaled = SPAD_tail / timebin_width_ps
    SPAD_std = SPAD_FWHM_scaled / (2 * np.sqrt(2 * np.log(2)))

    gaussian_laser_FWHM_scaled = gaussian_laser_FWHM / timebin_width_ps
    laser_std = gaussian_laser_FWHM_scaled / (2 * np.sqrt(2 * np.log(2)))

    # Time jitter caused by the SPAD (exponentially modified gaussian)
    mean = n_timebins * 0.2 # TODO: start value, should be computed automatically
    SPAD_jitter = exponenitally_modified_gaussian(mean, SPAD_std, 1 / SPAD_tail_scaled, n_timebins)

    # Time jitter caused by a laser pulse (gaussian)
    mean = n_timebins * 0.5
    laser_jitter = gaussian(mean, laser_std, n_timebins)
    laser_jitter_centered = np.roll(laser_jitter, shift=-int(mean))

    # Complete time jitter (SPAD and laser convolved)
    # TODO: not using rfft causes an error when creating the DiscreteGuideTable, but using it creates a baseline?
    #       Either fix the bug with DiscreteGuideTable or remove baseline
    # FIXED: by adding a small baseline (1e-8), CHECK if there is any other way to avoid this
    jitter = np.real(np.fft.ifft(np.fft.fft(SPAD_jitter) * np.fft.fft(laser_jitter_centered)))
    jitter = jitter / np.max(jitter) + 1e-8
    return jitter


def load_jitter_from_file(path):
    """

    """
    # TODO: not implemented yet
    return None


# TODO: Given simulation_data (loaded hdf5 file) & configuration, apply noise to the capture
def simulate_noise(capture_data, config_path, args):
    """

    """
    # Load capture data from file
    from tal.io import read_capture
    capture_data_path = args.capture_file
    capture_data = read_capture(capture_data_path)

    start_time = time.time()

    # Load transient data and histogram properties from the captured data
    H = capture_data.H
    timebin_width_opl = capture_data.delta_t
    timebin_width_ps = timebin_width_opl / c * 1e12
    start_opl = capture_data.t_start
    start_ps = start_opl / c * 1e12
    n_timebins = H.shape[0]
    capture_dimensionality = H.ndim - 1

    print(f'{H.shape=}')

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
    if noise_config['time_jitter_path'] == '' or noise_config['time_jitter_path'] is None:
        # Analytical time jitter function
        print('Generate parametric')
        jitter = generate_parametric_jitter(noise_config['time_jitter_FWHM'], noise_config['time_jitter_tail'],
                                            noise_config['laser_jitter_FWHM'], timebin_width_ps, n_timebins)
    else:
        # Recorded time jitter function from file
        print('Generate from file')
        jitter = np.load(noise_config['time_jitter_path'])

    # plt.plot(H[:, 16, 16]); plt.title('H'); plt.show()
    # plt.plot(jitter); plt.title('jitter'); plt.show()

    exposure_time = noise_config['exposure_time'] # Exposure time per measurement
    laser_frequency = noise_config['frequency']   # Laser frequency in MHz
    photon_detection_ratio = 0.0
    if noise_config['photon_detection_ratio'] > 0.0:
        photon_detection_ratio = noise_config['photon_detection_ratio']
    else:
        # TODO: compute from the excess voltage of the SPAD
        photon_detection_ratio = 1.0

    # Expected number of samples per measurement, taking into account the photon detection rate
    n_samples = exposure_time * laser_frequency * 1e6 * photon_detection_ratio
    H_noise = np.zeros(shape=H.shape)
    jitter_sampler = DiscreteGuideTable(jitter, random_state=np.random.RandomState())

    # TODO: loop implementation needed if we want to use the DiscreteGuideTable (only one dimensional arrays)
    #       With this resolution (32 x 32) the complete loop (create one sampler, sample twice and create histogram)
    #       takes around 0.6 seconds
    """
    i = 0
    for i in range(32):
        for j in range(32):
            H_sampler = DiscreteGuideTable(H[:, i, j], random_state=np.random.RandomState())
            H_sampled = H_sampler.rvs(int(n_samples))
            jitter_sampled = jitter_sampler.rvs(int(n_samples))
            H_histogram = np.histogram(H_sampled + jitter_sampled, bins=int(n_timebins))
    """
    H_sampler = DiscreteGuideTable(H[:, 16, 16], random_state=np.random.RandomState())

    # Simulate jitter noise for a single measurement for now
    H_sampled = H_sampler.rvs(int(n_samples))
    jitter_sampled = jitter_sampler.rvs(int(n_samples)) - (n_timebins * 0.2) # The center or start of the time_jitter should be either a parameter, or computed beforehand
    H_sampled_convolved = H_sampled + jitter_sampled
    H_noise[:, 16, 16] = np.histogram(H_sampled_convolved, bins=n_timebins)[0] # TODO: np.histogram generates discontinuities (eg some values go to 0, even if every timebin has actual samples)
    print(f'Noise simulation took {time.time() - start_time} seconds (SO FAR)')

    plt.plot(jitter); plt.title('System Jitter'); plt.show()
    plt.plot(H[:, 16, 16] / np.max(H[:, 16, 16]) * 250, label='H')
    # plt.plot(H_noise[:, 16, 16] / np.max(H_noise[:, 16, 16]) * 250, label='H noisy')
    plt.hist(H_sampled, bins=n_timebins); plt.show()
    plt.plot(jitter / np.max(jitter) * 250, label='jitter')
    plt.hist(jitter_sampled, bins=n_timebins); plt.show()

    return H_noise

