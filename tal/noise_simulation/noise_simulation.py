import numpy as np
import yaml
import os


# TODO: Given simulation_data (loaded hdf5 file) & configuration, apply noise to the capture
def simulate_noise(capture_data, config_path, args):
    # Load transient data and histogram properties from the captured data
    H = capture_data.H
    timebin_width_opl = capture_data.delta_t
    start_opl = capture_data.t_start
    n_timebins = capture_data.scene_info['original_nt']

    # Load noise simulation configuration from YAML file
    noise_config = None
    assert os.path.exists(config_path), f'{config_path} does not exist'
    assert os.path.isfile(config_path), f'{config_path} is not a TAL config file'
    try:
        noise_config = yaml.safe_load(open(config_path, 'r')) or dict()
    except yaml.YAMLError as exc:
        raise AssertionError(f'Invalid YAML format in noise simulation configuration file: {exc}') from exc

    print(f'{H.shape=}')
    print(f'{noise_config=}')
    H_noise = np.zeros(shape=H.shape)
    return H_noise
