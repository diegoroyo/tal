import os
import shutil
import yaml
from enum import Enum

from tal.util import local_file_path


class Config(Enum):
    MITSUBA2_TRANSIENT_NLOS_FOLDER = 'MITSUBA2_TRANSIENT_NLOS_FOLDER'


def get_home_path():
    return os.path.expanduser('~')


def get_config_filename():
    return os.path.expanduser('~/.tal.conf')


def _parse_config(lines):
    return dict(line.split('=') for line in lines)


def ask_for_config(param_name: Config, force_ask=True):
    """Most useful: ask for a specific key in the config,
    and if it does not exist ask the user for a value"""
    config_dict = read_config()
    param_name = param_name.value
    if force_ask or param_name not in config_dict or len(config_dict[param_name].strip()) == 0:
        print(f'{param_name} is not specified. Please write a value:')
        param_value = ''
        while len(param_value.strip()) == 0:
            param_value = input(f'{param_name}=')
        config_dict[param_name] = param_value
        write_config(config_dict)
    return config_dict[param_name]


def read_config() -> dict:
    config = get_config_filename()
    if not os.path.isfile(config):
        print('TAL configuration file not found. '
              f'Creating a new one in {config}...')
        try:
            shutil.copy(local_file_path('.tal.conf.example'), config)
        except Exception as exc:
            print(f'/!\\ Unknown error when creating config file: {exc}')

    with open(config, 'r') as f:
        return _parse_config(f.readlines())


def write_config(config_dict):
    config = get_config_filename()
    with open(config, 'w') as f:
        f.writelines(list(f'{key}={value}'
                          for key, value in config_dict.items()))


def read_yaml(filename: str):
    with open(filename, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise AssertionError(
                f'An error ocurred when parsing file ({filename}):\n{exc}')


def write_yaml_string(data_dict: dict) -> str:
    return yaml.dump(data_dict)
