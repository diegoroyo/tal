import dill
import os
import shutil
import yaml
import multiprocessing as mp
import numpy as np
from enum import Enum

from tal.util import local_file_path

""" Utils to read/write to persistent config file """


class Config(Enum):
    MITSUBA_VERSION = \
        ('MITSUBA_VERSION',
         'Version of Mitsuba to use (2 or 3)',
         '2',
         lambda s: s in ['2', '3'])
    MITSUBA2_TRANSIENT_NLOS_FOLDER = \
        ('MITSUBA2_TRANSIENT_NLOS_FOLDER',
         'Location of mitsuba2-transient-nlos installation folder',
         '',
         lambda s: os.path.isdir(s))
    MITSUBA3_TRANSIENT_NLOS_FOLDER = \
        ('MITSUBA3_TRANSIENT_NLOS_FOLDER',
         'Location of mitsuba3-transient-nlos installation folder',
         '',
         lambda s: os.path.isdir(s))


def get_home_path():
    return os.path.expanduser('~')


def get_config_filename():
    return os.path.expanduser('~/.tal.conf')


def _parse_config(lines):
    return dict(line.strip().split('=') for line in lines)


def ask_for_config(param_name: Config, force_ask=True):
    """Most useful: ask for a specific key in the config,
    and if it does not exist ask the user for a value"""
    config_dict = read_config()
    param_name, ask_query, default_value, is_valid = param_name.value
    param_ok = (
        param_name in config_dict and
        len(config_dict[param_name].strip()) > 0 and
        is_valid(config_dict[param_name])
    )
    default_value = (
        config_dict[param_name]
        if param_ok and len(default_value) == 0
        else default_value
    )
    if force_ask or not param_ok:
        print(f'{param_name} is not specified. Please specify:')
        param_value = ''
        while len(param_value.strip()) == 0 or not is_valid(param_value):
            param_value = input(
                f'{ask_query}{"" if len(default_value) == 0 else f" [default: {default_value}]"}: ')
            param_value = param_value.strip()
            if len(param_value) == 0:
                param_value = default_value
            if not is_valid(param_value):
                print('Invalid value.')
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
        f.writelines(list(f'{key}={value}\n'
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


""" Utils to read/write to the context configuration (tal.config) """


def get_memory_usage(*args):
    return sum(np.prod(shape) * item_size for shape, item_size in args) / 2 ** 30


def _run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def _apply_async(pool, fun, args, **kwargs):
    payload = dill.dumps((fun, args))
    return pool.apply_async(_run_dill_encoded, (payload,), **kwargs)


DEFAULT_CPU_PROCESSES = 1
DEFAULT_MEMORY_LIMIT_GB = None


class ResourcesConfig:
    """ See tal.set_resources """

    def __init__(self,
                 cpu_processes=DEFAULT_CPU_PROCESSES,
                 max_memory_gb=DEFAULT_MEMORY_LIMIT_GB):
        self.cpu_processes = cpu_processes
        self.max_memory_gb = max_memory_gb
        self.force_single_thread = cpu_processes == 1

    def __enter__(self):
        global TAL_RESOURCES_CONFIG
        self.old_resources = TAL_RESOURCES_CONFIG
        TAL_RESOURCES_CONFIG = self

    def __exit__(self, _, __, ___):
        global TAL_RESOURCES_CONFIG
        TAL_RESOURCES_CONFIG = self.old_resources
        del self.old_resources
        return False

    def split_work(self, f_work, data_in, data_out, f_mem_usage, slice_dims):

        def single_process():
            data_out[:] = f_work(data_in)

        if self.force_single_thread:
            single_process()
            return

        max_cpu = min(
            mp.cpu_count(),
            999 if self.cpu_processes == 'max' else self.cpu_processes
        )

        in_slice_dim, out_slice_dim = slice_dims

        def check_divisible(x):
            if in_slice_dim is not None and data_in.shape[in_slice_dim] % x != 0:
                return False
            if out_slice_dim is not None and data_out.shape[out_slice_dim] % x != 0:
                return False
            return True

        cpus = 1
        downscale = 1
        done = False
        DOWNSCALE_LIMIT = 128
        if self.max_memory_gb is None:
            data_dim = min((x for x in [
                           None if in_slice_dim is None else data_in.shape[in_slice_dim],
                           None if out_slice_dim is None else data_out.shape[out_slice_dim],
                           ] if x is not None), default=None)
            if data_dim is not None and data_dim < max_cpu:
                while check_divisible(downscale * 2) and downscale * 2 <= DOWNSCALE_LIMIT:
                    downscale *= 2
                cpus = downscale
            else:
                cpus = max_cpu
                downscale = int(2 ** np.ceil(np.log2(max_cpu)))
        else:
            while not done and downscale <= DOWNSCALE_LIMIT:
                if f_mem_usage((downscale, cpus)) > self.max_memory_gb and check_divisible(downscale * 2):
                    # computations do not fit in memory
                    downscale *= 2
                elif cpus < max_cpu and cpus < downscale and f_mem_usage((downscale, cpus + 1)) < self.max_memory_gb:
                    # we can use more cpus
                    cpus += 1
                elif cpus < max_cpu and f_mem_usage((downscale * 2, cpus + 1)) < self.max_memory_gb and check_divisible(downscale * 2):
                    # we can use more cpus only if we downscale more
                    downscale *= 2
                    cpus += 1
                else:
                    # found optimal configuration
                    done = True

        if downscale > DOWNSCALE_LIMIT:
            print(f'WARNING: Memory usage is too big even with the lowest configuration '
                  f'({f_mem_usage((downscale, cpus))} GiB used of {self.max_memory_gb} GiB available '
                  f'by splitting computations in {downscale} chunks).')

        if downscale == 1:
            single_process()
            return

        print(f'tal.resources: Using {cpus} processes out of {max_cpu} '
              f'and downscale {downscale}')

        if in_slice_dim is not None:
            in_shape = np.insert(data_in.shape, in_slice_dim + 1, downscale)
            in_shape[in_slice_dim] //= downscale
        if out_slice_dim is not None:
            out_shape = np.insert(data_out.shape, out_slice_dim + 1, downscale)
            out_shape[out_slice_dim] //= downscale

        with mp.Pool(processes=max_cpu) as pool:
            try:
                def do_work(din):
                    return _apply_async(pool, f_work, (din,),
                                        error_callback=lambda exc: print(f'/!\ Process found an exception: {exc}'))

                if in_slice_dim is not None:
                    data_in = np.moveaxis(
                        data_in.reshape(in_shape), in_slice_dim + 1, 0)
                if out_slice_dim is not None:
                    data_out = np.moveaxis(
                        data_out.reshape(out_shape), out_slice_dim + 1, 0)

                asyncs = []
                for in_slice in data_in:
                    asyncs.append(do_work(in_slice))

                pool.close()
                if out_slice_dim is None:
                    for i, async_ in enumerate(asyncs):
                        data_out += async_.get()
                else:
                    for i, (async_, out_slice) in enumerate(zip(asyncs, data_out)):
                        out_slice[:] = async_.get()
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise KeyboardInterrupt


TAL_RESOURCES_CONFIG = ResourcesConfig()


def get_resources():
    return TAL_RESOURCES_CONFIG


def set_resources(cpu_processes=DEFAULT_CPU_PROCESSES, memory_limit_gb=DEFAULT_MEMORY_LIMIT_GB):
    """
    Configure Y-TAL to use a specific number of CPU processes and a memory limit.

    Not all functions implemented in Y-TAL support this configuration, mostly the ones
    in tal.reconstruct (filtering and reconstruction).

    Default configuration is 1 CPU process and no memory limit.

    cpu_processes
        Can be an integer or 'max' to use all available CPU processes.

    memory_limit_gb
        Can be an integer or None to use no memory limit.
    """
    global TAL_RESOURCES_CONFIG
    TAL_RESOURCES_CONFIG = ResourcesConfig(cpu_processes, memory_limit_gb)
