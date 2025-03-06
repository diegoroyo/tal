import os
import shutil
import yaml
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from tal.util import local_file_path
from tal.log import log, LogLevel, TQDMLogRedirect

""" Utils to read/write to persistent config file """


class Config(Enum):
    MITSUBA_VERSION = \
        ('MITSUBA_VERSION',
         'Version of Mitsuba to use (2 or 3)',
         '3',
         lambda s: s in ['2', '3'])
    MITSUBA2_TRANSIENT_NLOS_FOLDER = \
        ('MITSUBA2_TRANSIENT_NLOS_FOLDER',
         'Location of mitsuba2-transient-nlos installation folder',
         '',
         lambda s: os.path.isdir(s))
    MITSUBA3_TRANSIENT_NLOS_FOLDER = \
        ('MITSUBA3_TRANSIENT_NLOS_FOLDER',
         'Path to your compiled Mitsuba 3 (not mitransient). If you installed mitsuba from pip, write \'pip\'',
         '',
         lambda s: os.path.isdir(s) or s == 'pip')
    LOG_LEVEL = \
        ('LOG_LEVEL',
         f'Logging level ({", ".join(LogLevel.__members__)})',
         'INFO',
         lambda s: s in LogLevel.__members__)


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
        log(LogLevel.PROMPT, f'{param_name} is not specified. Please specify:')
        param_value = ''
        while len(param_value.strip()) == 0 or not is_valid(param_value):
            log(LogLevel.PROMPT,
                f'{ask_query}{"" if len(default_value) == 0 else f" [default: {default_value}]"}: ', end='')
            param_value = input()
            param_value = param_value.strip()
            if len(param_value) == 0:
                param_value = default_value
            if not is_valid(param_value):
                log(LogLevel.PROMPT, 'Invalid value.')
        # custom interaction for MITSUBA3_TRANSIENT_NLOS parameter
        # where the user notifies that it has installed mitsuba 3 through pip
        if param_value == 'pip':
            param_value = ''
        config_dict[param_name] = param_value
        write_config(config_dict)
    return config_dict[param_name]


def read_config() -> dict:
    config = get_config_filename()
    if not os.path.isfile(config):
        log(LogLevel.INFO, 'TAL configuration file not found. '
            f'Creating a new one in {config}...')
        try:
            shutil.copy(local_file_path('.tal.conf.example'), config)
        except Exception as exc:
            log(LogLevel.ERROR,
                f'/!\\ Unknown error when creating config file: {exc}')

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


DEFAULT_CPU_PROCESSES = 1
DEFAULT_DOWNSCALE = None
DEFAULT_CALLBACK = None


class ResourcesConfig:
    """ See tal.set_resources """

    def __init__(self,
                 cpu_processes=DEFAULT_CPU_PROCESSES,
                 downscale=DEFAULT_DOWNSCALE,
                 callback=DEFAULT_CALLBACK):
        self.cpu_processes = cpu_processes
        self.downscale = downscale
        self.callback = callback
        self.force_single_thread = cpu_processes == 1 and downscale is None

    def __enter__(self):
        global TAL_RESOURCES_CONFIG
        self.old_resources = TAL_RESOURCES_CONFIG
        TAL_RESOURCES_CONFIG = self

    def __exit__(self, _, __, ___):
        global TAL_RESOURCES_CONFIG
        TAL_RESOURCES_CONFIG = self.old_resources
        del self.old_resources
        return False

    def split_work(self, f_work, data_in, data_out, slice_dims):

        # TODO(diego): here's some legacy code to set the maximum
        # memory limit of the program. Now e.g. fbp does not use as
        # much memory as before, so this is not really necessary.
        # It would be good to have, but it needs to be tested.
        #
        # import resource
        # import psutil
        # import gc
        # and also with other parameters' names
        # gc.collect()
        # _, hard = resource.getrlimit(resource.RLIMIT_AS)
        # free_memory_bytes = int(psutil.virtual_memory().free * 0.98)
        # free_memory_gb = free_memory_bytes / 2 ** 30
        # log(LogLevel.INFO, 'tal.resources: Setting memory limit to '
        #       f'{free_memory_gb:.2f} GiB.')
        # resource.setrlimit(resource.RLIMIT_AS, (free_memory_bytes, hard))

        def single_process():
            data_out[:] = f_work(data_in)

        if self.force_single_thread:
            single_process()
            return

        in_slice_dim, out_slice_dim = slice_dims

        cpus = min(
            mp.cpu_count(),
            999 if self.cpu_processes == 'max' else self.cpu_processes
        )
        max_downscale = 1024
        downscale = min(max_downscale, self.downscale or cpus)

        if downscale == 1:
            single_process()
            return

        def make_slice(arr, slice_obj=None, slice_dim=-1):
            return tuple(slice_obj if i == slice_dim else slice(None)
                         for i in range(arr.ndim))

        assert in_slice_dim is not None, \
            'Input slice dimension must be specified.'

        assert in_slice_dim is None or out_slice_dim is None or data_in.shape[in_slice_dim] == data_out.shape[out_slice_dim], \
            'Inconsistent dimensions for input and output slices.'

        # maybe downscale is modified because it does not exactly fit with 0 remainder
        old_downscale = None

        if in_slice_dim is None:
            in_slices = [make_slice(data_in),]
        else:
            in_slice_size = int(
                np.ceil(data_in.shape[in_slice_dim] / downscale))

            old_downscale = downscale
            downscale = int(
                np.ceil(data_in.shape[in_slice_dim] / in_slice_size))
            if old_downscale == downscale:
                old_downscale = None

            in_slices = [make_slice(data_in, slice(i * in_slice_size, (i + 1) * in_slice_size), in_slice_dim)
                         for i in range(downscale)]

        if out_slice_dim is None:
            out_slices = [make_slice(data_out),] * len(in_slices)
        else:
            out_slice_size = int(
                np.ceil(data_out.shape[out_slice_dim] / downscale))
            out_slices = [make_slice(data_out, slice(i * out_slice_size, (i + 1) * out_slice_size), out_slice_dim)
                          for i in range(downscale)]

        if old_downscale is None:
            log(LogLevel.INFO, f'tal.resources: Using {cpus} CPU processes '
                f'and downscale {downscale}')
        else:
            log(LogLevel.INFO, f'tal.resources: Using {cpus} CPU processes '
                f'and downscale {downscale} (instead of {old_downscale})')

        with ThreadPoolExecutor(max_workers=cpus) as pool:
            try:
                futures = []
                for in_slice, out_slice in zip(in_slices, out_slices):
                    if out_slice_dim is None:
                        # result needs to be added to data_out later
                        # there would be a race condition if we added it now
                        futures.append(pool.submit(f_work, data_in[in_slice]))
                    else:
                        # result can be added to data_out now
                        # there is no race condition
                        def _process_chunk(i_slice, o_slice):
                            data_out[o_slice] = f_work(data_in[i_slice])

                        futures.append(pool.submit(
                            lambda i_slice=in_slice, o_slice=out_slice: 
                                _process_chunk(i_slice, o_slice)
                        ))

                pool.shutdown(wait=True)
                for i, (f, out_slice) in tqdm(enumerate(zip(futures, out_slices)), total=len(futures),
                                 desc='tal.resources progress', file=TQDMLogRedirect()):
                    # see comment above - either add data now or it was added before
                    if out_slice_dim is None:
                        data_out[out_slice] += f.result()
                    else:
                        f.result()
                    if self.callback is not None:
                        self.callback(data_out, i, len(futures))
            # NOTE(diego): See TODO about memory limit above
            # except MemoryError:
            #     pool.terminate()
            #     pool.join()
            #     raise MemoryError(f'tal.resources: Memory error, {free_memory_gb:.2f} GiB is not enough. '
            #                        'Either decrease CPU processes, increase downscale, or move to another system.')
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False)
                raise KeyboardInterrupt


TAL_RESOURCES_CONFIG = ResourcesConfig()


def get_resources():
    return TAL_RESOURCES_CONFIG


def set_resources(cpu_processes=DEFAULT_CPU_PROCESSES, downscale=DEFAULT_DOWNSCALE, callback=DEFAULT_CALLBACK):
    """
    Configures how Y-TAL should execute your code so you can manage CPU/RAM use.

    Not all functions implemented in Y-TAL support this configuration, mostly the ones
    in tal.reconstruct (filtering and reconstruction).

    Default configuration is to process all the data simultaneously in 1 CPU process.

    cpu_processes
        Can be an integer or 'max' to use all available CPU processes.

    downscale
        Can be an integer or None.
        If integer, the data will be processed in smaller chunks (e.g. if downscale = 8, the
            data will be processed 1/8th at a time).
        If None, all the data will be processed at the same time.
        In any case, downscale will be increased automatically make sure that all cpu_processes
            can run at the same time.

    callback
        A function that will be called after each chunk of data is processed.
        It will receive (1) the data, (2) the current chunk index and (3) the total number of chunks as arguments.
        This is useful to show a progress bar, or partial reconstructions for example.
        See tal.callbacks for some default implementations which may or may not work for your case.
        e.g. callback=lambda x: `tal.plot.amplitude_phase(x[0], title=f'Chunk {x[1]} of {x[2]}')`
    """
    global TAL_RESOURCES_CONFIG
    TAL_RESOURCES_CONFIG = ResourcesConfig(cpu_processes, downscale, callback)
