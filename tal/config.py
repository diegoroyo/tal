import dill
import os
import shutil
import yaml
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from enum import Enum

# FIXME decide if we keep this here or not
import resource
import psutil
import gc

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


def _run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def _apply_async(pool, fun, args, **kwargs):
    payload = dill.dumps((fun, args))
    return pool.apply_async(_run_dill_encoded, (payload,), **kwargs)


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

        # FIXME decide what to do with memory management
        # and also with other parameters' names
        # gc.collect()
        # _, hard = resource.getrlimit(resource.RLIMIT_AS)
        # free_memory_bytes = int(psutil.virtual_memory().free * 0.98)
        # free_memory_gb = free_memory_bytes / 2 ** 30
        # print('tal.resources: Setting memory limit to '
        #       f'{free_memory_gb:.2f} GiB.')
        # resource.setrlimit(resource.RLIMIT_AS, (free_memory_bytes, hard))

        def single_process():
            data_out[:] = f_work(data_in)

        if self.force_single_thread:
            single_process()
            return

        in_slice_dim, out_slice_dim = slice_dims

        def check_divisible(x):
            if in_slice_dim is not None and data_in.shape[in_slice_dim] % x != 0:
                return False
            if out_slice_dim is not None and data_out.shape[out_slice_dim] % x != 0:
                return False
            return True

        cpus = min(
            mp.cpu_count(),
            999 if self.cpu_processes == 'max' else self.cpu_processes
        )
        max_downscale = 128
        if self.downscale is None:
            downscale = 1
        else:
            downscale = 2**int(np.ceil(np.log2(self.downscale)))
        downscale = min(max_downscale, downscale)
        while downscale < cpus and downscale < max_downscale and check_divisible(downscale * 2):
            downscale *= 2

        data_dim = min((x for x in [
                        None if in_slice_dim is None else data_in.shape[in_slice_dim],
                        None if out_slice_dim is None else data_out.shape[out_slice_dim],
                        ] if x is not None), default=None)

        if downscale == 1:
            single_process()
            return

        print(f'tal.resources: Using {cpus} CPU processes '
              f'and downscale {downscale}.')

        if in_slice_dim is not None:
            in_shape = np.insert(data_in.shape, in_slice_dim + 1, downscale)
            in_shape[in_slice_dim] //= downscale
        if out_slice_dim is not None:
            out_shape = np.insert(data_out.shape, out_slice_dim + 1, downscale)
            out_shape[out_slice_dim] //= downscale

        with mp.Pool(processes=cpus) as pool:
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
                tqdm_kwargs = {
                    'total': len(asyncs),
                    'desc': 'tal.resources progress',
                }
                if out_slice_dim is None:
                    for i, async_ in tqdm(enumerate(asyncs), **tqdm_kwargs):
                        data_out += async_.get()
                        if self.callback is not None:
                            self.callback(data_out, i, len(asyncs))
                else:
                    for i, (async_, out_slice) in tqdm(enumerate(zip(asyncs, data_out)), **tqdm_kwargs):
                        out_slice[:] = async_.get()
                        if self.callback is not None:
                            self.callback(data_out, i, len(asyncs))
                pool.join()
            # FIXME decide if this should be here
            # except MemoryError:
            #     pool.terminate()
            #     pool.join()
            #     raise MemoryError(f'tal.resources: Memory error, {free_memory_gb:.2f} GiB is not enough.'
            #                       ' Either decrease CPU processes, increase downscale, or move to another system.')
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
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
