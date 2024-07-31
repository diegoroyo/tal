import argparse
import os
from tal import __version__ as tal_version
from tal.util import fdent
from tal.log import log, LogLevel


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def get_plot_functions():
    """
    Returns (function_names, function_params) where
    function_names is a list of str
    function_params is a list of tuples
        (param_name, param_type)
    """
    from inspect import getmembers, isfunction, signature
    import tal.plot
    function_names = list()
    function_param_names = dict()
    function_param_data = set()
    for name, func in getmembers(tal.plot, isfunction):
        function_names.append(name)
        parameters = signature(func).parameters
        parameter_names = list(filter(lambda p: p not in ['data', 'data_list', 'args', 'kwargs'],
                                      list(parameters)))
        function_param_names[name] = parameter_names
        function_param_data.update(
            list(map(lambda p: (p, parameters[p].annotation),
                     parameter_names)))
    return function_names, function_param_names, function_param_data


def get_tal_version_string():
    from tal.render.util import import_mitsuba_backend

    def f(path):
        return os.path.split(os.path.abspath(path))[0]

    try:
        render_backend = import_mitsuba_backend()
        render_backend_version = f'{render_backend.get_name()}\n{render_backend.get_long_version()}'
    except AssertionError:
        render_backend_version = 'No render backend found'

    return f'tal v{tal_version} ({f(__file__)})\n\n* Render backend: {render_backend_version}'


class LazyVersionAction(argparse.Action):
    def __init__(self, option_strings, version=None, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings,
                         dest=dest, default=default, nargs=0, help=help)
        self.version = version

    def __call__(self, parser, namespace, values, option_string=None):
        version_str = self.version() if callable(self.version) else self.version
        parser.exit(message=f'{version_str}\n')


def main():
    from tal.log import _set_default_log_level
    _set_default_log_level()

    parser = argparse.ArgumentParser(
        description=f'Y-TAL - (Your) Transient Auxiliary Library - v{tal_version}', formatter_class=SmartFormatter)
    parser.add_argument('-v', '--version', action=LazyVersionAction,
                        version=get_tal_version_string)
    subparsers = parser.add_subparsers(
        help='Command', required=True, dest='command')

    # config commands
    config_parser = subparsers.add_parser(
        'config', help='Edit the TAL configuration file')

    # render commands
    render_parser = subparsers.add_parser(
        'render', help='Create, edit or execute renders of simulated NLOS scene data captures', formatter_class=SmartFormatter)
    render_parser.add_argument('config_file', nargs='*', help=fdent('''\
                                    R|Can be:
                                    1) Path to a TAL scene config YAML file
                                    2) Path to a TAL scene directory (must have a YAML file inside with the same name as the directory)
                                    3) 'new <folder_name>' to create a new folder (i.e. tal render new <folder_name>)'''))
    render_parser.add_argument('-t', '--threads',
                               type=int, default=os.cpu_count() - 1, required=False,
                               help='Number of CPU threads')
    render_parser.add_argument('-s', '--seed',
                               type=int, default=0, required=False,
                               help='Random seed for the sampler. Without setting this value to different values, the same results will be produced everytime')
    render_parser.add_argument('-n', '--nice',
                               type=int, default=0, required=False,
                               help='Change +/- in nice factor. Positive values = lower priority. Negative values = higher priority (needs sudo)')
    render_parser.add_argument('-g', '--gpu',
                               dest='gpus', nargs='*', type=int, default=[], required=False,
                               help='Select which GPUs should be used by Mitsuba via the CUDA_VISIBLE_DEVICES env. variable')
    render_parser.add_argument('-r', '--dry-run',
                               dest='dry_run', action='store_true',
                               help='Do not execute mitsuba, just print out the commands that would be executed (if any)')
    render_parser.add_argument('--no-steady',
                               dest='do_steady_renders', action='store_false',
                               help='Disable generation of steady state images during render')
    render_parser.add_argument('--no-ground-truth',
                               dest='do_ground_truth_renders', action='store_false',
                               help='Disable generation of ground truth depths and normals from the hidden geometry')
    render_parser.add_argument('--no-logging',
                               dest='do_logging', action='store_false',
                               help='Disable logging of mitsuba output')
    render_parser.add_argument('--no-partial-results',
                               dest='keep_partial_results', action='store_false',
                               help='Delete the "partial" folder (which stores raw render results) after finishing the render and generating the final HDF5 file')

    # plot commands
    plot_parser = subparsers.add_parser(
        'plot', help='Plot capture data using one of the configured methods', formatter_class=SmartFormatter)
    plot_func_names, plot_func_param_names, plot_func_param_data = get_plot_functions()
    plot_parser.add_argument('preset', help=fdent('''\
                                    R|Plot method. Can be one of:
                                        {v}''', v='\n'.join(plot_func_names)))
    plot_parser.add_argument('capture_files', nargs='*',
                             help='One or more paths to capture files')
    for var_name, var_type in plot_func_param_data:
        plot_parser.add_argument(
            '--{}'.format(var_name.replace('_', '-')), type=var_type, required=False)

    args = parser.parse_args()

    if args.command == 'config':
        from tal.config import get_config_filename
        log(LogLevel.INFO, 'TAL configuration file is located at {l}'.format(
            l=get_config_filename()))
        log(LogLevel.INFO, 'Updating render backend configuration...')
        from tal.config import ask_for_config, Config
        version = ask_for_config(Config.MITSUBA_VERSION, force_ask=True)
        if version == '2':
            ask_for_config(
                Config.MITSUBA2_TRANSIENT_NLOS_FOLDER, force_ask=True)
        elif version == '3':
            ask_for_config(
                Config.MITSUBA3_TRANSIENT_NLOS_FOLDER, force_ask=True)
            found = False
            while not found:
                log(LogLevel.INFO, 'Checking if Mitsuba 3 can be found...')
                try:
                    import mitsuba as mi
                    log(LogLevel.INFO, f'OK Mitsuba 3 can be found (without setpath.sh at {mi.__file__})')
                    found = True
                except ModuleNotFoundError:
                    from tal.render.mitsuba3_transient_nlos import add_mitsuba_to_path
                    add_mitsuba_to_path()
                    try:
                        import mitsuba as mi
                        log(LogLevel.INFO, f'OK Mitsuba 3 can be found (with setpath.sh at {mi.__file__})')
                        found = True
                    except ModuleNotFoundError:
                        log(LogLevel.PROMPT, 'Mitsuba 3 cannot be found. '
                            'Please install Mitsuba 3 using \'pip install mitsuba\' (and write \'pip\' in the next prompt) '
                            'or compile Mitsuba 3 and point to your installation folder:')
                        ask_for_config(
                            Config.MITSUBA3_TRANSIENT_NLOS_FOLDER, force_ask=True)
        else:
            raise AssertionError(
                f'Invalid MITSUBA_VERSION={version}, must be one of (2, 3)')
        log(LogLevel.INFO, 'Done.')
    elif args.command == 'render':
        config_file = args.config_file
        assert len(config_file) != 1 or config_file[0] != 'new', \
            'You must specify a folder name: tal render new <new_folder_name>'
        assert (len(config_file) == 1) or \
            (len(config_file) == 2 and config_file[0].lower() == 'new'), \
            'Usage: tal render new <new_folder_name> | tal render <config_file.yaml>"'
        if config_file[0].lower() == 'new':
            new_folder_name = config_file[1]
            from tal.render import create_nlos_scene
            create_nlos_scene(new_folder_name, args)
        else:
            from tal.render import render_nlos_scene
            config_file = config_file[0]
            render_nlos_scene(config_file, args)
    elif args.command == 'plot':
        import tal.plot
        from tal.io import read_capture
        command = args.preset
        assert command in plot_func_names, \
            'Unknown plot command: {}'.format(command)
        data = list()
        labels = []
        for capture_file in args.capture_files:
            log(LogLevel.INFO, f'Reading {capture_file}...')
            labels.append(capture_file)
            data.append(read_capture(capture_file))
        if len(data) == 1:
            data = data[0]
        if 'labels' in plot_func_param_names[command]:
            setattr(args, 'labels', labels)
        other_args = list(map(
            lambda p: None if p not in args else getattr(args, p),
            plot_func_param_names[command]))
        getattr(tal.plot, command)(data, *other_args)
    else:
        raise AssertionError('Invalid command? Check argparse')


if __name__ == '__main__':
    main()
