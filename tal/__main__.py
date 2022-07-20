import argparse
import os
from tal.util import fdent


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


def main():
    parser = argparse.ArgumentParser(
        description='TAL - Transient Auxiliary Library', formatter_class=SmartFormatter)
    subparsers = parser.add_subparsers(
        help='Command', required=True, dest='command')

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
                               help='Number of threads')
    render_parser.add_argument('-n', '--nice',
                               type=int, default=0, required=False,
                               help='Change +/- in nice factor. Positive values = lower priority. Negative values = higher priority (needs sudo)')
    render_parser.add_argument('-q', '--quiet',
                               dest='quiet', action='store_true',
                               help='Disable progress bars and other verbose outputs')
    render_parser.add_argument('-r', '--dry-run',
                               dest='dry_run', action='store_true',
                               help='Do not execute mitsuba, just print out the commands that would be executed')
    render_parser.add_argument('--no-steady',
                               dest='do_steady_renders', action='store_false',
                               help='Disable generation of steady state images')
    render_parser.add_argument('--no-logging',
                               dest='do_logging', action='store_false',
                               help='Disable logging of mitsuba2 output')
    render_parser.add_argument('--no-partial-results',
                               dest='keep_partial_results', action='store_false',
                               help='Remove the "partial" folder which stores temporal data after creating the final hdf5 file (e.g. multiple experiments for confocal/exhaustive)')

    # render commands
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

    if args.command == 'render':
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
        for capture_file in args.capture_files:
            print(f'Reading {capture_file}...')
            data.append(read_capture(capture_file))
        if len(data) == 1:
            data = data[0]
        other_args = list(map(
            lambda p: None if p not in args else getattr(args, p),
            plot_func_param_names[command]))
        getattr(tal.plot, command)(data, *other_args)
    else:
        raise AssertionError('Invalid command? Check argparse')


if __name__ == '__main__':
    main()
