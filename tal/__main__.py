import argparse
import os
from tal.util import fdent


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


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
    else:
        raise AssertionError('Invalid command? Check argparse')


if __name__ == '__main__':
    main()
