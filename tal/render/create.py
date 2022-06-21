import os
import shutil
from tal.config import local_file_path
from tal.render.mitsuba2_transient_nlos import get_material_keys, get_materials
from tal.util import fdent
from textwrap import indent
import tal
import datetime


def create_nlos_scene(folder_name, args):
    folder_name = os.path.abspath(folder_name)
    path, name = os.path.split(folder_name)

    assert os.path.isdir(path), f'{path} does not exist'
    if os.path.exists(folder_name):
        answer = None
        num_files = sum(map(lambda e: len(e[2]), os.walk(folder_name)))
        while answer is None:
            answer = input(
                f'WARNING: {folder_name} already exists and contains {num_files} file{"" if num_files == 1 else "s"}.\n'
                f'Erase ALL and replace? (y/N): ')
        print()
        if answer.lower() != 'y' and answer.lower() != 'yes':
            print('Operation cancelled.')
            exit()
        else:
            shutil.rmtree(folder_name)

    try:
        os.mkdir(folder_name)
    except OSError as exc:
        raise AssertionError(f'Invalid permissions: {exc}') from exc

    config_name = os.path.join(folder_name, f'{name}.yaml')
    default_yaml_data = open(local_file_path(
        'render/scene_defaults.yaml'), 'r').readlines()
    first_uncomment = default_yaml_data.index('geometry:\n')
    data_commented, data_uncommented = \
        default_yaml_data[:first_uncomment], default_yaml_data[first_uncomment:]
    data_commented = ''.join(list(
        map(lambda l: l if len(l.strip()) == 0 or l.startswith('#') else f'#{l}', data_commented)))
    default_yaml_data = ''.join([*data_commented, *data_uncommented])

    def print_material(item):
        key, value = item

        material_keys = list(
            map(lambda s: f'{s}: ${s}', get_material_keys(value)))
        value = indent(value, '#| ')

        if len(material_keys) == 0:
            return fdent('''\
                #| id: {key}
                {value}''', key=key, value=value)
        else:
            material_keys = indent('\n'.join(material_keys), '#| ')
            return fdent('''\
                #| id: {key}
                {material_keys}
                {value}''', key=key, material_keys=material_keys, value=value)

    default_material_data = '\n\n'.join(
        map(print_material, get_materials().items()))

    with open(config_name, 'w') as f:
        f.write(
            '# TAL v{v} NLOS scene description file: {r}\n'.format(
                v=tal.__version__, r='https://github.com/diegoroyo/tal'))
        f.write('# Created on {d} with experiment name "{n}"\n'.format(
                d=datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M'), n=name))
        f.write('#\n')
        f.write('# All commented variables show their default value\n')
        f.write('# To render the scene with this configuration, execute:\n')
        f.write('#   [python3 -m] tal render {c}\n'.format(
                c=config_name))
        f.write('name: {name}\n'.format(name=name))
        f.write('\n')
        f.write(default_yaml_data)
        f.write(default_material_data)

    if not args.quiet:
        print(f'Success! Now:\n'
              f'1) Edit the configuration file in {config_name}\n'
              f'2) Render with tal render {config_name}')
