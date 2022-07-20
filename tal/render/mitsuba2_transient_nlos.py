def _get_setpath_location():
    from tal.config import ask_for_config, Config
    import os
    force_ask = False
    setpath_ok = False
    while not setpath_ok:
        setpath_location = os.path.join(
            ask_for_config(Config.MITSUBA2_TRANSIENT_NLOS_FOLDER,
                           force_ask=force_ask),
            'setpath.sh')
        if os.path.isfile(setpath_location):
            setpath_ok = True
        else:
            force_ask = True
            print(f'setpath.sh cannot be found in {setpath_location}.')
            print()
    return setpath_location


try:
    import mitsuba  # pyright: reportMissingImports=false
except ModuleNotFoundError:
    import sys
    import subprocess
    setpath_location = _get_setpath_location()

    command = ['env', '-i', '/bin/bash',
               '-c', f'source {setpath_location} && printenv']
    p = subprocess.check_output(command).decode('utf-8')
    for line in p.split('\n')[:-1]:
        (key, _, value) = line.partition('=')
        if key == 'PYTHONPATH':
            for directory in value.split(':')[:-1]:
                sys.path.append(directory)


def mitsuba_set_variant(s):
    import mitsuba
    mitsuba.set_variant(s)


def get_material_keys(s):
    import re
    return list(map(lambda e: e[1:],
                    re.compile(r'\$[a-zA-Z_]*').findall(s)))


def get_materials():
    from tal.util import fdent
    return {
        'white': fdent(f'''\
            <bsdf type="diffuse">
                <rgb name="reflectance" value="1.0, 1.0, 1.0"/>
            </bsdf>'''),
        'copper': fdent(f'''\
            <bsdf type="roughconductor">
                <string name="material" value="Cu"/>
                <string name="distribution" value="beckmann"/>
                <float name="alpha" value="$alpha"/>
            </bsdf>'''),
        'custom': '$text'
    }


def run_mitsuba(scene_xml_path, exr_path, defines,
                experiment_name, logfile, args, sensor_index=0):
    import re
    import time
    import subprocess
    import os
    from tqdm import tqdm
    # execute mitsuba command (sourcing setpath.sh before)
    num_threads = args.threads
    command = ['mitsuba',
               '-o', exr_path,
               '-s', str(sensor_index),
               '-t', str(num_threads)]
    for key, value in defines.items():
        command += ['-D', f'{key}={value}']
    command += [scene_xml_path]

    nice = args.nice
    command = ['nice', '-n', str(nice), " ".join(command)]

    setpath_location = _get_setpath_location()
    command = ['/bin/bash', '-c',
               f'source "{setpath_location}" && {" ".join(command)}']

    if args.dry_run:
        print(' '.join(command))
        return

    if args.quiet:
        # simplified version, block until done rendering
        mitsuba_process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # need to pass the command through stdbuf to be able to read the progress bar
        command = ['stdbuf', '-o0'] + command
        mitsuba_process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # read the progress bar and pass the info to the user through a tqdm bar
        # this is totally not overengineering-trust me-this saves so much time
        progress_re = re.compile(
            r'Rendering \[(=* *)\] \([\d\.]+\w+, ETA: ([\d\.]+\w+)\)')
        read_opl = defines.get('auto_detect_bins', False)
        if read_opl:
            opl_output = ''
            opl_re = re.compile(
                r'limits: \[(\d+\.\d+), \d+\.\d+\] with bin width (\d+\.\d+)')
        with tqdm(desc=experiment_name, total=100, ascii=True, leave=False,
                  bar_format='{desc} |{bar}| [{n:.2f}%{postfix}] ') as pbar:
            output = None
            while output is None or len(output) > 0:
                output = mitsuba_process.stdout.read(160)
                try:
                    output = output.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                if logfile is not None:
                    logfile.write(output)
                    logfile.flush()
                if read_opl:
                    opl_output += output
                    matches = opl_re.findall(opl_output)
                    if len(matches) > 0:
                        start_opl, bin_width_opl = matches[-1]
                        start_opl = float(start_opl)
                        bin_width_opl = float(bin_width_opl)
                        print('Auto-detected histogram: '
                              f'start_opl={start_opl:.4f}, bin_width_opl={bin_width_opl:.6f}')
                        defines.update(start_opl=start_opl)
                        defines.update(bin_width_opl=bin_width_opl)
                        read_opl = False
                        del opl_output
                matches = progress_re.findall(output)
                if len(matches) > 0:
                    progress, eta = matches[-1]
                    completed = progress.count('=')
                    not_completed = progress.count(' ')
                    progress = 100 * completed / (completed + not_completed)
                    pbar.update(progress - pbar.n)
                    pbar.set_postfix_str(f'ETA: {eta}')
                    if not_completed == 0:
                        break
                    time.sleep(1)

    # wait for the process to end
    mitsuba_process.communicate()
    assert mitsuba_process.returncode == 0, \
        f'Mitsuba returned with error code {mitsuba_process.returncode}'
    if os.path.isdir(exr_path):
        # assume transient render
        n_exr_frames = len(os.listdir(exr_path))
        assert n_exr_frames > 0, 'No frames were rendered?'
    else:
        # assume steady state render
        assert os.path.isfile(exr_path), 'No frames were rendered?'


def read_mitsuba_bitmap(path: str):
    from mitsuba.core import Bitmap
    import numpy as np
    return np.array(Bitmap(path), copy=False)


def read_mitsuba_streakbitmap(path: str, exr_format='RGBA'):
    """
    Reads all the images x-t that compose the streak image.

    :param dir: path where the images x-t are stored
    :return: a streak image of shape [time, width, height]
    """
    import re
    import glob
    import os
    import numpy as np
    from tqdm import tqdm

    # FIXME(diego): for now this assumes that the EXR that it reads
    # are in RGBA format, and returns an image with 3 channels,
    # in the case of polarized light it should return something else
    if exr_format != 'RGBA':
        raise NotImplementedError(
            'Formats different from RGBA are not implemented')

    xtframes = glob.glob(os.path.join(
        glob.escape(path), f'frame_*.exr'))
    xtframes = sorted(xtframes,
                      key=lambda x: int(re.compile(r'\d+').findall(x)[-1]))
    number_of_xtframes = len(xtframes)
    first_img = read_mitsuba_bitmap(xtframes[0])
    streak_img = np.empty(
        (number_of_xtframes, *first_img.shape), dtype=first_img.dtype)
    with tqdm(desc=f'Reading {path}', total=number_of_xtframes, ascii=True) as pbar:
        for i_xtframe in range(number_of_xtframes):
            other = read_mitsuba_bitmap(xtframes[i_xtframe])
            streak_img[i_xtframe] = np.nan_to_num(other, nan=0.)
            pbar.update(1)

    # for now streak_img has dimensions (y, x, time, channels)
    assert streak_img.shape[-1] == 4, \
        f'Careful, streak_img has shape {streak_img.shape} (i.e. its probably not RGBA as we assume)'
    # and we want it as (time, x, y)
    return np.sum(np.transpose(streak_img), axis=0)
