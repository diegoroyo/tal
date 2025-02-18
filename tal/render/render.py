import os
import shutil
import yaml
import tal
import platform
from tal.io.capture_data import NLOSCaptureData
from tal.enums import FileFormat, GridFormat, HFormat, GroundTruthFormat
from tal.config import local_file_path
from tal.log import log, LogLevel, TQDMLogRedirect
from tal.render.util import get_grid_xyz, expand_xy_dims
import datetime
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

from tal.render.util import import_mitsuba_backend


def _read_config_and_init_mitsuba_variant(config_path, args):
    # FIXME: on macOS "spawn" method, which is the default since 3.8,
    # is considered more safe than "fork", but requires serialization methods available
    # to send the objects to the spawned process. So a proper fix would be to add them
    # (see e.g. https://stackoverflow.com/a/65513291 and
    # https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    # for more details)
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork')

    config_path = os.path.abspath(config_path)

    assert os.path.exists(config_path), \
        f'{config_path} does not exist'

    path, name = os.path.split(config_path)
    candidate_filename = os.path.join(path, name, f'{name}.yaml')
    if os.path.isdir(config_path) and os.path.isfile(candidate_filename):
        config_path = candidate_filename

    assert os.path.isfile(config_path), \
        f'{config_path} is not a TAL config file'

    try:
        scene_config = yaml.safe_load(
            open(config_path, 'r')) or dict()

        if scene_config.get('mitsuba_variant', '').startswith('cuda'):
            assert len(args.gpus) > 0, \
                'You must specify at least one GPU to use CUDA. Use tal --gpu <id1> <id2> ...'
            gpu_ids = ','.join(map(str, args.gpus))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        mitsuba_backend = import_mitsuba_backend()

        scene_defaults = yaml.safe_load(
            open(local_file_path('render/scene_defaults.yaml'), 'r'))
        scene_defaults['mitsuba_variant'] = mitsuba_backend.get_default_variant()

    except yaml.YAMLError as exc:
        raise AssertionError(
            f'Invalid YAML format in TAL config file: {exc}') from exc
    scene_config = {**scene_defaults, **scene_config}
    if not args.dry_run:
        mitsuba_backend.set_variant(scene_config['mitsuba_variant'])

    return mitsuba_backend, scene_config, config_path


def _check_progress_and_create_folders(config_path, args):
    config_dir, config_filename = os.path.split(config_path)
    try:
        in_progress = False
        progress_file = os.path.join(config_dir, 'IN_PROGRESS')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_folder = f.read()
            if os.path.exists(os.path.join(config_dir, progress_folder)):
                in_progress = True
            else:
                log(LogLevel.INFO, 'The IN_PROGRESS file is stale, removing it...')
                os.remove(progress_file)
        if in_progress:
            log(LogLevel.INFO,
                f'Found a render in progress ({progress_folder}), continuing...')
        if not in_progress:
            progress_folder = datetime.datetime.now().strftime(r'%Y%m%d-%H%M%S')

        root_dir = os.path.join(config_dir, progress_folder)
        partial_results_dir = os.path.join(root_dir, 'partial')
        steady_dir = os.path.join(root_dir, 'steady')
        log_dir = os.path.join(root_dir, 'logs')

        if not in_progress:
            os.mkdir(root_dir)
            os.mkdir(partial_results_dir)
            os.mkdir(steady_dir)
            os.mkdir(log_dir)
            shutil.copy(
                config_path,
                os.path.join(root_dir, f'{config_filename}.old'))
        with open(progress_file, 'w') as f:
            f.write(progress_folder)
    except OSError as exc:
        raise AssertionError(f'Invalid permissions: {exc}') from exc

    return root_dir, partial_results_dir, steady_dir, log_dir, progress_file


def __write_scene_xmls(args, mitsuba_backend, scene_config, root_dir):
    steady_xml, ground_truth_xml, nlos_xml = mitsuba_backend.get_scene_xml(
        scene_config, random_seed=args.seed)

    steady_scene_xml = os.path.join(root_dir, 'steady_scene.xml')
    with open(steady_scene_xml, 'w') as f:
        f.write(steady_xml)

    ground_truth_scene_xml = os.path.join(
        root_dir, 'ground_truth_scene.xml')
    with open(ground_truth_scene_xml, 'w') as f:
        f.write(ground_truth_xml)

    nlos_scene_xml = os.path.join(root_dir, 'nlos_scene.xml')
    with open(nlos_scene_xml, 'w') as f:
        f.write(nlos_xml)

    return steady_scene_xml, ground_truth_scene_xml, nlos_scene_xml


def __write_metadata_and_get_laser_lookats(args, scene_config):
    """ Compute laser_lookats """

    laser_lookats = []
    scan_type = scene_config['scan_type']
    num_bins = scene_config['num_bins']
    sensor_width = scene_config['sensor_width']
    sensor_height = scene_config['sensor_height']
    laser_width = scene_config['laser_width']
    laser_height = scene_config['laser_height']
    if scan_type == 'single':
        laser_width = 1
        laser_height = 1

    relay_wall = next(filter(
        lambda g: g['name'] == scene_config['relay_wall'],
        scene_config['geometry']))
    assert 'rot_degrees_x' not in relay_wall and \
        'rot_degrees_y' not in relay_wall and \
        'rot_degrees_z' not in relay_wall, \
        'Relay wall displacement/rotation is NYI'

    laser_aperture_start_x = scene_config['laser_aperture_start_x'] or 0
    laser_aperture_start_y = scene_config['laser_aperture_start_y'] or 0
    laser_aperture_end_x = scene_config['laser_aperture_end_x'] or 1
    laser_aperture_end_y = scene_config['laser_aperture_end_y'] or 1

    if scan_type == 'single':
        laser_lookat_x = \
            scene_config['laser_lookat_x'] or sensor_width / 2
        laser_lookat_y = \
            scene_config['laser_lookat_y'] or sensor_height / 2
        laser_lookats.append((0, 0, laser_lookat_x, laser_lookat_y))
    elif scan_type == 'exhaustive' or scan_type == 'confocal':
        assert not (scan_type == 'confocal' and
                    (laser_width != sensor_width or
                        laser_height != sensor_height)), \
            'If using scan_type=confocal, sensor_{width|height} must match laser_{width|height}'

        for y in range(laser_height):
            for x in range(laser_width):
                # start in (0, 1) space
                laser_lookat_x = (x + 0.5) / laser_width
                laser_lookat_y = (y + 0.5) / laser_height
                # take aperture into account
                laser_lookat_x = laser_aperture_start_x + \
                    laser_lookat_x * \
                    (laser_aperture_end_x - laser_aperture_start_x)
                laser_lookat_y = laser_aperture_start_y + \
                    laser_lookat_y * \
                    (laser_aperture_end_y - laser_aperture_start_y)
                # finally store in sensor space (0, sensor_width)
                laser_lookat_x *= sensor_width
                laser_lookat_y *= sensor_height
                laser_lookats.append((x, y, laser_lookat_x, laser_lookat_y))
    else:
        raise AssertionError(
            'Invalid scan_type, must be one of {single|exhaustive|confocal}')

    """ Create NLOSCaptureData and write all metadata """

    # TODO(diego): rotate sensor_grid_xyz and laser_grid_xyz based on relay wall rotation
    displacement = np.array([
        relay_wall['displacement_x'],
        relay_wall['displacement_y'],
        relay_wall['displacement_z']])
    sensor_grid_xyz = get_grid_xyz(
        sensor_width, sensor_height, relay_wall['scale_x'], relay_wall['scale_y'])
    sensor_grid_xyz += displacement
    if scan_type == 'single':
        px = relay_wall['scale_x'] * \
            ((laser_lookat_x / sensor_width) * 2 - 1)
        py = relay_wall['scale_y'] * \
            ((laser_lookat_y / sensor_height) * 2 - 1)
        laser_grid_xyz = np.array([[
            [px, py, 0],
        ]], dtype=np.float32)
    else:
        laser_grid_xyz = get_grid_xyz(
            laser_width, laser_height, relay_wall['scale_x'], relay_wall['scale_y'],
            ax0=laser_aperture_start_x, ax1=laser_aperture_end_x,
            ay0=laser_aperture_start_y, ay1=laser_aperture_end_y)
    laser_grid_xyz += displacement

    # TODO(diego): rotate [0, 0, 1] by rot_degrees_x (assmes RW is a plane)
    # or use a more generalist approach
    sensor_grid_normals = expand_xy_dims(
        np.array([0, 0, 1]), sensor_width, sensor_height)
    laser_grid_normals = expand_xy_dims(
        np.array([0, 0, 1]), laser_width, laser_height)

    capture_data = NLOSCaptureData()
    if scan_type == 'single' or scan_type == 'confocal':
        capture_data.H = np.zeros(
            (num_bins, sensor_width, sensor_height),
            dtype=np.float32)
        capture_data.H_format = HFormat.T_Sx_Sy
    elif scan_type == 'exhaustive':
        capture_data.H = np.zeros(
            (num_bins, laser_width, laser_height, sensor_width, sensor_height),
            dtype=np.float32)
        capture_data.H_format = HFormat.T_Lx_Ly_Sx_Sy
    else:
        raise AssertionError(
            'Invalid scan_type, must be one of {single|exhaustive|confocal}')
    capture_data.sensor_xyz = np.array([
        scene_config['sensor_x'],
        scene_config['sensor_y'],
        scene_config['sensor_z'],
    ], dtype=np.float32)
    capture_data.sensor_grid_xyz = sensor_grid_xyz
    capture_data.sensor_grid_normals = sensor_grid_normals
    capture_data.sensor_grid_format = GridFormat.X_Y_3
    capture_data.laser_xyz = np.array([
        scene_config['laser_x'],
        scene_config['laser_y'],
        scene_config['laser_z'],
    ], dtype=np.float32)
    capture_data.laser_grid_xyz = laser_grid_xyz
    capture_data.laser_grid_normals = laser_grid_normals
    capture_data.laser_grid_format = GridFormat.X_Y_3
    capture_data.delta_t = scene_config['bin_width_opl']
    capture_data.t_start = scene_config['start_opl']
    capture_data.t_accounts_first_and_last_bounces = \
        scene_config['account_first_and_last_bounces']
    capture_data.scene_info = {
        'tal_version': tal.__version__,
        'config': scene_config,
        'args': vars(args),
    }

    return scan_type, laser_lookats, capture_data


def __run_mitsuba(args, log_path, mitsuba_backend, mitsuba_variant, scene_xml, hdr_path, defines,
                  experiment_name, render_name, sensor_index, check_done=lambda: False):
    if check_done():
        log(LogLevel.INFO, f'Skipping {render_name} for {experiment_name}')
        return
    if not args.dry_run:
        mitsuba_backend.set_variant(mitsuba_variant)

    class Tee:
        def __init__(self, pipe=None, logfile=None):
            self.pipe = pipe
            self.logfile = logfile

        def write(self, data):
            if self.pipe is None:
                log(LogLevel.INFO, data)
            else:
                self.pipe.send(data)
            if logfile is not None:
                self.logfile.write(data)

        def flush(self):
            if self.logfile is not None and not self.logfile.closed:
                self.logfile.flush()

        def close(self):
            if self.pipe is not None:
                self.pipe.send(None)
                self.pipe.close()
            if self.logfile is not None:
                self.logfile.flush()
                self.logfile.close()

    logfile = None
    if args.do_logging and not args.dry_run:
        logfile = open(log_path, 'w')

    run_in_different_process = \
        os.name != 'nt' and not mitsuba_variant.startswith('cuda')

    pipe_w = None
    if run_in_different_process:
        # NOTE: Windows does not support multiprocessing
        # and CUDA is not initalized if its run on a different process
        pipe_r, pipe_w = multiprocessing.Pipe()

    output = Tee(pipe_w, logfile)
    # NOTE: something here has a memory leak (probably Mitsuba-related)
    # We run Mitsuba in a separate process to ensure that the leaks do not add up
    # as they can fill your RAM in exhaustive scans
    run_mitsuba_f = partial(mitsuba_backend.run_mitsuba, scene_xml, hdr_path, defines,
                            render_name, args, output, sensor_index)
    if not run_in_different_process:
        run_mitsuba_f()
    else:
        process = multiprocessing.Process(target=run_mitsuba_f)
        try:
            process.start()
            while True:
                line = pipe_r.recv()
                if line:
                    if len(line.strip()) > 0:
                        log(LogLevel.INFO, line)
                else:
                    raise EOFError
        except EOFError:
            pipe_r.close()
            process.join()
        except KeyboardInterrupt:
            process.terminate()
            raise KeyboardInterrupt


class RenderException(Exception):
    pass


def __merge_gt_results(args, mitsuba_backend, capture_data, gt_path):
    if not args.do_ground_truth_renders:
        return capture_data

    gt_image = mitsuba_backend.read_mitsuba_bitmap(gt_path)
    depth = gt_image[:, :, 0:3]
    normals = gt_image[:, :, 3:6]
    capture_data.scene_info['ground_truth'] = {
        'format': GroundTruthFormat.X_Y,
        'depth': depth,
        'normals': normals,
    }

    return capture_data


def __merge_nlos_results(args, mitsuba_backend, capture_data, partial_results_dir, experiment_name, scan_type, laser_lookats):

    if scan_type == 'single':
        hdr_path, _ = mitsuba_backend.partial_laser_path(
            partial_results_dir,
            experiment_name,
            *laser_lookats[0][0:2])
        capture_data.H = mitsuba_backend.read_transient_image(hdr_path)
    elif scan_type == 'exhaustive' or scan_type == 'confocal':
        if len(laser_lookats) > 1:
            laser_lookats = tqdm(
                laser_lookats, desc='Merging partial results...',
                file=TQDMLogRedirect(), ascii=True, total=len(laser_lookats))
        try:
            for laser_lookat_ix, laser_lookat_iy, _, __ in laser_lookats:
                x = laser_lookat_ix
                y = laser_lookat_iy
                hdr_path, _ = mitsuba_backend.partial_laser_path(
                    partial_results_dir,
                    experiment_name,
                    laser_lookat_ix, laser_lookat_iy)
                if scan_type == 'confocal':
                    capture_data.H[:, x:x+1, y:y+1, ...] = \
                        mitsuba_backend.read_transient_image(hdr_path)
                elif scan_type == 'exhaustive':
                    capture_data.H[:, x, y, ...] = \
                        mitsuba_backend.read_transient_image(hdr_path)
                else:
                    raise AssertionError
        except Exception as exc:
            raise RenderException from exc
    else:
        raise AssertionError(
            'Invalid scan_type, must be one of {single|exhaustive|confocal}')

    return capture_data


def _main_render(config_path, args,
                 mitsuba_backend, scene_config,
                 root_dir, partial_results_dir, steady_dir, log_dir,
                 progress_file, num_retries=0):
    """ General initialization """

    steady_scene_xml, ground_truth_scene_xml, nlos_scene_xml = \
        __write_scene_xmls(args, mitsuba_backend, scene_config, root_dir)
    experiment_name = scene_config['name']
    mitsuba_variant = scene_config['mitsuba_variant']

    """ Steady state + ground truth """

    if args.do_steady_renders:
        def render_steady(render_name, sensor_index):
            log(LogLevel.INFO,
                f'{render_name} for {experiment_name} steady render...')
            hdr_ext = mitsuba_backend.get_hdr_extension()
            hdr_path = os.path.join(partial_results_dir,
                                    f'{experiment_name}_{render_name}.{hdr_ext}')
            ldr_path = os.path.join(steady_dir,
                                    f'{experiment_name}_{render_name}.png')
            log_path = os.path.join(
                log_dir, f'{experiment_name}_{render_name}.log')

            __run_mitsuba(args, log_path, mitsuba_backend, mitsuba_variant, steady_scene_xml, hdr_path, dict(),
                          experiment_name, render_name, sensor_index, check_done=lambda: os.path.exists(ldr_path))

            if not args.dry_run:
                mitsuba_backend.convert_hdr_to_ldr(hdr_path, ldr_path)

        render_steady('back_view', 0)
        render_steady('side_view', 1)

    gt_render_name = 'ground_truth'
    gt_ext = mitsuba_backend.get_hdr_extension()
    gt_path = os.path.join(partial_results_dir,
                           f'{experiment_name}_{gt_render_name}.{gt_ext}')
    if args.do_ground_truth_renders:
        log(LogLevel.INFO, f'{gt_render_name} for {experiment_name}...')
        log_path = os.path.join(
            log_dir, f'{experiment_name}_{gt_render_name}.log')

        __run_mitsuba(args, log_path, mitsuba_backend, mitsuba_variant, ground_truth_scene_xml, gt_path, dict(),
                      experiment_name, gt_render_name, 0, check_done=lambda: os.path.exists(gt_path))

    """ NLOS renders """

    scan_type, laser_lookats, capture_data = \
        __write_metadata_and_get_laser_lookats(args, scene_config)

    pbar = tqdm(
        enumerate(laser_lookats), desc=f'Rendering {experiment_name} ({scan_type})...',
        file=TQDMLogRedirect(), ascii=True, total=len(laser_lookats))
    for i, (laser_lookat_ix, laser_lookat_iy, laser_lookat_px, laser_lookat_py) in pbar:
        try:
            hdr_path, is_dir = mitsuba_backend.partial_laser_path(
                partial_results_dir, experiment_name, laser_lookat_ix, laser_lookat_iy)
            if is_dir and not os.path.exists(hdr_path):
                os.mkdir(hdr_path)
        except OSError as exc:
            raise AssertionError(f'Invalid permissions: {exc}') from exc
        defines = {
            'laser_lookat_x': laser_lookat_px,
            'laser_lookat_y': laser_lookat_py,
        }
        log_path = os.path.join(
            log_dir,
            f'{experiment_name}_L[{laser_lookat_ix}][{laser_lookat_iy}].log')
        render_name = f'Laser {i + 1} of {len(laser_lookats)}'

        __run_mitsuba(args, log_path, mitsuba_backend, mitsuba_variant, nlos_scene_xml, hdr_path, defines,
                      experiment_name, render_name, 0, check_done=lambda: os.path.exists(hdr_path))

        if scan_type == 'exhaustive' and i == 0:
            size_bytes = os.path.getsize(hdr_path)
            final_size_gb = size_bytes * len(laser_lookats) / 2**30
            pbar.set_description(
                f'Rendering {experiment_name} ({scan_type}, estimated size: {final_size_gb:.2f} GB)...')

    if args.dry_run:
        return

    """ Generate final HDF5 file"""

    log(LogLevel.INFO, 'Reading partial results and generating HDF5 file...')
    try:
        capture_data = __merge_gt_results(
            args, mitsuba_backend, capture_data, gt_path)
        capture_data = __merge_nlos_results(
            args, mitsuba_backend, capture_data, partial_results_dir, experiment_name, scan_type, laser_lookats)
    except RenderException:
        if num_retries >= 10:
            raise AssertionError(
                f'Failed to read partial results after {num_retries} retries')
        # TODO(diego): Mitsuba sometimes fails to write some images,
        # it seems like some sort of race condition
        # If there is a partial result missing, just re-launch for now
        mitsuba_backend.remove_transient_image(hdr_path)
        log(LogLevel.INFO,
            f'We missed some partial results (iteration {i} failed because: {exc}), re-launching...')
        return _main_render(config_path, args, num_retries=num_retries + 1)

    hdf5_path = os.path.join(root_dir, f'{experiment_name}.hdf5')
    tal.io.write_capture(hdf5_path, capture_data,
                         file_format=FileFormat.HDF5_TAL)
    log(LogLevel.INFO, f'Stored result in {hdf5_path}')

    if not args.keep_partial_results:
        log(LogLevel.INFO,
            f'Cleaning partial results in {partial_results_dir}...')
        shutil.rmtree(partial_results_dir, ignore_errors=True)
        log(LogLevel.INFO, f'All clean.')

    os.remove(progress_file)

    return hdf5_path


def render_nlos_scene(config_path, args):
    mitsuba_backend, scene_config, config_path = \
        _read_config_and_init_mitsuba_variant(config_path, args)

    root_dir, partial_results_dir, steady_dir, log_dir, progress_file = \
        _check_progress_and_create_folders(config_path, args)

    try:
        return _main_render(config_path, args,
                            mitsuba_backend, scene_config,
                            root_dir, partial_results_dir, steady_dir, log_dir,
                            progress_file)
    except KeyboardInterrupt:
        delete = None
        while delete is None:
            try:
                log(LogLevel.PROMPT,
                    f'Render cancelled. '
                    f'Delete the directory {root_dir}? (y/n): ', end='')
                answer = input()
                if answer.lower() == 'y':
                    delete = True
                elif answer.lower() == 'n':
                    delete = False
            except KeyboardInterrupt:
                pass
        if delete:
            shutil.rmtree(root_dir, ignore_errors=True)
            os.remove(progress_file)
