import os
import sys
import shutil
import yaml
import tal
from tal.io.capture_data import NLOSCaptureData
from tal.enums import FileFormat, GridFormat, HFormat
from tal.config import local_file_path
from tal.log import log, LogLevel, TQDMLogRedirect
import datetime
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing.queues import Queue
from functools import partial

from tal.render.util import import_mitsuba_backend


def render_nlos_scene(config_path, args, num_retries=0):
    config_path = os.path.abspath(config_path)

    assert os.path.exists(config_path), \
        f'{config_path} does not exist'

    path, name = os.path.split(config_path)
    candidate_filename = os.path.join(path, name, f'{name}.yaml')
    if os.path.isdir(config_path) and os.path.isfile(candidate_filename):
        config_path = candidate_filename

    assert os.path.isfile(config_path), \
        f'{config_path} is not a TAL config file'

    config_dir, config_filename = os.path.split(config_path)

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

    steady_xml, nlos_xml = mitsuba_backend.get_scene_xml(
        scene_config, random_seed=args.seed, quiet=args.quiet)

    try:
        in_progress = False
        progress_file = os.path.join(config_dir, 'IN_PROGRESS')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_folder = f.read()
            if os.path.exists(os.path.join(config_dir, progress_folder)) and not args.quiet:
                in_progress = True
            else:
                log(LogLevel.INFO, 'The IN_PROGRESS file is stale, removing it...')
                os.remove(progress_file)
        if in_progress and not args.quiet:
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

    try:
        steady_scene_xml = os.path.join(root_dir, 'steady_scene.xml')
        with open(steady_scene_xml, 'w') as f:
            f.write(steady_xml)

        nlos_scene_xml = os.path.join(root_dir, 'nlos_scene.xml')
        with open(nlos_scene_xml, 'w') as f:
            f.write(nlos_xml)

        laser_lookats = []
        name = scene_config['name']
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
        assert 'displacement_x' not in relay_wall and \
            'displacement_y' not in relay_wall and \
            'displacement_z' not in relay_wall and \
            'rot_degrees_x' not in relay_wall and \
            'rot_degrees_y' not in relay_wall and \
            'rot_degrees_z' not in relay_wall, \
            'Relay wall displacement/rotation is NYI'

        def get_grid_xyz(nx, ny, relay_wall_scale):
            p = relay_wall_scale
            xg = np.stack((np.linspace(-p, p, num=2*nx + 1)[1::2],)*ny, axis=1)
            yg = np.stack((np.linspace(-p, p, num=2*ny + 1)[1::2],)*nx, axis=0)
            assert xg.shape[0] == yg.shape[0] == nx and xg.shape[1] == yg.shape[1] == ny, \
                'Incorrect shapes'
            return np.stack([xg, yg, np.zeros((nx, ny))], axis=-1).astype(np.float32)

        def expand(vec, x, y):
            assert len(vec) == 3
            return vec.reshape(1, 1, 3).repeat(x, axis=0).repeat(y, axis=1)

        if scan_type == 'single':
            laser_lookat_x = \
                scene_config['laser_lookat_x'] or sensor_width / 2
            laser_lookat_y = \
                scene_config['laser_lookat_y'] or sensor_height / 2
            laser_lookats.append((laser_lookat_x, laser_lookat_y))
        elif scan_type == 'exhaustive' or scan_type == 'confocal':
            assert not (scan_type == 'confocal' and
                        (laser_width != sensor_width or
                            laser_height != sensor_height)), \
                'If using scan_type=confocal, sensor_{width|height} must match laser_{width|height}'
            for y in range(laser_height):
                for x in range(laser_width):
                    laser_lookat_x = (x + 0.5) * sensor_width / laser_width
                    laser_lookat_y = (y + 0.5) * sensor_height / laser_height
                    laser_lookats.append((laser_lookat_x, laser_lookat_y))
        else:
            raise AssertionError(
                'Invalid scan_type, must be one of {single|exhaustive|confocal}')

        # TODO(diego): rotate + translate (asssumes no rot/trans)
        # or use a more generalist approach that does not need to be rectangular
        sensor_grid_xyz = get_grid_xyz(
            sensor_width, sensor_height, relay_wall['scale'])
        if scan_type == 'single':
            px = relay_wall['scale'] * \
                ((laser_lookat_x / sensor_width) * 2 - 1)
            py = relay_wall['scale'] * \
                ((laser_lookat_y / sensor_height) * 2 - 1)
            laser_grid_xyz = np.array([[
                [px, py, 0],
            ]], dtype=np.float32)
        else:
            laser_grid_xyz = get_grid_xyz(
                laser_width, laser_height, relay_wall['scale'])
        # TODO(diego): rotate [0, 0, 1] by rot_degrees_x (assmes RW is a plane)
        # or use a more generalist approach
        sensor_grid_normals = expand(
            np.array([0, 0, 1]), sensor_width, sensor_height)
        laser_grid_normals = expand(
            np.array([0, 0, 1]), laser_width, laser_height)

        experiment_name = scene_config['name']

        class StdoutQueue(Queue):
            def __init__(self, *args, **kwargs):
                ctx = multiprocessing.get_context()
                super(StdoutQueue, self).__init__(*args, **kwargs, ctx=ctx)

            def write(self, msg):
                self.put(msg)

        if args.do_steady_renders:
            def render_steady(render_name, sensor_index):
                if not args.quiet:
                    log(LogLevel.INFO,
                        f'{render_name} for {experiment_name} steady render...')
                hdr_ext = mitsuba_backend.get_hdr_extension()
                hdr_path = os.path.join(partial_results_dir,
                                        f'{experiment_name}_{render_name}.{hdr_ext}')
                ldr_path = os.path.join(steady_dir,
                                        f'{experiment_name}_{render_name}.png')
                if os.path.exists(ldr_path) and not args.quiet:
                    pass  # skip
                else:
                    logfile = None
                    if args.do_logging and not args.dry_run:
                        logfile = open(os.path.join(
                            log_dir, f'{experiment_name}_{render_name}.log'), 'w')
                    # NOTE: something here has a memory leak (probably Mitsuba-related)
                    # We run Mitsuba in a separate process to ensure that the leaks do not add up
                    # as they can fill your RAM in exhaustive scans
                    queue = StdoutQueue()
                    run_mitsuba_f = partial(mitsuba_backend.run_mitsuba, steady_scene_xml, hdr_path, dict(),
                                            render_name, logfile, args, sensor_index, queue)
                    if os.name == 'nt':
                        # NOTE: Windows does not support multiprocessing
                        run_mitsuba_f()
                    else:
                        process = multiprocessing.Process(target=run_mitsuba_f)
                        try:
                            process.start()
                            process.join()
                        except KeyboardInterrupt:
                            process.terminate()
                            raise KeyboardInterrupt
                    if args.do_logging and not args.dry_run:
                        while not queue.empty():
                            e = queue.get()
                            if isinstance(e, Exception):
                                raise e
                            else:
                                logfile.write(e)
                        queue.close()
                        logfile.close()
                    if not args.dry_run:
                        mitsuba_backend.convert_hdr_to_ldr(hdr_path, ldr_path)

            render_steady('back_view', 0)
            render_steady('side_view', 1)

        pbar = tqdm(
            enumerate(laser_lookats), desc=f'Rendering {experiment_name} ({scan_type})...',
            file=TQDMLogRedirect(), ascii=True, total=len(laser_lookats))
        for i, (laser_lookat_x, laser_lookat_y) in pbar:
            try:
                hdr_path, is_dir = mitsuba_backend.partial_laser_path(
                    partial_results_dir, experiment_name, laser_lookat_x, laser_lookat_y)
                if os.path.exists(hdr_path) and not args.quiet:
                    continue  # skip
                if is_dir:
                    os.mkdir(hdr_path)
            except OSError as exc:
                raise AssertionError(f'Invalid permissions: {exc}') from exc
            defines = {
                'laser_lookat_x': laser_lookat_x,
                'laser_lookat_y': laser_lookat_y,
            }
            logfile = None
            if args.do_logging and not args.dry_run:
                logfile = open(os.path.join(
                    log_dir,
                    f'{experiment_name}_L[{laser_lookat_x}][{laser_lookat_y}].log'), 'w')
            queue = StdoutQueue()
            render_name = f'Laser {i + 1} of {len(laser_lookats)}'
            # NOTE: something here has a memory leak (probably Mitsuba-related)
            # We run Mitsuba in a separate process to ensure that the leaks do not add up
            # as they can fill your RAM in exhaustive scans
            run_mitsuba_f = partial(mitsuba_backend.run_mitsuba, nlos_scene_xml, hdr_path, defines,
                                    render_name, sys.stdout, args, queue=queue)
            if os.name == 'nt':
                # NOTE: Windows does not support multiprocessing
                run_mitsuba_f()
            else:
                process = multiprocessing.Process(target=run_mitsuba_f)
                try:
                    process.start()
                    process.join()
                except KeyboardInterrupt:
                    process.terminate()
                    raise KeyboardInterrupt
            if scan_type == 'exhaustive' and i == 0:
                size_bytes = os.path.getsize(hdr_path)
                final_size_gb = size_bytes * len(laser_lookats) / 2**30
                pbar.set_description(
                    f'Rendering {experiment_name} ({scan_type}, estimated size: {final_size_gb:.2f} GB)...')
            if args.do_logging and not args.dry_run:
                while not queue.empty():
                    e = queue.get()
                    if isinstance(e, Exception):
                        raise e
                    else:
                        logfile.write(e)
                queue.close()
                logfile.close()

        if args.dry_run:
            return

        if not args.quiet:
            log(LogLevel.INFO, 'Merging partial results...')

        capture_data = NLOSCaptureData()
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
        # NOTE(diego): we do not store volume information for now
        # capture_data.volume_format = VolumeFormat.X_Y_Z_3
        capture_data.delta_t = scene_config['bin_width_opl']
        capture_data.t_start = scene_config['start_opl']
        capture_data.t_accounts_first_and_last_bounces = \
            scene_config['account_first_and_last_bounces']
        capture_data.scene_info = {
            'tal_version': tal.__version__,
            'config': scene_config,
            'args': vars(args),
        }
        if scan_type == 'single':
            hdr_path, _ = mitsuba_backend.partial_laser_path(
                partial_results_dir,
                experiment_name,
                *laser_lookats[0])
            capture_data.H = mitsuba_backend.read_transient_image(hdr_path)
            capture_data.H_format = HFormat.T_Sx_Sy
        elif scan_type == 'exhaustive' or scan_type == 'confocal':
            if scan_type == 'exhaustive':
                capture_data.H = np.empty(
                    (num_bins, laser_width, laser_height,
                     sensor_width, sensor_height),
                    dtype=np.float32)
                capture_data.H_format = HFormat.T_Lx_Ly_Sx_Sy
            elif scan_type == 'confocal':
                capture_data.H = np.empty(
                    (num_bins, laser_width, laser_height),
                    dtype=np.float32)
                capture_data.H_format = HFormat.T_Sx_Sy
            else:
                raise AssertionError

            e_laser_lookats = enumerate(laser_lookats)
            if not args.quiet and len(laser_lookats) > 1:
                e_laser_lookats = tqdm(
                    e_laser_lookats, desc='Merging partial results...',
                    file=TQDMLogRedirect(), ascii=True, total=len(laser_lookats))
            try:
                for i, (laser_lookat_x, laser_lookat_y) in e_laser_lookats:
                    x = i % laser_width
                    y = i // laser_width
                    hdr_path, _ = mitsuba_backend.partial_laser_path(
                        partial_results_dir,
                        experiment_name,
                        laser_lookat_x, laser_lookat_y)
                    if scan_type == 'confocal':
                        capture_data.H[:, x:x+1, y:y+1, ...] = \
                            mitsuba_backend.read_transient_image(hdr_path)
                    elif scan_type == 'exhaustive':
                        capture_data.H[:, x, y, ...] = \
                            mitsuba_backend.read_transient_image(hdr_path)
                    else:
                        raise AssertionError
            except Exception as exc:
                if num_retries >= 10:
                    raise AssertionError(
                        f'Failed to read partial results after {num_retries} retries')
                mitsuba_backend.remove_transient_image(hdr_path)
                # TODO Mitsuba sometimes fails to write some images,
                # it seems like some sort of race condition
                # If there is a partial result missing, just re-launch for now
                log(LogLevel.INFO,
                    f'We missed some partial results (iteration {i} failed because: {exc}), re-launching...')
                return render_nlos_scene(config_path, args, num_retries=num_retries + 1)
        else:
            raise AssertionError(
                'Invalid scan_type, must be one of {single|exhaustive|confocal}')

        hdf5_path = os.path.join(root_dir, f'{experiment_name}.hdf5')
        tal.io.write_capture(capture_data, hdf5_path,
                             file_format=FileFormat.HDF5_TAL)

        if not args.quiet:
            log(LogLevel.INFO, f'Stored result in {hdf5_path}')

        # remove IN_PROGRESS file
        os.remove(progress_file)

        if args.keep_partial_results:
            return

        if not args.quiet:
            log(LogLevel.INFO,
                f'Cleaning partial results in {partial_results_dir}...')

        shutil.rmtree(partial_results_dir)

        if not args.quiet:
            log(LogLevel.INFO, f'All clean.')
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
            shutil.rmtree(root_dir)
            os.remove(progress_file)
