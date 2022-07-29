from functools import partial
from multiprocessing import dummy
import os
import shutil
import yaml
import tal
from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import FileFormat, GridFormat, HFormat
from tal.util import fdent, write_img, tonemap_ldr
from tal.render.mitsuba2_transient_nlos import (
    get_material_keys, get_materials,
    read_mitsuba_streakbitmap, read_mitsuba_bitmap,
    run_mitsuba, mitsuba_set_variant
)
from tal.config import local_file_path
import datetime
import numpy as np
from tqdm import tqdm


def get_scene_xml(config, quiet=False):
    def s(value):
        if isinstance(value, bool):
            return str(value).lower()
        else:
            return value

    def v(key):
        return s(config[key])

    # integrator
    integrator_steady = fdent(f'''\
        <integrator type="path"/>''')

    integrator_nlos = fdent(f'''\
        <integrator type="transientpath">
            <integer name="block_size" value="1"/>
            <integer name="max_depth" value="{v('integrator_max_depth')}"/>
            <integer name="filter_depth" value="{v('integrator_filter_depth')}"/>
            <boolean name="discard_direct_paths" value="{v('integrator_discard_direct_paths')}"/>
            <boolean name="nlos_laser_sampling" value="{v('integrator_nlos_laser_sampling')}"/>
            <boolean name="nlos_hidden_geometry_sampling" value="{v('integrator_nlos_hidden_geometry_sampling')}"/>
            <boolean name="nlos_hidden_geometry_sampling_do_mis" value="{v('integrator_nlos_hidden_geometry_sampling_do_mis')}"/>
            <boolean name="nlos_hidden_geometry_sampling_includes_relay_wall" value="{v('integrator_nlos_hidden_geometry_sampling_includes_relay_wall')}"/>
        </integrator>''')

    if 'polarized' in v('mitsuba_variant'):
        def add_stokes(s, itype):
            return fdent('''\
                <integrator type="{itype}">
                    {s}
                </integrator>''', itype=itype, s=s)
        integrator_steady = add_stokes(integrator_steady, 'stokes')
        integrator_nlos = add_stokes(integrator_nlos, 'transientstokes')

    # relay wall
    geometry_names = list(map(lambda g: g['name'], v('geometry')))
    assert len(geometry_names) == len(set(geometry_names)), \
        'One or more geometry names is duplicated'
    relay_wall_name = v('relay_wall')
    assert relay_wall_name in geometry_names, \
        f'Relay wall is set to {relay_wall_name}, but must be one of {geometry_names}'

    dummy_lights_and_geometry_steady = fdent(f'''\
        <!-- Colored spheres to mark the laser and sensor positions -->
        <shape type="sphere">
            <point name="center" x="{v('sensor_x')}" y="{v('sensor_y')}" z="{v('sensor_z')}"/>
            <float name="radius" value="0.05"/>
            <bsdf type="diffuse" id="red">
                <rgb name="reflectance" value="1.0, 0.0, 0.0"/>
            </bsdf>
        </shape>
        <shape type="sphere">
            <point name="center" x="{v('laser_x')}" y="{v('laser_y')}" z="{v('laser_z')}"/>
            <float name="radius" value="0.05"/>
            <bsdf type="diffuse" id="blue">
                <rgb name="reflectance" value="0.0, 0.0, 1.0"/>
            </bsdf>
        </shape>
        
        <!-- Illuminate all the scene -->
        <emitter type="point">
            <rgb name="intensity" value="1.0, 1.0, 1.0"/>
            <point name="position" x="30" y="0" z="30"/>
        </emitter>
        <emitter type="point">
            <rgb name="intensity" value="1.0, 1.0, 1.0"/>
            <point name="position" x="-30" y="0" z="30"/>
        </emitter>
        <emitter type="point">
            <rgb name="intensity" value="1.0, 1.0, 1.0"/>
            <point name="position" x="0" y="30" z="30"/>
        </emitter>
        <emitter type="point">
            <rgb name="intensity" value="1.0, 1.0, 1.0"/>
            <point name="position" x="0" y="-30" z="30"/>
        </emitter>''')

    sensors_steady = fdent(f'''\
        <!-- Sensor 0: back view -->
        <sensor type="perspective">
            <string name="fov_axis" value="smaller"/>
            <float name="near_clip" value="0.01"/>
            <float name="far_clip" value="1000"/>
            <float name="fov" value="30"/>
            <transform name="to_world">
                <lookat origin="0, 0, 5"
                        target="0, 0, 0"
                            up="0, 1, 0"/>
            </transform>
            <sampler type="independent">
                <integer name="sample_count" value="512"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="512"/>
                <integer name="height" value="512"/>
                <rfilter name="rfilter" type="gaussian"/>
                <boolean name="high_quality_edges" value="false"/>
            </film>
        </sensor>

        <!-- Sensor 1: side view -->
        <sensor type="perspective">
            <string name="fov_axis" value="smaller"/>
            <float name="near_clip" value="0.01"/>
            <float name="far_clip" value="1000"/>
            <float name="fov" value="45"/>
            <transform name="to_world">
                <lookat origin="5, 0, 1.5"
                        target="0, 0, 1.5"
                            up="0, 1, 0"/>
            </transform>
            <sampler type="independent">
                <integer name="sample_count" value="512"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="512"/>
                <integer name="height" value="512"/>
                <rfilter name="rfilter" type="gaussian"/>
                <boolean name="high_quality_edges" value="false"/>
            </film>
        </sensor>''')

    confocal_capture = 'false'
    if v('scan_type') == 'confocal' or v('scan_type') == 'exhaustive':
        confocal_capture = 'true'
    elif v('scan_type') != 'single':
        raise AssertionError(
            'scan_type should be one of {single|confocal|exhaustive}')
    sensor_nlos = fdent(f'''\
        <sensor type="nloscapturemeter">
            <sampler type="independent">
                <integer name="sample_count" value="{v('sample_count')}"/>
            </sampler>

            <emitter type="projector">
                <rgb name="irradiance" value="1.0, 1.0, 1.0"/>
                <float name="fov" value="{0.2 if v('integrator_nlos_laser_sampling') else 2}"/>
            </emitter>

            <boolean name="confocal" value="{confocal_capture}"/>
            <boolean name="account_first_and_last_bounces" value="{v('account_first_and_last_bounces')}"/>
            <point name="sensor_origin" x="{v('sensor_x')}" y="{v('sensor_y')}" z="{v('sensor_z')}"/>
            <point name="laser_origin" x="{v('laser_x')}" y="{v('laser_y')}" z="{v('laser_z')}"/>
            <point name="laser_lookat_pixel" x="$laser_lookat_x" y="$laser_lookat_y" z="0"/>
            <film type="streakhdrfilm" name="streakfilm">
                <integer name="width" value="{v('sensor_width')}"/>
                <integer name="height" value="{v('sensor_height')}"/>
                <string name="component_format" value="float32"/>

                <integer name="num_bins" value="{v('num_bins')}"/>
                <boolean name="auto_detect_bins" value="{v('auto_detect_bins')}"/>
                <float name="bin_width_opl" value="{v('bin_width_opl')}"/>
                <float name="start_opl" value="{v('start_opl')}"/>

                <rfilter name="rfilter" type="box"/>
                <!-- NOTE: tfilters are not implemented yet -->
                <!-- <rfilter name="tfilter" type="box"/>  -->
                <boolean name="high_quality_edges" value="false"/>
            </film>
        </sensor>''')

    materials = get_materials()
    shapes_steady = []
    shapes_nlos = []
    for gdict in v('geometry'):
        def g(key):
            return s(gdict.get(key, None))

        name = g('name')
        is_relay_wall = name == relay_wall_name
        if is_relay_wall and g('mesh')['type'] != 'rectangle' and not quiet:
            print('WARNING: Relay wall does not work well with meshes that are '
                  'not of type "rectangle" because of wrong UV mapping. '
                  'Please make sure that you know what you are doing')
        shape_name = f'<!-- {name}{" (RELAY WALL)" if is_relay_wall else ""} -->'
        description = g('description')
        if description is not None and len(description.strip()) > 0:
            shape_name = fdent('''\
                {shape_name}
                <!--
                    {description}
                -->''', shape_name=shape_name, description=description.strip())

        material_id = g('material')['id']
        assert material_id in materials.keys(), \
            f'Geometry {name} has material {material_id}, but it must be one of {", ".join(materials.keys())}'
        shape_material = materials[material_id]
        material_keys = get_material_keys(shape_material)
        for key in material_keys:
            assert key in g('material'), \
                f'Material {material_id} for geometry {name} must have a value for {key}'
            shape_material.replace(f'${key}', g('material')[key])

        shape_transform = fdent(f'''\
            <transform name="to_world">
                <scale x="{g('scale') or 1.0}" y="{g('scale') or 1.0}" z="{g('scale') or 1.0}"/>
                <rotate x="1" angle="{g('rot_degrees_x') or 0.0}"/>
                <rotate y="1" angle="{g('rot_degrees_y') or 0.0}"/>
                <rotate z="1" angle="{g('rot_degrees_z') or 0.0}"/>
                <translate x="{g('displacement_x') or 0.0}" y="{g('displacement_y') or 0.0}" z="{g('displacement_z') or 0.0}"/>
            </transform>''')

        shape_contents_steady = f'{shape_material}\n{shape_transform}'
        newline = '\n'
        shape_contents_nlos = f'{shape_material}\n{shape_transform}{f"{newline}{sensor_nlos}" if is_relay_wall else ""}'

        if g('mesh')['type'] == 'obj':
            assert 'filename' in g('mesh'), \
                f'Missing mesh filename for geometry "{name}". ' \
                f'It is required because its mesh type is set as OBJ.'
            filename = g('mesh')['filename']
            assert os.path.isfile(filename), \
                f'{filename} does not exist for geometry "{name}"'

            def shapify(content, filename):
                return fdent('''\
                {shape_name}
                <shape type="obj">
                    <string name="filename" value="{filename}"/>

                    {content}
                </shape>''', shape_name=shape_name, filename=filename, content=content)

            shapes_steady.append(shapify(shape_contents_steady, filename))
            shapes_nlos.append(shapify(shape_contents_nlos, filename))
        elif g('mesh')['type'] == 'rectangle':
            def shapify(content):
                return fdent('''\
                {shape_name}
                <shape type="rectangle">
                    {content}
                </shape>''', shape_name=shape_name, content=content)

            shapes_steady.append(shapify(shape_contents_steady))
            shapes_nlos.append(shapify(shape_contents_nlos))
        else:
            raise AssertionError(
                f'Shape not yet supported: {g("mesh")["type"]}')

    shapes_steady = '\n\n'.join(shapes_steady)
    shapes_nlos = '\n\n'.join(shapes_nlos)

    file_steady = fdent('''\
        <!-- Auto-generated using TAL v{version} -->
        <scene version="2.2.1">
            {integrator_steady}

            {dummy_lights_and_geometry_steady}

            {shapes_steady}

            {sensors_steady}
        </scene>''',
                        version=tal.__version__,
                        integrator_steady=integrator_steady,
                        dummy_lights_and_geometry_steady=dummy_lights_and_geometry_steady,
                        shapes_steady=shapes_steady,
                        sensors_steady=sensors_steady)

    file_nlos = fdent('''\
        <!-- Auto-generated using TAL v{version} -->
        <scene version="2.2.1">
            {integrator_nlos}

            {shapes_nlos}
        </scene>''',
                      version=tal.__version__,
                      integrator_nlos=integrator_nlos,
                      shapes_nlos=shapes_nlos)

    return file_steady, file_nlos


def render_nlos_scene(config_path, args):
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
        scene_defaults = yaml.safe_load(
            open(local_file_path('render/scene_defaults.yaml'), 'r'))
    except yaml.YAMLError as exc:
        raise AssertionError(
            f'Invalid YAML format in TAL config file: {exc}') from exc
    scene_config = {**scene_defaults, **scene_config}

    mitsuba_set_variant(scene_config['mitsuba_variant'])
    steady_xml, nlos_xml = get_scene_xml(scene_config, quiet=args.quiet)

    try:
        root_dir = os.path.join(
            config_dir, datetime.datetime.now().strftime(r'%Y%m%d-%H%M%S'))
        partial_results_dir = os.path.join(root_dir, 'partial')
        steady_dir = os.path.join(root_dir, 'steady')
        log_dir = os.path.join(root_dir, 'logs')
        os.mkdir(root_dir)
        os.mkdir(partial_results_dir)
        os.mkdir(steady_dir)
        os.mkdir(log_dir)
        shutil.copy(
            config_path,
            os.path.join(root_dir, f'{config_filename}.old'))
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
            xg = np.stack((np.linspace(-p, p, num=2*nx + 1)[1::2],)*ny, axis=0)
            yg = np.stack((np.linspace(-p, p, num=2*ny + 1)[1::2],)*nx, axis=1)
            assert xg.shape[0] == yg.shape[0] == ny and xg.shape[1] == yg.shape[1] == nx, \
                'Incorrect shapes'
            return np.stack([xg, yg, np.zeros((nx, ny))], axis=-1).astype(np.float32)

        def expand(vec, x, y):
            assert len(vec) == 3
            return vec.reshape(1, 1, 3).repeat(x, axis=0).repeat(y, axis=1)

        # FIXME(diego): rotate + translate (asssumes no rot/trans)
        # or use a more generalist approach that does not need to be rectangular
        sensor_grid_xyz = get_grid_xyz(
            sensor_width, sensor_height, relay_wall['scale'])
        laser_grid_xyz = get_grid_xyz(
            laser_width, laser_height, relay_wall['scale'])
        # FIXME(diego): rotate [0, 0, 1] by rot_degrees_x (assmes RW is a plane)
        # or use a more generalist approach
        sensor_grid_normals = expand(
            np.array([0, 0, 1]), sensor_width, sensor_height)
        laser_grid_normals = expand(
            np.array([0, 0, 1]), laser_width, laser_height)

        if scan_type == 'single':
            laser_width = 1.0
            laser_height = 1.0
            laser_lookat_x = \
                scene_config['laser_lookat_x'] or (sensor_width - 1) / 2
            laser_lookat_y = \
                scene_config['laser_lookat_y'] or (sensor_height - 1) / 2
            laser_lookats.append((laser_lookat_x, laser_lookat_y))
        elif scan_type == 'exhaustive' or scan_type == 'confocal':
            assert not (scan_type == 'confocal' and
                        (laser_width != sensor_width or
                            laser_height != sensor_height)), \
                'If using scan_type=confocal, sensor_{width|height} must match laser_{width|height}'
            for x in range(laser_width):
                for y in range(laser_height):
                    laser_lookats.append((x + 0.5, y + 0.5))
        else:
            raise AssertionError(
                'Invalid scan_type, must be one of {single|exhaustive|confocal}')

        experiment_name = scene_config['name']

        if args.do_steady_renders:
            def render_steady(render_name, sensor_index):
                if not args.quiet:
                    print(f'{render_name} for {experiment_name} steady render...')
                exr_path = os.path.join(partial_results_dir,
                                        f'{experiment_name}_{render_name}.exr')
                png_path = os.path.join(steady_dir,
                                        f'{experiment_name}_{render_name}.png')
                logfile = None
                if args.do_logging and not args.dry_run:
                    logfile = open(os.path.join(
                        log_dir, f'{experiment_name}_{render_name}.log'), 'w')
                run_mitsuba(steady_scene_xml, exr_path, dict(),
                            render_name, logfile, args, sensor_index)
                if args.do_logging and not args.dry_run:
                    logfile.close()
                if not args.dry_run:
                    image = read_mitsuba_bitmap(exr_path)
                    image = tonemap_ldr(image)
                    write_img(png_path, image)

            render_steady('back_view', 0)
            render_steady('front_view', 1)

        def partial_laser_dir(lx, ly):
            return os.path.join(partial_results_dir, f'{experiment_name}_L[{lx}][{ly}]'.replace('.', '_'))

        for i, (laser_lookat_x, laser_lookat_y) in tqdm(
                enumerate(laser_lookats), desc=f'Rendering {experiment_name} ({scan_type})...',
                ascii=True, total=len(laser_lookats)):
            try:
                exr_dir = partial_laser_dir(laser_lookat_x, laser_lookat_y)
                os.mkdir(exr_dir)
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
            run_mitsuba(nlos_scene_xml, exr_dir, defines,
                        f'Laser {i} of {len(laser_lookats)}', logfile, args)
            if args.do_logging and not args.dry_run:
                logfile.close()

        if args.dry_run:
            return

        if not args.quiet:
            print('Merging partial results...')

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
            'config': scene_config,
            'args': vars(args),
        }
        if scan_type == 'single':
            capture_data.H = read_mitsuba_streakbitmap(
                partial_laser_dir(*laser_lookats[0]))
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

            for x in range(laser_width):
                for y in range(laser_height):
                    capture_data.H[:, x, y, ...] = read_mitsuba_streakbitmap(
                        partial_laser_dir(x + 0.5, y + 0.5))
        else:
            raise AssertionError(
                'Invalid scan_type, must be one of {single|exhaustive|confocal}')

        hdf5_path = os.path.join(root_dir, f'{experiment_name}.hdf5')
        tal.io.write_capture(capture_data, hdf5_path,
                             file_format=FileFormat.HDF5_TAL)

        if not args.quiet:
            print(f'Stored result in {hdf5_path}')

        if args.keep_partial_results:
            return

        if not args.quiet:
            print(f'Cleaning partial results in {partial_results_dir}...')

        shutil.rmtree(partial_results_dir)

        if not args.quiet:
            print(f'All clean.')
    except KeyboardInterrupt:
        delete = None
        while delete is None:
            try:
                answer = input(f'Render cancelled. '
                               f'Delete the directory {root_dir}? (y/n): ')
                if answer.lower() == 'y':
                    delete = True
                elif answer.lower() == 'n':
                    delete = False
            except KeyboardInterrupt:
                pass
        if delete:
            shutil.rmtree(root_dir)
