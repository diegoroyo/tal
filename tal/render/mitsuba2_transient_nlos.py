from tal.log import log, LogLevel, TQDMLogRedirect

# pyright: reportMissingImports=false


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
            log(LogLevel.ERROR,
                f'setpath.sh cannot be found in {setpath_location}.', end='\n\n')
    return setpath_location


try:
    import mitsuba
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


def get_name():
    return 'mitsuba2-transient-nlos'


def get_scene_version():
    return '2.2.1'


def get_long_version():
    return '2.2.1'


def get_default_variant():
    return 'scalar_rgb'


def set_variant(s):
    import mitsuba
    if not s.startswith('scalar_'):
        raise AssertionError(
            f'Variant {s} is not supported. It must start with "scalar_"')
    mitsuba.set_variant(s)


def get_hdr_extension():
    return 'exr'


def convert_hdr_to_ldr(hdr_path, ldr_path):
    from tal.util import write_img, tonemap_ldr
    image = read_mitsuba_bitmap(hdr_path)
    image = tonemap_ldr(image)
    write_img(ldr_path, image)


def partial_laser_path(partial_results_dir, experiment_name, lx, ly):
    import os
    return os.path.join(partial_results_dir, f'{experiment_name}_L[{lx}][{ly}]'.replace('.', '_')), True


def read_transient_image(path):
    return _read_mitsuba_streakbitmap(path)


def remove_transient_image(path):
    log(LogLevel.WARNING, f'Skipping the removal of {path}')
    return


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


def get_scene_xml(config, random_seed=0):
    import os
    import tal
    from tal.util import fdent

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

    integrator_ground_truth = fdent(f'''\
        <integrator type="aov">
            <string name="aovs" value="pp:position, nn:geo_normal"/>
        </integrator>''')

    integrator_nlos = fdent(f'''\
        <integrator type="transientpath">
            <integer name="block_size" value="1"/>
            <integer name="max_depth" value="{v('integrator_max_depth')}"/>
            <integer name="filter_depth" value="{v('integrator_filter_depth')}"/>
            <boolean name="discard_direct_paths" value="{v('integrator_discard_direct_paths')}"/>
            <boolean name="nlos_laser_sampling" value="{v('integrator_nlos_laser_sampling')}"/>
            <boolean name="nlos_hidden_geometry_sampling" value="{v('integrator_nlos_hidden_geometry_sampling')}"/>
            <boolean name="nlos_hidden_geometry_sampling_do_rroulette" value="{v('integrator_nlos_hidden_geometry_sampling_do_rroulette')}"/>
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
                <integer name="seed" value="{random_seed}"/>
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
    if v('scan_type') == 'confocal':
        confocal_capture = 'true'
    elif v('scan_type') != 'single' and v('scan_type') != 'exhaustive':
        raise AssertionError(
            'scan_type should be one of {single|confocal|exhaustive}')
    sensor_nlos = fdent(f'''\
        <sensor type="nloscapturemeter">
            <sampler type="independent">
                <integer name="sample_count" value="{v('sample_count')}"/>
                <integer name="seed" value="{random_seed}"/>
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
                <string name="pixel_format" value="rgb"/>
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
    shapes_ground_truth = []
    shapes_nlos = []
    for gdict in v('geometry'):
        def g(key):
            return s(gdict.get(key, None))

        name = g('name')
        is_relay_wall = name == relay_wall_name
        if is_relay_wall and g('mesh')['type'] != 'rectangle':
            log(LogLevel.WARNING, 'Relay wall does not work well with meshes that are '
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
            shape_material = shape_material.replace(
                f'${key}', g('material')[key])

        shape_transform = fdent(f'''\
            <transform name="to_world">
                <scale x="{g('scale') or 1.0}" y="{g('scale') or 1.0}" z="{g('scale') or 1.0}"/>
                <rotate x="1" angle="{g('rot_degrees_x') or 0.0}"/>
                <rotate y="1" angle="{g('rot_degrees_y') or 0.0}"/>
                <rotate z="1" angle="{g('rot_degrees_z') or 0.0}"/>
                <translate x="{g('displacement_x') or 0.0}" y="{g('displacement_y') or 0.0}" z="{g('displacement_z') or 0.0}"/>
            </transform>''')

        shape_contents_steady = f'{shape_material}\n{shape_transform}'
        shape_contents_ground_truth = shape_contents_steady
        newline = '\n'
        shape_contents_nlos = f'{shape_material}\n{shape_transform}{f"{newline}{sensor_nlos}" if is_relay_wall else ""}'

        if is_relay_wall:
            # Transform on the ortographic camera depends on relay wall
            sensor_ground_truth_transform = fdent(f'''\
                <transform name="to_world">
                    <scale x="{g('scale_x') or 1.0}" y="{g('scale_y') or 1.0}"/>
                    <rotate x="1" angle="{g('rot_degrees_x') or 0.0}"/>
                    <rotate y="1" angle="{g('rot_degrees_y') or 0.0}"/>
                    <rotate z="1" angle="{g('rot_degrees_z') or 0.0}"/>
                    <translate x="{g('displacement_x') or 0.0}"
                               y="{g('displacement_y') or 0.0}"
                               z="{g('displacement_z') or 0.0}"/>
                </transform>''')

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

            shapes_steady.append(
                shapify(shape_contents_steady, filename))
            shapes_ground_truth.append(
                shapify(shape_contents_ground_truth, filename))
            shapes_nlos.append(
                shapify(shape_contents_nlos, filename))
        elif g('mesh')['type'] == 'rectangle' or g('mesh')['type'] == 'sphere':
            def shapify(content):
                return fdent('''\
                {shape_name}
                <shape type="{shape_type}">
                    {content}
                </shape>''', shape_name=shape_name, content=content,
                             shape_type=g('mesh')['type'])

            shapes_steady.append(
                shapify(shape_contents_steady, filename))
            shapes_ground_truth.append(
                shapify(shape_contents_ground_truth, filename))
            shapes_nlos.append(
                shapify(shape_contents_nlos, filename))
        else:
            raise AssertionError(
                f'Shape not yet supported: {g("mesh")["type"]}')

    shapes_steady = '\n\n'.join(shapes_steady)
    shapes_ground_truth = '\n\n'.join(shapes_ground_truth)
    shapes_nlos = '\n\n'.join(shapes_nlos)

    # Ground truth sensor declared here, after relay wall reading
    sensor_ground_truth = fdent('''\
        <!-- Ortographic camera (whose aperture corresponds to the relay wall) for depth and normals -->
        <sensor type="orthographic">
            {sensor_ground_truth_transform}
            <sampler type="independent">
                <integer name="sample_count" value="128"/>
                <integer name="seed" value="{random_seed}"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="1028"/>
                <integer name="height" value="1028"/>
                <rfilter type="box">
                    <!-- <float name="radius" value="0.5"/> -->
                </rfilter>
            </film>
        </sensor>''',
                                sensor_ground_truth_transform=sensor_ground_truth_transform,
                                random_seed=random_seed)

    file_steady = fdent('''\
        <!-- Auto-generated using TAL v{tal_version} -->
        <scene version="{mitsuba_version}">
            {integrator_steady}

            {dummy_lights_and_geometry_steady}

            {shapes_steady}

            {sensors_steady}
        </scene>''',
                        tal_version=tal.__version__,
                        mitsuba_version=get_scene_version(),
                        integrator_steady=integrator_steady,
                        dummy_lights_and_geometry_steady=dummy_lights_and_geometry_steady,
                        shapes_steady=shapes_steady,
                        sensors_steady=sensors_steady)

    file_ground_truth = fdent('''\
        <!-- Auto-generated using TAL v{tal_version} -->
        <scene version="{mitsuba_version}">
            {integrator_ground_truth}

            {shapes_ground_truth}

            {sensor_ground_truth}
        </scene>''',
                              tal_version=tal.__version__,
                              mitsuba_version=get_scene_version(),
                              integrator_ground_truth=integrator_ground_truth,
                              shapes_ground_truth=shapes_ground_truth,
                              sensor_ground_truth=sensor_ground_truth)

    file_nlos = fdent('''\
        <!-- Auto-generated using TAL v{tal_version} -->
        <scene version="{mitsuba_version}">
            {integrator_nlos}

            {shapes_nlos}
        </scene>''',
                      tal_version=tal.__version__,
                      mitsuba_version=get_scene_version(),
                      integrator_nlos=integrator_nlos,
                      shapes_nlos=shapes_nlos)

    return file_steady, file_ground_truth, file_nlos


def run_mitsuba(scene_xml_path, hdr_path, defines,
                experiment_name, logfile, args, sensor_index=0, queue=None):
    import re
    import time
    import subprocess
    import os
    from tqdm import tqdm
    # execute mitsuba command (sourcing setpath.sh before)
    num_threads = args.threads
    command = ['mitsuba',
               '-o', hdr_path,
               '-s', str(sensor_index),
               '-t', str(num_threads)]
    for key, value in defines.items():
        command += ['-D', f'{key}={value}']
    command += [scene_xml_path]

    nice = args.nice
    command = ['nice', '-n', str(nice), " ".join(command)]

    setpath_location = _get_setpath_location()

    if args.dry_run:
        # add extra commas to make it easier to copy-paste
        command = ['/bin/bash', '-c',
                   f'"source \\"{setpath_location}\\" && {" ".join(command)}"']
        log(LogLevel.PROMPT, ' '.join(command))
        return
    else:
        command = ['/bin/bash', '-c',
                   f'source "{setpath_location}" && {" ".join(command)}']

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
              file=TQDMLogRedirect(),
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
                    log(LogLevel.INFO, 'Auto-detected histogram: '
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
    if os.path.isdir(hdr_path):
        # assume transient render
        n_exr_frames = len(os.listdir(hdr_path))
        assert n_exr_frames > 0, 'No frames were rendered?'
    else:
        # assume steady state render
        assert os.path.isfile(hdr_path), 'No frames were rendered?'


def read_mitsuba_bitmap(path: str):
    from mitsuba.core import Bitmap
    import numpy as np
    return np.array(Bitmap(path), copy=False)


def _read_mitsuba_streakbitmap(path: str, exr_format='RGB'):
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

    # NOTE(diego): for now this assumes that the EXR that it reads
    # are in RGB format, and returns an image with 3 channels,
    # in the case of polarized light it can return something else
    if exr_format != 'RGB':
        raise NotImplementedError(
            'Formats different from RGB are not implemented')

    xtframes = glob.glob(os.path.join(
        glob.escape(path), f'frame_*.exr'))
    xtframes = sorted(xtframes,
                      key=lambda x: int(re.compile(r'\d+').findall(x)[-1]))
    number_of_xtframes = len(xtframes)
    first_img = read_mitsuba_bitmap(xtframes[0])
    streak_img = np.empty(
        (number_of_xtframes, *first_img.shape), dtype=first_img.dtype)
    with tqdm(desc=f'Reading {path}', total=number_of_xtframes, file=TQDMLogRedirect(), ascii=True) as pbar:
        for i_xtframe in range(number_of_xtframes):
            other = read_mitsuba_bitmap(xtframes[i_xtframe])
            streak_img[i_xtframe] = np.nan_to_num(other, nan=0.)
            pbar.update(1)

    # for now streak_img has dimensions (y, x, time, channels)
    assert streak_img.shape[-1] == 3, \
        f'Careful, streak_img has shape {streak_img.shape} (i.e. its probably not RGB as we assume, last dimension should be 3)'
    # and we want it as (time, x, y)
    return np.sum(np.transpose(streak_img), axis=0)
