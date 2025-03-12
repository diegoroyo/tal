from tal.config import read_config, Config
from tal.log import log, LogLevel, TQDMLogRedirect

# pyright: reportMissingImports=false


def _get_setpath_location():
    from tal.config import ask_for_config, Config
    import os
    force_ask = False
    setpath_ok = False
    while not setpath_ok:
        setpath_location = os.path.join(
            ask_for_config(Config.MITSUBA3_TRANSIENT_NLOS_FOLDER,
                           force_ask=force_ask),
            'build', 'setpath.sh')
        if os.path.isfile(setpath_location):
            setpath_ok = True
        else:
            force_ask = True
            log(LogLevel.ERROR,
                f'setpath.sh cannot be found in {setpath_location}.', end='\n\n')
    return setpath_location


def add_mitsuba_to_path():
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
                sys.path.insert(0, directory)


custom_path = read_config().get(
    Config.MITSUBA3_TRANSIENT_NLOS_FOLDER.value[0], None)
if custom_path:
    add_mitsuba_to_path()


def get_name():
    return 'mitransient (mitsuba3-transient-nlos)'


def get_scene_version():
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    return mi.__version__


def get_long_version():
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    import mitransient as mitr
    import os

    def f(path):
        return os.path.split(os.path.abspath(path))[0]

    return f'mitransient v{mitr.__version__} ({f(mitr.__file__)})\n' \
        f'mitsuba v{mi.__version__} ({f(mi.__file__)})'


def get_default_variant():
    return 'llvm_ad_rgb'


def set_variant(s):
    import mitsuba
    if not (s.startswith('llvm_') or s.startswith('cuda_')):
        raise AssertionError(
            f'Variant {s} is not supported. It must start with "llvm_" or "cuda_"')
    mitsuba.set_variant(s)


def get_hdr_extension():
    return 'npy'


def convert_hdr_to_ldr(hdr_path, ldr_path):
    from tal.util import write_img, tonemap_ldr
    image = read_mitsuba_bitmap(hdr_path)
    image = tonemap_ldr(image)
    write_img(ldr_path, image)


def partial_laser_path(partial_results_dir, experiment_name, lx, ly):
    import os
    filename = os.path.join(
        partial_results_dir, f'{experiment_name}_L[{lx}][{ly}]'.replace('.', '_'))
    return f'{filename}.{get_hdr_extension()}', False


def read_transient_image(path):
    return _read_mitsuba_streakbitmap(path)


def remove_transient_image(path):
    import os
    if os.path.exists(path):
        os.remove(path)


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
        <integrator type="transient_nlos_path">
            <integer name="block_size" value="1"/>
            <integer name="max_depth" value="{v('integrator_max_depth')}"/>
            <integer name="filter_bounces" value="{v('integrator_filter_bounces')}"/>
            <boolean name="discard_direct_paths" value="{v('integrator_discard_direct_paths')}"/>
            <boolean name="nlos_laser_sampling" value="{v('integrator_nlos_laser_sampling')}"/>
            <boolean name="nlos_hidden_geometry_sampling" value="{v('integrator_nlos_hidden_geometry_sampling')}"/>
            <boolean name="nlos_hidden_geometry_sampling_do_rroulette" value="{v('integrator_nlos_hidden_geometry_sampling_do_rroulette')}"/>
            <boolean name="nlos_hidden_geometry_sampling_includes_relay_wall" value="{v('integrator_nlos_hidden_geometry_sampling_includes_relay_wall')}"/>
            <string name="temporal_filter" value="box"/>
        </integrator>''')

    if 'polarized' in v('mitsuba_variant'):
        # TODO(diego): mitsuba3 does not have a transient stokes integrator yet
        # https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/integrators/stokes.cpp
        # https://github.com/diegoroyo/mitsuba2-transient-nlos/blob/a270850d1f9b9d863e759f880048df665cd7d2a1/src/integrators/transientstokes.cpp
        # https://github.com/diegoroyo/mitsuba3-transient-nlos/blob/main/mitransient/integrators/transientnlospath.py
        # the end goal would be to generate a transient stokes plugin in python (like transient_nlos_path.py)
        # that implements the functionality of stokes.cpp similar to mitsuba2-transient-nlos's transientstokes.cpp
        raise NotImplementedError(
            'Polarized variants are not implemented in mitsuba3 yet')

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
            <point name="center" x="{v('laser_x')}" y="{v('laser_y')}" z="{v('laser_z')}"/>
            <float name="radius" value="0.05"/>
            <bsdf type="diffuse" id="red">
                <rgb name="reflectance" value="1.0, 0.0, 0.0"/>
            </bsdf>
        </shape>
        <shape type="sphere">
            <point name="center" x="{v('sensor_x')}" y="{v('sensor_y')}" z="{v('sensor_z')}"/>
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

    pixel_format = 'luminance' if 'mono' in v('mitsuba_variant') else 'rgb'

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
                            up="0, -1, 0"/>
            </transform>
            <sampler type="independent">
                <integer name="sample_count" value="512"/>
                <integer name="seed" value="{random_seed}"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="512"/>
                <integer name="height" value="512"/>
                <string name="pixel_format" value="{pixel_format}"/>
                <rfilter name="rfilter" type="gaussian"/>
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
                            up="0, -1, 0"/>
            </transform>
            <sampler type="independent">
                <integer name="sample_count" value="512"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="512"/>
                <integer name="height" value="512"/>
                <string name="pixel_format" value="{pixel_format}"/>
                <rfilter name="rfilter" type="gaussian"/>
            </film>
        </sensor>''')

    laser_x = v('laser_x')
    laser_y = v('laser_y')
    laser_z = v('laser_z')
    laser_fov = 0.2 if v('integrator_nlos_laser_sampling') else 2
    laser_emission_mode = v('laser_emission_mode') or 'rgb'
    laser_emission_value = v('laser_emission') or '1.0, 1.0, 1.0'
    if laser_emission_mode == 'mono':
        laser_emission = '<spectrum name="irradiance" value="{value}"/>'
    elif laser_emission_mode == 'rgb':
        laser_emission = '<rgb name="irradiance" value="{value}"/>'
    elif laser_emission_mode == 'spectrum':
        laser_emission = '<spectrum name="irradiance" value="{value}"/>'
    else:
        raise AssertionError(
            'laser_emission_mode should be one of {mono|rgb|spectrum}')
    laser_emission = laser_emission.format(value=laser_emission_value)

    laser_nlos = fdent('''\
        <emitter type="projector" id="laser">
            <transform name="to_world">
                <translate x="{laser_x}"
                           y="{laser_y}"
                           z="{laser_z}"/>
            </transform>
            {laser_emission}
            <float name="fov" value="{laser_fov}"/>
        </emitter>''',
                       laser_x=laser_x, laser_y=laser_y, laser_z=laser_z,
                       laser_emission=laser_emission, laser_fov=laser_fov)

    if v('scan_type') == 'confocal':
        film_width = 1
        film_height = 1
        confocal_config = fdent(f'''\
            <integer name="original_film_width" value="{v('sensor_width')}"/>
            <integer name="original_film_height" value="{v('sensor_height')}"/>
        ''')
    elif v('scan_type') == 'single' or v('scan_type') == 'exhaustive':
        film_width = v('sensor_width')
        film_height = v('sensor_height')
        confocal_config = ''
    else:
        raise AssertionError(
            'scan_type should be one of {single|confocal|exhaustive}')
    histogram_mode = v('histogram_mode') or 'time'
    if histogram_mode == 'time':
        film_xml = fdent('''\
            <film type="transient_hdr_film">
                <integer name="width" value="{film_width}"/>
                <integer name="height" value="{film_height}"/>

                <integer name="temporal_bins" value="{num_bins}"/>
                <!-- <boolean name="auto_detect_bins" value="{auto_detect_bins}"/> -->
                <float name="bin_width_opl" value="{bin_width_opl}"/>
                <float name="start_opl" value="{start_opl}"/>
                <rfilter type="box">
                    <!-- <float name="radius" value="0.5"/> -->
                </rfilter>
            </film>''',
                         film_width=film_width,
                         film_height=film_height,
                         num_bins=v('num_bins'),
                         auto_detect_bins=v('auto_detect_bins'),
                         bin_width_opl=v('bin_width_opl'),
                         start_opl=v('start_opl'))
    elif histogram_mode == 'frequency':
        assert v('wl_mean') is not None, 'wl_mean must be specified'
        assert v('wl_sigma') is not None, 'wl_sigma must be specified'
        film_xml = fdent('''\
            <film type="phasor_hdr_film">
                <integer name="width" value="{film_width}"/>
                <integer name="height" value="{film_height}"/>
                <!-- <boolean name="auto_detect_bins" value="{auto_detect_bins}"/> -->

                <float name="wl_mean" value="{wl_mean}"/>
                <float name="wl_sigma" value="{wl_sigma}"/>
                <integer name="temporal_bins" value="{num_bins}"/>
                <float name="bin_width_opl" value="{bin_width_opl}"/>
                <float name="start_opl" value="{start_opl}"/>
                <rfilter type="box">
                    <!-- <float name="radius" value="0.5"/> -->
                </rfilter>
            </film>''',
                         film_width=film_width,
                         film_height=film_height,
                         auto_detect_bins=v('auto_detect_bins'),
                         wl_mean=v('wl_mean'),
                         wl_sigma=v('wl_sigma'),
                         num_bins=v('num_bins'),
                         bin_width_opl=v('bin_width_opl'),
                         start_opl=v('start_opl'))
    else:
        raise AssertionError(
            'histogram_mode should be one of {time|frequency}')
    sensor_nlos = fdent('''\
        <sensor type="nlos_capture_meter">
            <sampler type="independent">
                <integer name="sample_count" value="{sample_count}"/>
                <integer name="seed" value="{random_seed}"/>
            </sampler>

            {confocal_config}
            <boolean name="account_first_and_last_bounces" value="{account_first_and_last_bounces}"/>
            <point name="sensor_origin" x="{sensor_x}"
                                        y="{sensor_y}"
                                        z="{sensor_z}"/>
            {film_xml}
        </sensor>''',
                        sample_count=v('sample_count'),
                        random_seed=random_seed,
                        confocal_config=confocal_config,
                        account_first_and_last_bounces=v(
                            'account_first_and_last_bounces'),
                        sensor_x=v('sensor_x'),
                        sensor_y=v('sensor_y'),
                        sensor_z=v('sensor_z'),
                        film_xml=film_xml)

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
                <scale x="{g('scale_x') or 1.0}" y="{g('scale_y') or 1.0}" z="{g('scale_z') or 1.0}"/>
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
                <shape type="{shape_type}"
                             {is_relay_wall}>
                    {content}
                </shape>''',
                             shape_name=shape_name,
                             shape_type=g('mesh')['type'],
                             is_relay_wall=' id="relay_wall"' if is_relay_wall else '',
                             content=content)

            shapes_steady.append(
                shapify(shape_contents_steady))
            shapes_ground_truth.append(
                shapify(shape_contents_ground_truth))
            shapes_nlos.append(
                shapify(shape_contents_nlos))
        else:
            raise AssertionError(
                f'Shape not yet supported: {g("mesh")["type"]}')

    shapes_steady = '\n\n'.join(shapes_steady)
    shapes_ground_truth = '\n\n'.join(shapes_ground_truth)
    shapes_nlos = '\n\n'.join(shapes_nlos)

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
                <string name="pixel_format" value="{pixel_format}"/>
                <rfilter type="box">
                    <!-- <float name="radius" value="0.5"/> -->
                </rfilter>
            </film>
        </sensor>''',
                                sensor_ground_truth_transform=sensor_ground_truth_transform,
                                random_seed=random_seed,
                                pixel_format=pixel_format)

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

            {laser_nlos}

            {shapes_nlos}
        </scene>''',
                      tal_version=tal.__version__,
                      mitsuba_version=get_scene_version(),
                      integrator_nlos=integrator_nlos,
                      laser_nlos=laser_nlos,
                      shapes_nlos=shapes_nlos)

    return file_steady, file_ground_truth, file_nlos


def run_mitsuba(scene_xml_path, hdr_path, defines,
                experiment_name, args, pipe_output, sensor_index=0):
    try:
        import mitsuba as mi
        import mitransient as mitr
        import numpy as np
        from tqdm import tqdm
        from mitransient.integrators.common import TransientADIntegrator
        from mitransient.films.phasor_hdr_film import PhasorHDRFilm
        import sys
        import os

        sys.stdout = pipe_output
        sys.stderr = pipe_output
        if os.name == 'posix':
            # Nice only available in posix systems
            os.nice(args.nice)

        if args.dry_run:
            return

        # set the CUDA_VISIBLE_DEVICES again
        # don't know if it's necessary but does not hurt
        if mi.variant().startswith('cuda'):
            assert len(args.gpus) > 0, \
                'You must specify at least one GPU to use CUDA. Use tal --gpu <id1> <id2> ...'
            gpu_ids = ','.join(map(str, args.gpus))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        focus_laser = False
        if 'laser_lookat_x' in defines and 'laser_lookat_y' in defines:
            focus_laser = True
            laser_lookat_x = defines.pop('laser_lookat_x')
            laser_lookat_y = defines.pop('laser_lookat_y')

        scene = mi.load_file(scene_xml_path, **defines)
        integrator = scene.integrator()
        mitr.utils.set_thread_count(args.threads)

        if focus_laser:
            def find_id(array, eid):
                same_id = list(filter(lambda e: e.id() == eid, array))
                assert len(same_id) == 1, f'Expected 1 element with id {eid}'
                return same_id[0]
            mitr.nlos.focus_emitter_at_relay_wall_pixel(
                mi.Point2f(laser_lookat_x, laser_lookat_y),
                find_id(scene.shapes(), 'relay_wall'),
                find_id(scene.emitters(), 'laser'))

        # prepare
        if isinstance(integrator, TransientADIntegrator):
            progress_bar = tqdm(total=100, desc=experiment_name,
                                file=TQDMLogRedirect(),
                                ascii=True, leave=False)

            def update_progress(p):
                progress_bar.n = int(p * 100)
                progress_bar.refresh()

            steady_image, transient_image = integrator.render(
                scene, sensor=sensor_index, progress_callback=update_progress)
            result = np.array(transient_image)
            if result.ndim == 2:
                nt, nc = result.shape
                result = result.reshape((nt, 1, 1, nc))
            result = np.moveaxis(result, 2, 0)
            result = np.swapaxes(result, 1, 2)
            # result has shape (nt, nx, ny, nchannels)
            if result.ndim == 4:
                # sum all channels
                result = np.sum(result, axis=-1)
            if isinstance(scene.sensors()[0].film(), PhasorHDRFilm):
                # convert (real, imag) to complex
                result = result[::2, ...] + 1j * result[1::2, ...]
            progress_bar.close()
            del steady_image, transient_image, progress_bar
        else:
            image = integrator.render(scene, sensor=sensor_index)
            result = np.array(image)

        np.save(hdr_path, result)

        del result, scene, integrator
    except Exception as e:
        import traceback
        print('/!\ Mitsuba process threw an exception:', e, file=sys.stderr)
        print('', file=sys.stderr)
        print('Traceback:', file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        pipe_output.close()


def read_mitsuba_bitmap(path: str):
    import numpy as np
    return np.load(path)


def _read_mitsuba_streakbitmap(path: str, exr_format='RGB'):
    import numpy as np
    return np.load(path)
