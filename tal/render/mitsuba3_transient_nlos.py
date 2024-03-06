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
    return 'mitsuba3-transient-nlos'


def get_version():
    return '3.3.0'


def get_default_variant():
    return 'llvm_mono'


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
    image = _read_mitsuba_bitmap(hdr_path)
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


def get_scene_xml(config, random_seed=0, quiet=False):
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

    confocal_capture = 'false'
    if v('scan_type') == 'confocal':
        confocal_capture = 'true'
    elif v('scan_type') != 'single' and v('scan_type') != 'exhaustive':
        raise AssertionError(
            'scan_type should be one of {single|confocal|exhaustive}')
    sensor_nlos = fdent(f'''\
        <sensor type="nlos_capture_meter">
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
            <film type="transient_hdr_film">
                <integer name="width" value="{v('sensor_width')}"/>
                <integer name="height" value="{v('sensor_height')}"/>

                <integer name="temporal_bins" value="{v('num_bins')}"/>
                <!-- <boolean name="auto_detect_bins" value="{v('auto_detect_bins')}"/> -->
                <float name="bin_width_opl" value="{v('bin_width_opl')}"/>
                <float name="start_opl" value="{v('start_opl')}"/>
                <rfilter type="box">
                    <!-- <float name="radius" value="0.5"/> -->
                </rfilter>
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
        elif g('mesh')['type'] == 'rectangle' or g('mesh')['type'] == 'sphere':
            def shapify(content):
                return fdent('''\
                {shape_name}
                <shape type="{shape_type}">
                    {content}
                </shape>''', shape_name=shape_name, content=content,
                             shape_type=g('mesh')['type'])

            shapes_steady.append(shapify(shape_contents_steady))
            shapes_nlos.append(shapify(shape_contents_nlos))
        else:
            raise AssertionError(
                f'Shape not yet supported: {g("mesh")["type"]}')

    shapes_steady = '\n\n'.join(shapes_steady)
    shapes_nlos = '\n\n'.join(shapes_nlos)

    file_steady = fdent('''\
        <!-- Auto-generated using TAL v{tal_version} -->
        <scene version="{mitsuba_version}">
            {integrator_steady}

            {dummy_lights_and_geometry_steady}

            {shapes_steady}

            {sensors_steady}
        </scene>''',
                        tal_version=tal.__version__,
                        mitsuba_version=get_version(),
                        integrator_steady=integrator_steady,
                        dummy_lights_and_geometry_steady=dummy_lights_and_geometry_steady,
                        shapes_steady=shapes_steady,
                        sensors_steady=sensors_steady)

    file_nlos = fdent('''\
        <!-- Auto-generated using TAL v{tal_version} -->
        <scene version="{mitsuba_version}">
            {integrator_nlos}

            {shapes_nlos}
        </scene>''',
                      tal_version=tal.__version__,
                      mitsuba_version=get_version(),
                      integrator_nlos=integrator_nlos,
                      shapes_nlos=shapes_nlos)

    return file_steady, file_nlos


def run_mitsuba(scene_xml_path, hdr_path, defines,
                experiment_name, logfile, args, sensor_index=0, queue=None):
    try:
        import mitsuba as mi
        import mitransient as mitr
        import numpy as np
        from tqdm import tqdm
        from mitransient.integrators.common import TransientADIntegrator
        import sys
        import os

        sys.stdout = queue
        sys.stderr = queue
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

        scene = mi.load_file(scene_xml_path, **defines)
        integrator = scene.integrator()

        mi.Thread.set_thread_count(args.threads)

        # prepare
        if isinstance(integrator, TransientADIntegrator):
            integrator.prepare_transient(scene, sensor_index)

            progress_bar = None
            if not args.quiet:
                progress_bar = tqdm(total=100, desc=experiment_name,
                                    file=TQDMLogRedirect(),
                                    ascii=True, leave=False)

            def update_progress(p):
                if not args.quiet:
                    progress_bar.n = int(p * 100)
                    progress_bar.refresh()

            steady_image, transient_image = integrator.render(
                scene, progress_callback=update_progress)
            result = np.array(transient_image)
            if result.ndim == 2:
                nt, nc = result.shape
                result = result.reshape((nt, 1, 1, nc))
            result = np.moveaxis(result, 2, 0)
            # result has shape (nt, nx, ny, nchannels)
            if result.ndim == 4:
                # sum all channels
                result = np.sum(result, axis=-1)
            if not args.quiet:
                progress_bar.close()
            del steady_image, transient_image, progress_bar
        else:
            image = integrator.render(scene, sensor_index)
            result = np.array(image)

        np.save(hdr_path, result)
        del result, scene, integrator
    except Exception as e:
        queue.write(e)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def _read_mitsuba_bitmap(path: str):
    import numpy as np
    return np.load(path)


def _read_mitsuba_streakbitmap(path: str, exr_format='RGB'):
    import numpy as np
    return np.load(path)
