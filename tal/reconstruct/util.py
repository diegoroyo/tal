from tal.enums import HFormat, GridFormat, VolumeFormat, CameraSystem
from tal.io.capture_data import NLOSCaptureData
from tal.log import log, LogLevel
from typing import Union
import numpy as np


def _infer_volume_format(volume_xyz, volume_format, do_log=True):
    # infer volume_format if it is unknown
    if volume_format == VolumeFormat.UNKNOWN:
        assert volume_xyz.shape[-1] == 3, \
            'Could not infer volume_format. Please specify it manually.'
        if volume_xyz.ndim == 2:
            if do_log:
                log(LogLevel.INFO,
                    'tal.reconstruct.utils: Assuming that volume_xyz is N_3')
            volume_format = VolumeFormat.N_3
        elif volume_xyz.ndim == 3:
            if do_log:
                log(LogLevel.INFO,
                    'tal.reconstruct.utils: Assuming that volume_xyz is X_Y_3')
            volume_format = VolumeFormat.X_Y_3
        elif volume_xyz.ndim == 4:
            if do_log:
                log(LogLevel.INFO,
                    'tal.reconstruct.utils: Assuming that volume_xyz is X_Y_Z_3')
            volume_format = VolumeFormat.X_Y_Z_3
        else:
            raise AssertionError(
                'Could not infer volume_format. Please specify it manually.')

    return volume_format


def convert_to_N_3(data: NLOSCaptureData,
                   volume_xyz: NLOSCaptureData.VolumeXYZType,
                   volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
                   projector_focus: Union[NLOSCaptureData.Array3,
                                          NLOSCaptureData.VolumeXYZType] = None,
                   try_optimize_convolutions: bool = False):
    """
        try_optimize_convolutions performs many checks:
        - H, {laser|sensor}_grid_xyz and volume_xyz have X, Y components (e.g. they are not N_3)
        - The laser_grid_xyz's slices are parallel to the projector_focus' slices
        - The sensor_grid_xyz's slices are parallel to the volume_xyz's slices
        - The points in the {laser|sensor}_grid_xyz's slices are sampled at the same rate
            as {projector_focus|volume_xyz}'s points
    """

    volume_format = _infer_volume_format(
        volume_xyz, volume_format, do_log=True)
    try:
        assert projector_focus is not None
        projector_focus = np.array(projector_focus)
        projector_focus_format = _infer_volume_format(
            projector_focus, VolumeFormat.UNKNOWN, do_log=False)
    except AssertionError:
        projector_focus_format = VolumeFormat.UNKNOWN

    # this variable is set to false during the conversion
    optimize_projector_convolutions = try_optimize_convolutions and \
        projector_focus_format in [VolumeFormat.X_Y_3, VolumeFormat.X_Y_Z_3]
    optimize_camera_convolutions = try_optimize_convolutions and \
        volume_format in [VolumeFormat.X_Y_3, VolumeFormat.X_Y_Z_3]

    is_laser_paired_to_sensor = data.is_laser_paired_to_sensor()

    if data.H_format == HFormat.T_Si:
        optimize_projector_convolutions = False
        optimize_camera_convolutions = False
        nt, ns = data.H.shape
        if is_laser_paired_to_sensor:
            nl = ns
        H = data.H.reshape(nt, 1, ns)
    elif data.H_format == HFormat.T_Sx_Sy:
        optimize_projector_convolutions = False
        nt, nsx, nsy = data.H.shape
        if is_laser_paired_to_sensor:
            nlx, nly = nsx, nsy
        H = data.H.reshape(nt, 1, nsx * nsy)
    elif data.H_format == HFormat.T_Li_Si:
        optimize_projector_convolutions = False
        optimize_camera_convolutions = False
        nt, nl, ns = data.H.shape
        H = data.H.reshape(nt, nl, ns)
    elif data.H_format == HFormat.T_Lx_Ly_Sx_Sy:
        nt, nlx, nly, nsx, nsy = data.H.shape
        H = data.H.reshape(nt, nlx * nly, nsx * nsy)
    else:
        raise AssertionError(f'H_format {data.H_format} not implemented')

    if data.laser_grid_format == GridFormat.N_3:
        optimize_projector_convolutions = False
        laser_grid_xyz = data.laser_grid_xyz
    elif data.laser_grid_format == GridFormat.X_Y_3:
        try:
            assert nlx == data.laser_grid_xyz.shape[0] and nly == data.laser_grid_xyz.shape[1], \
                'laser_grid_xyz.shape does not match with H.shape'
        except NameError:
            # nlx, nly not defined, OK
            nlx, nly, _ = data.laser_grid_xyz.shape
            pass
        laser_grid_xyz = data.laser_grid_xyz.reshape(nlx * nly, 3)
        optimize_projector_convolutions = optimize_projector_convolutions and nlx > 1 and nly > 1
    else:
        raise AssertionError(
            f'laser_grid_format {data.laser_grid_format} not implemented')

    if data.sensor_grid_format == GridFormat.N_3:
        optimize_camera_convolutions = False
        sensor_grid_xyz = data.sensor_grid_xyz
    elif data.sensor_grid_format == GridFormat.X_Y_3:
        try:
            assert nsx == data.sensor_grid_xyz.shape[0] and nsy == data.sensor_grid_xyz.shape[1], \
                'sensor_grid_xyz.shape does not match with H.shape'
        except NameError:
            # nsx, nsy not defined, OK
            nsx, nsy, _ = data.sensor_grid_xyz.shape
            pass
        sensor_grid_xyz = data.sensor_grid_xyz.reshape(nsx * nsy, 3)
        optimize_camera_convolutions = optimize_camera_convolutions and nsx > 1 and nsy > 1
    else:
        raise AssertionError(
            f'sensor_grid_format {data.sensor_grid_format} not implemented')

    if is_laser_paired_to_sensor:
        assert H.shape[1] == 1 and H.shape[2] == laser_grid_xyz.shape[0], \
            'H.shape does not match with laser_grid_xyz.shape.'
    else:
        assert H.shape[1] == laser_grid_xyz.shape[0], \
            'H.shape does not match with laser_grid_xyz.shape.'
    assert H.shape[2] == sensor_grid_xyz.shape[0], \
        'H.shape does not match with sensor_grid_xyz.shape.'

    try:
        assert projector_focus is not None
        assert projector_focus_format in [
            VolumeFormat.X_Y_3, VolumeFormat.X_Y_Z_3]
        npx, npy = projector_focus.shape[:2]
        assert npx > 1 and npy > 1

        # list of (Z, 3) normals of all Z positions in the plane
        if projector_focus_format == VolumeFormat.X_Y_3:
            z_index = Ellipsis
        else:
            z_index = 0
        p_a = projector_focus[0, 0, z_index, :]
        p_b = projector_focus[-1, 0, z_index, :]
        p_c = projector_focus[0, -1, z_index, :]
        p_n = np.cross(p_b - p_a, p_c - p_a).reshape((-1, 3))
        assert np.all(np.linalg.norm(p_n, axis=-1) > 1e-5)
        p_n /= np.linalg.norm(p_n, axis=-1, keepdims=True)
        p_dx = np.linalg.norm(
            projector_focus[1, 0, z_index, :] - projector_focus[0, 0, z_index, :])
        p_dy = np.linalg.norm(
            projector_focus[0, 1, z_index, :] - projector_focus[0, 0, z_index, :])
    except AssertionError:
        optimize_projector_convolutions = False

    assert volume_format.xyz_dim_is_last(), 'Unexpected volume_format'
    try:
        assert volume_format in [VolumeFormat.X_Y_3, VolumeFormat.X_Y_Z_3]
        nvx, nvy = volume_xyz.shape[:2]
        assert nvx > 1 and nvy > 1

        # list of (Z, 3) normals of all Z positions in the plane
        if volume_format == VolumeFormat.X_Y_3:
            z_index = Ellipsis
        else:
            z_index = 0
        v_a = volume_xyz[0, 0, z_index, :]
        v_b = volume_xyz[-1, 0, z_index, :]
        v_c = volume_xyz[0, -1, z_index, :]
        v_n = np.cross(v_b - v_a, v_c - v_a).reshape((-1, 3))
        assert np.all(np.linalg.norm(v_n, axis=-1) > 1e-5)
        v_n /= np.linalg.norm(v_n, axis=-1, keepdims=True)
        v_dx = np.linalg.norm(
            volume_xyz[1, 0, z_index, :] - volume_xyz[0, 0, z_index, :])
        v_dy = np.linalg.norm(
            volume_xyz[0, 1, z_index, :] - volume_xyz[0, 0, z_index, :])
    except AssertionError:
        optimize_camera_convolutions = False

    try:
        assert optimize_projector_convolutions
        assert data.laser_grid_format == GridFormat.X_Y_3
        laser_grid_xyz = laser_grid_xyz.reshape((nlx, nly, 3))
        l_a = laser_grid_xyz[0, 0, ..., :]
        l_b = laser_grid_xyz[-1, 0, ..., :]
        l_c = laser_grid_xyz[0, -1, ..., :]
        l_n = np.cross(l_b - l_a, l_c - l_a).reshape((1, 3))
        l_n /= np.linalg.norm(l_n, axis=-1, keepdims=True)

        l_dx = np.linalg.norm(
            laser_grid_xyz[1, 0, ..., :] - laser_grid_xyz[0, 0, ..., :])
        l_dy = np.linalg.norm(
            laser_grid_xyz[0, 1, ..., :] - laser_grid_xyz[0, 0, ..., :])

        dot_lp = np.sum(l_n * p_n, axis=-1)
        assert np.allclose(np.abs(dot_lp), 1)
        assert np.isclose(p_dx, l_dx) and np.isclose(p_dy, l_dy)
        log(LogLevel.INFO, 'tal.reconstruct.utils: Optimizing for projector convolutions.')
    except AssertionError:
        optimize_projector_convolutions = False
        laser_grid_xyz = laser_grid_xyz.reshape((-1, 3))

    try:
        assert optimize_camera_convolutions
        assert data.sensor_grid_format == GridFormat.X_Y_3
        sensor_grid_xyz = sensor_grid_xyz.reshape((nsx, nsy, 3))
        s_a = sensor_grid_xyz[0, 0, ..., :]
        s_b = sensor_grid_xyz[-1, 0, ..., :]
        s_c = sensor_grid_xyz[0, -1, ..., :]
        s_n = np.cross(s_b - s_a, s_c - s_a).reshape((1, 3))
        s_n /= np.linalg.norm(s_n, axis=-1, keepdims=True)

        s_dx = np.linalg.norm(
            sensor_grid_xyz[1, 0, ..., :] - sensor_grid_xyz[0, 0, ..., :])
        s_dy = np.linalg.norm(
            sensor_grid_xyz[0, 1, ..., :] - sensor_grid_xyz[0, 0, ..., :])

        dot_sv = np.sum(s_n * v_n, axis=-1)
        assert np.allclose(np.abs(dot_sv), 1)
        assert np.isclose(v_dx, s_dx) and np.isclose(v_dy, s_dy)
        log(LogLevel.INFO, 'tal.reconstruct.utils: Optimizing for camera convolutions.')
    except AssertionError:
        optimize_camera_convolutions = False
        sensor_grid_xyz = sensor_grid_xyz.reshape((-1, 3))

    if projector_focus is not None and not optimize_projector_convolutions:
        projector_focus = projector_focus.reshape((-1, 3))

    if not optimize_camera_convolutions:
        volume_xyz = volume_xyz.reshape((-1, 3))

    return (H, laser_grid_xyz, sensor_grid_xyz, volume_xyz,
            optimize_projector_convolutions, optimize_camera_convolutions)


def convert_reconstruction_from_N_3(data: NLOSCaptureData,
                                    reconstructed_volume_n3: Union[NLOSCaptureData.SingleReconstructionType,
                                                                   NLOSCaptureData.ExhaustiveReconstructionType],
                                    volume_xyz: NLOSCaptureData.VolumeXYZType,
                                    volume_format: VolumeFormat,
                                    camera_system: CameraSystem,
                                    projector_focus:  Union[NLOSCaptureData.Array3,
                                                            NLOSCaptureData.VolumeXYZType] = None):

    volume_format = _infer_volume_format(
        volume_xyz, volume_format, do_log=False)

    if camera_system.is_transient():
        shape = (data.H.shape[data.H_format.time_dim()],)
    else:
        shape = ()

    assert volume_format.xyz_dim_is_last(), 'Unexpected volume_format'

    is_exhaustive_reconstruction = \
        camera_system.implements_projector() \
        and projector_focus is not None and projector_focus.size > 3

    if is_exhaustive_reconstruction:
        # add an additional shape dimension for the exhaustive reconstruction
        shape += projector_focus.shape[:-1]

    shape += volume_xyz.shape[:-1]

    return reconstructed_volume_n3.reshape(shape)
