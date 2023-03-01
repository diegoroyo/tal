from tal.enums import HFormat, GridFormat, VolumeFormat, CameraSystem
from tal.io.capture_data import NLOSCaptureData
import numpy as np


def _infer_volume_format(volume_xyz, volume_format, log=True):
    # infer volume_format if it is unknown
    if volume_format == VolumeFormat.UNKNOWN:
        assert volume_xyz.shape[-1] == 3, \
            'Could not infer volume_format. Please specify it manually.'
        if volume_xyz.ndim == 2:
            if log:
                print('tal.reconstruct.utils: Assuming that volume_xyz is N_3')
            volume_format = VolumeFormat.N_3
        elif volume_xyz.ndim == 3:
            if log:
                print('tal.reconstruct.utils: Assuming that volume_xyz is X_Y_3')
            volume_format = VolumeFormat.X_Y_3
        elif volume_xyz.ndim == 4:
            if log:
                print('tal.reconstruct.utils: Assuming that volume_xyz is X_Y_Z_3')
            volume_format = VolumeFormat.X_Y_Z_3
        else:
            raise AssertionError(
                'Could not infer volume_format. Please specify it manually.')

    return volume_format


def convert_to_N_3(data: NLOSCaptureData,
                   volume_xyz: NLOSCaptureData.VolumeXYZType,
                   volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
                   try_optimize_convolutions: bool = False):

    volume_format = _infer_volume_format(volume_xyz, volume_format, log=True)

    # this variable is set to false during the conversion
    can_optimize_convolutions = \
        volume_format in [VolumeFormat.X_Y_3, VolumeFormat.X_Y_Z_3]

    if data.H_format == HFormat.T_Si:
        can_optimize_convolutions = False
        H = data.H
    elif data.H_format == HFormat.T_Sx_Sy:
        nt, nsx, nsy = data.H.shape
        H = data.H.reshape(nt, nsx * nsy)
    else:
        raise AssertionError(f'H_format {data.H_format} not implemented')

    # FIXME(diego) also convert laser data and confocal/exhaustive H measurements

    if data.sensor_grid_format == GridFormat.N_3:
        can_optimize_convolutions = False
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
    else:
        raise AssertionError(
            f'sensor_grid_format {data.sensor_grid_format} not implemented')

    assert H.shape[1] == sensor_grid_xyz.shape[0], \
        'H.shape does not match with sensor_grid_xyz.shape. Different number of points than measurements.'

    assert volume_format.xyz_dim_is_last(), 'Unexpected volume_format'
    try:
        assert try_optimize_convolutions and can_optimize_convolutions
        nvx, nvy = volume_xyz.shape[:2]
        assert nvx == nsx and nvy == nsy
        H = H.reshape((nt, nvx, nvy))
        sensor_grid_xyz = sensor_grid_xyz.reshape((nvx, nvy, 3))
        volume_xyz = volume_xyz
        print('tal.reconstruct.utils: Optimizing for convolutions. '
              'Other algorithms (e.g. tal.reconstruct.pf_dev) should also log that it is being used.')
    except AssertionError:
        volume_xyz = volume_xyz.reshape((-1, 3))

    return (H, data.laser_grid_xyz, sensor_grid_xyz, volume_xyz)


def convert_reconstruction_from_N_3(data: NLOSCaptureData,
                                    # FIXME(diego) type
                                    reconstructed_volume_n3: np.ndarray,
                                    volume_xyz: NLOSCaptureData.VolumeXYZType,
                                    volume_format: VolumeFormat,
                                    camera_system: CameraSystem):

    volume_format = _infer_volume_format(volume_xyz, volume_format, log=False)

    if camera_system.is_transient():
        time_dim = (data.H.shape[data.H_format.time_dim()],)
    else:
        time_dim = ()

    assert volume_format.xyz_dim_is_last(), 'Unexpected volume_format'
    return reconstructed_volume_n3.reshape(
        time_dim + volume_xyz.shape[:-1])
