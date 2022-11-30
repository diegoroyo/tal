from tal.enums import HFormat, GridFormat, VolumeFormat, CameraSystem
from tal.io.capture_data import NLOSCaptureData
import numpy as np


def convert_to_N_3(data: NLOSCaptureData,
                   volume_xyz: NLOSCaptureData.VolumeXYZType,
                   volume_format: VolumeFormat):
    if data.H_format == HFormat.T_Si:
        H = data.H
    elif data.H_format == HFormat.T_Sx_Sy:
        nt, nsx, nsy = data.H.shape
        H = data.H.reshape(nt, nsx * nsy)
    else:
        raise AssertionError(f'H_format {data.H_format} not implemented')

    # FIXME(diego) also convert laser data and confocal/exhaustive H measurements

    if data.sensor_grid_format == GridFormat.N_3:
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

    if volume_format == VolumeFormat.X_Y_Z_3:
        volume_xyz_n3 = volume_xyz.reshape((-1, 3))
    elif volume_format == VolumeFormat.N_3:
        volume_xyz_n3 = volume_xyz
    else:
        raise AssertionError('volume_format must be specified')

    return (H, data.laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3)


def convert_reconstruction_from_N_3(data: NLOSCaptureData,
                                    reconstructed_volume_n3: np.ndarray,  # FIXME type
                                    volume_xyz: NLOSCaptureData.VolumeXYZType,
                                    volume_format: VolumeFormat,
                                    camera_system: CameraSystem):
    if camera_system.is_transient():
        time_dim = (data.H.shape[data.H_format.time_dim()],)
    else:
        time_dim = ()

    assert volume_format.xyz_dim_is_last(), 'Unexpected volume_format'
    return reconstructed_volume_n3.reshape(
        time_dim + volume_xyz.shape[:-1])
