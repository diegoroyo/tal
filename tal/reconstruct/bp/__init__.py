from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat, GridFormat, VolumeFormat, CameraSystem
from typing import Union

import numpy as np
_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]


def solve(data: NLOSCaptureData,
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = None,
          camera_system: CameraSystem = CameraSystem.STEADY) -> np.array:
    """
    NOTE: Does _NOT_ attempt to compensate effects caused by attenuation:
      - cos decay i.e. {sensor|laser}_grid_normals are ignored
      - 1/d^2 decay
    """
    if data.H_format == HFormat.T_Si:
        H = data.H
    elif data.H_format == HFormat.T_Sx_Sy:
        nt, nsx, nsy = data.H.shape
        H = data.H.reshape(nt, nsx * nsy)
    else:
        raise AssertionError(f'H_format {data.H_format} not implemented')

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

    if volume_format == VolumeFormat.X_Y_Z_3:
        nvx, nvy, nvz, _ = volume_xyz.shape
        volume_xyz_n3 = volume_xyz.reshape((-1, 3))
    elif volume_format == VolumeFormat.N_3:
        volume_xyz_n3 = volume_xyz
    else:
        raise AssertionError('volume_format must be specified')

    from tal.reconstruct.bp.backprojection import backproject
    reconstructed_volume_n3 = backproject(
        H, data.laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3,
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t,
        data.laser_xyz, data.sensor_xyz,
        progress=True)

    if camera_system.is_transient():
        assert data.H_format == HFormat.T_Si or data.H_format == HFormat.T_Sx_Sy or data.H_format == HFormat.T_Lx_Ly_Sx_Sy, \
            'Cannot find time dimension given H_format'
        time_dim = (data.H.shape[0],)
    else:
        time_dim = ()

    if volume_format == VolumeFormat.X_Y_Z_3:
        reconstructed_volume = reconstructed_volume_n3.reshape(
            time_dim + (nvx, nvy, nvz))
    elif volume_format == VolumeFormat.N_3:
        reconstructed_volume = reconstructed_volume_n3
    else:
        raise AssertionError('volume_format must be specified')

    return reconstructed_volume
