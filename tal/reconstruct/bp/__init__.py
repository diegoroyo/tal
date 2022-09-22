from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import VolumeFormat
from typing import Union

import numpy as np
_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]


def solve(data: NLOSCaptureData,
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = None) -> np.array:
    """
    NOTE: Does _NOT_ attempt to compensate effects caused by attenuation:
      - cos decay i.e. {sensor|laser}_grid_normals are ignored
      - 1/d^2 decay
    """
    from tal.reconstruct.bp.backprojection import backproject
    if volume_format == VolumeFormat.X_Y_Z_3:
        vsx, vsy, vsz, _ = volume_xyz.shape
        volume_xyz_n3 = volume_xyz.reshape((-1, 3))
    elif volume_format == VolumeFormat.N_3:
        volume_xyz_n3 = volume_xyz
    else:
        raise AssertionError('volume_format must be specified')

    reconstructed_volume_n3 = backproject(
        data, volume_xyz_n3, progress=True)

    if volume_format == VolumeFormat.X_Y_Z_3:
        reconstructed_volume = reconstructed_volume_n3.reshape((vsx, vsy, vsz))
    elif volume_format == VolumeFormat.N_3:
        reconstructed_volume = reconstructed_volume_n3
    else:
        raise AssertionError('volume_format must be specified')

    return reconstructed_volume
