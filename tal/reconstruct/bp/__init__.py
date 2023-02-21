"""
tal.reconstruct.bp
==================

Reconstruction using the backprojection algorithm.
See "Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging."

Filtered backprojection is available in the fbp submodule, or by pre-filtering (see tal.reconstruct.filter_H).

Does _NOT_ attempt to compensate effects caused by attenuation:
    - cos decay i.e. {sensor|laser}_grid_normals are ignored
    - 1/d^2 decay
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
import numpy as np


def solve(data: NLOSCaptureData,
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
          camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT,
          progress: bool = True) -> np.array:  # FIXME(diego): volume type
    """
    See module description of tal.reconstruct.bp

    data
        See tal.io.read_capture

    volume_xyz
        Multi-dimensional array with XYZ coordinates of the volume voxels.
        See tal.enums.VolumeFormat for possible input formats.
        See e.g. tal.reconstruct.get_volume_min_max_resolution for utilities to generate this array.

    volume_format
        See tal.enums.VolumeFormat

    camera_system
        See tal.enums.CameraSystem

    progress
        If True, shows a progress bar with estimated time remaining.
    """
    from tal.reconstruct.utils import convert_to_N_3, convert_reconstruction_from_N_3
    H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3 = \
        convert_to_N_3(data, volume_xyz, volume_format)

    from tal.reconstruct.bp.backprojection import backproject
    reconstructed_volume_n3 = backproject(
        H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3,
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t,
        data.laser_xyz, data.sensor_xyz,
        progress=progress)

    return convert_reconstruction_from_N_3(data, reconstructed_volume_n3, volume_xyz, volume_format, camera_system)
