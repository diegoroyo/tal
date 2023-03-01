"""
tal.reconstruct.pf_dev
======================

Reconstruction using the phasor fields framework.
See "Non-Line-of-Sight Imaging using Phasor Field Virtual Wave Optics."

Does _NOT_ attempt to compensate effects caused by attenuation:
    - cos decay i.e. {sensor|laser}_grid_normals are ignored
    - 1/d^2 decay
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
import numpy as np


def solve(data: NLOSCaptureData,
          wl_mean: float,
          wl_sigma: float,
          border: str = 'zero',
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
          camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT,
          progress: bool = True,
          try_optimize_convolutions: bool = True) -> np.array:  # FIXME(diego) type
    """
    See module description of tal.reconstruct.pf_dev

    data
        See tal.io.read_capture

    wl_mean, wl_sigma, border
        Filter parameters. See tal.reconstruct.filter_H

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

    try_optimize_convolutions
        When volume_xyz consists of depth-slices (Z-slices) that are parallel to the XY relay wall,
        the computation can be optimized to use less memory and be much faster.
        It is recommended to set this to True.
        You can generate these depth-slices with tal.reconstruct.get_volume_project_rw(...).
    """
    from tal.reconstruct.utils import convert_to_N_3, convert_reconstruction_from_N_3
    H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3 = \
        convert_to_N_3(data, volume_xyz, volume_format,
                       try_optimize_convolutions=try_optimize_convolutions)

    from tal.reconstruct.pf_dev.phasor_fields import backproject_pf_multi_frequency
    reconstructed_volume_n3 = backproject_pf_multi_frequency(
        H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3,
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t,
        wl_mean, wl_sigma, border,
        data.laser_xyz, data.sensor_xyz,
        progress=progress)

    return convert_reconstruction_from_N_3(data, reconstructed_volume_n3, volume_xyz, volume_format, camera_system)
