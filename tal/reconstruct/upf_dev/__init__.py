"""
tal.reconstruct.upf_dev
======================

Same as pf_dev but with already filtered data.

TODO: This is in an early state, probably will be merged with pf_dev.
And (hopefully) it will be better documented too.
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
from typing import Union
import numpy as np


def solve(data: NLOSCaptureData,
          frequencies: np.ndarray,  # FIXME
          weights: np.ndarray = None,  # FIXME
          nt: int = 1,  # FIXME
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
          camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT,
          projector_focus: Union[NLOSCaptureData.Array3,
                                 NLOSCaptureData.VolumeXYZType] = None,
          progress: bool = True,
          try_optimize_convolutions: bool = True) -> Union[NLOSCaptureData.SingleReconstructionType,
                                                           NLOSCaptureData.ExhaustiveReconstructionType]:
    """
    See module description of tal.reconstruct.pf_dev

    data
        See tal.io.read_capture

    frequencies, weights, nt
        TODO

    volume_xyz
        Multi-dimensional array with XYZ coordinates of the volume voxels.
        See tal.enums.VolumeFormat for possible input formats.
        See e.g. tal.reconstruct.get_volume_min_max_resolution for utilities to generate this array.

    volume_format
        See tal.enums.VolumeFormat

    camera_system
        See tal.enums.CameraSystem

    projector_focus
        Setting that changes how the virtual light is focused onto the hidden volume.
        'None' focuses the virtual light at the same points as the virtual camera (confocal camera)
            This is the behaviour of most imaging algorithms,
            especially the tal.enums.CameraSystem.CONFOCAL_TIME_GATED camera.
        When projector_focus = [x, y, z] if you have multiple laser points in your data,
            the illumination will be focused towards the point [x, y, z].
            This behaviour is especially useful for tal.enums.CameraSystem.PROJECTOR_CAMERA.
        When projector_focus = volume_xyz, this will yield
            a NLOSCaptureData.ExhaustiveReconstructionType with all possible projector_focus points.

    progress
        If True, shows a progress bar with estimated time remaining.

    try_optimize_convolutions
        When volume_xyz consists of depth-slices (Z-slices) that are 
            1) coplanar to the XY relay wall, and
            2) sampled at the same rate,
        the computation can be optimized to use less memory and be much faster.
        It is recommended to leave this set this to True. If it is not possible to apply
        this optimization, tal.reconstruct.pf_dev will fall back to the default implementation.
        You can generate these depth-slices with tal.reconstruct.get_volumeXXX functions.
    """
    raise NotImplementedError('This version of upf_dev is deprecated. '
                              'If you want to use it, use a previous version of TAL or contact the authors.')
    from tal.reconstruct.util import convert_to_N_3, convert_reconstruction_from_N_3
    H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3, \
        optimize_projector_convolutions, optimize_camera_convolutions = \
        convert_to_N_3(data, volume_xyz, volume_format,
                       try_optimize_convolutions=try_optimize_convolutions)

    from tal.reconstruct.upf_dev.phasor_fields import backproject_pf_multi_frequency
    reconstructed_volume_n3 = backproject_pf_multi_frequency(
        H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3, volume_xyz.shape[:-1],
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t, data.is_laser_paired_to_sensor(),
        projector_focus,
        frequencies, weights, nt,
        optimize_projector_convolutions, optimize_camera_convolutions,
        data.laser_xyz, data.sensor_xyz,
        progress=progress)

    mutliple_projector_points = \
        camera_system.implements_projector() \
        and projector_focus is not None and projector_focus.size > 3

    return convert_reconstruction_from_N_3(data, reconstructed_volume_n3, volume_xyz, volume_format, camera_system,
                                           is_exhaustive_reconstruction=mutliple_projector_points)
