"""
tal.reconstruct.bp
==================

Reconstruction using the backprojection algorithm.
See "Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging."

This implementation is particularly well suited when:

- The 3D volume that you want to reconstruct is *not* coplanar to the relay wall
and/or
- You want a time-gated reconstruction, not a time-resolved reconstruction
and/or
- You have a projector camera system and only want to focus light to a single point

If that is not your case, you might want to check the tal.reconstruct.pf_dev module.

Filtered backprojection is available in the fbp submodule, or by pre-filtering (see tal.reconstruct.filter_H).

Does _NOT_ attempt to compensate effects caused by attenuation:
    - cos decay of {sensor|laser}_grid_normals are ignored
    - cos decay of hidden geometry is ignored
    - 1/d^2 decay
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
import numpy as np


def solve(data: NLOSCaptureData,
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
          camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT,
          projector_focus: NLOSCaptureData.Array3 = None,
          compensate_invsq: bool = False,
          progress: bool = True) -> NLOSCaptureData.SingleReconstructionType:
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

    projector_focus
        Setting that changes how the virtual light is focused onto the hidden volume.
        'None' focuses the virtual light at the same points as the virtual camera (confocal camera)
            This is the behaviour of most imaging algorithms,
            especially the tal.enums.CameraSystem.CONFOCAL_TIME_GATED camera.
        When projector_focus = [x, y, z] if you have multiple laser points in your data,
            the illumination will be focused towards the point [x, y, z].
            This behaviour is especially useful for tal.enums.CameraSystem.PROJECTOR_CAMERA.
        In the pf_dev module you can set projector_focus = volume_xyz and that will yield
            a NLOSCaptureData.ExhaustiveReconstructionType with all possible projector_focus points.

    compensate_invsq
        If True, the inverse square falloff of light is compensated for, i.e., objects further away
        from the relay wall will appear brighter in the reconstruction.

    progress
        If True, shows a progress bar with estimated time remaining.
    """
    from tal.reconstruct.util import convert_to_N_3, convert_reconstruction_from_N_3
    if projector_focus is not None:
        projector_focus = np.array(projector_focus)
    H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3, _, __ = \
        convert_to_N_3(data, volume_xyz, volume_format)

    from tal.reconstruct.bp.backprojection import backproject
    reconstructed_volume_n3 = backproject(
        H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3, volume_xyz.shape[:-1],
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t, data.is_laser_paired_to_sensor(),
        projector_focus,
        data.laser_xyz, data.sensor_xyz,
        compensate_invsq=compensate_invsq,
        progress=progress)

    return convert_reconstruction_from_N_3(data, reconstructed_volume_n3, volume_xyz, volume_format, camera_system)
