"""
tal.reconstruct.fbp
===================

Reconstruction using filtered backprojection.

Shortcut to call reconstruct.filter_H (pf filter) and reconstruct.bp.solve (see respective docstrings for details)
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
          projector_focus: NLOSCaptureData.Array3 = None,
          compensate_invsq: bool = False,
          progress: bool = True) -> NLOSCaptureData.SingleReconstructionType:
    """
    See module description of tal.reconstruct.fbp

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
    from tal.reconstruct import filter_H
    old_H = data.H
    data.H = filter_H(data, filter_name='pf', border=border,
                      wl_mean=wl_mean, wl_sigma=wl_sigma)

    from tal.reconstruct.bp import solve as bp_solve
    H_1 = bp_solve(data,
                   volume_xyz=volume_xyz,
                   volume_format=volume_format,
                   camera_system=camera_system,
                   projector_focus=projector_focus,
                   compensate_invsq=compensate_invsq,
                   progress=progress)
    data.H = old_H
    return H_1
