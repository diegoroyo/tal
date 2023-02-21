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
          camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT) -> np.array:  # FIXME(diego) type
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
                   camera_system=camera_system)
    data.H = old_H
    return H_1
