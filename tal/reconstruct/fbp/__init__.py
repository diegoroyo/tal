from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
import numpy as np


def solve(data: NLOSCaptureData,
          wl_mean: float,
          wl_sigma: float,
          border: str = 'zero',
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = None,
          camera_system: CameraSystem = CameraSystem.STEADY) -> np.array:  # FIXME type
    """
    Shortcut to call reconstruct.filter_H (pf filter) and reconstruct.bp.solve
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
