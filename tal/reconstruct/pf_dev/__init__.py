from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
from tal.reconstruct.utils import convert_to_N_3, convert_reconstruction_from_N_3
import numpy as np


def solve(data: NLOSCaptureData,
          wl_mean: float,
          wl_sigma: float,
          edges: str = 'zero',
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = None,
          camera_system: CameraSystem = CameraSystem.STEADY) -> np.array:  # FIXME type
    """
    NOTE: Does _NOT_ attempt to compensate effects caused by attenuation:
      - cos decay i.e. {sensor|laser}_grid_normals are ignored
      - 1/d^2 decay
    TODO(diego): docs
    """
    H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3 = \
        convert_to_N_3(data, volume_xyz, volume_format)

    from tal.reconstruct.pf_dev.phasor_fields import backproject_pf_multi_frequency
    reconstructed_volume_n3 = backproject_pf_multi_frequency(
        H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3,
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t,
        wl_mean, wl_sigma, edges,
        data.laser_xyz, data.sensor_xyz,
        progress=True)

    return convert_reconstruction_from_N_3(data, reconstructed_volume_n3, volume_xyz, volume_format, camera_system)
