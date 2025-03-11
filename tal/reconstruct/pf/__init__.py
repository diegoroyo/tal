"""
tal.reconstruct.pf
==================

Reconstruction using the phasor fields framework.
See "Non-Line-of-Sight Imaging using Phasor Field Virtual Wave Optics."

Does _NOT_ attempt to compensate effects caused by attenuation:
    - cos decay i.e. {sensor|laser}_grid_normals are ignored
    - 1/d^2 decay
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import CameraSystem, VolumeFormat
from typing import Any, Tuple
import numpy as np

c = 299_792_458

def solve(data: NLOSCaptureData, c_wavelength: float, wave_cycles: float,
          camera_system: CameraSystem = CameraSystem.TRANSIENT_T0,
          volume: NLOSCaptureData.VolumeXYZType = None) -> np.ndarray:
    """
    Reconstruct a NLOS captured data with a gaussian pulse in the volume
    using phasor fields and the RSD propagation.
    @param data         : NLOS capture data at the relay wall
    @param c_wavelength : Central wavelength of the virtual illumination pulse
                          used for the reconstruction.
    @param wave_cycles  : Number of central wavelength cycles of the virtual
                          illumination pulse
    @param camera_system: Phasor Fields camera system
    @param volume       : Desired volume in the 3d space to reconstruct from 
                          the relay wall
    @return             : Reconstruction from the relay wall data to the given
                          volume, in the given form.    
    """
    from tal.reconstruct.pf.phasor_fields import reconstruct
    from tal.reconstruct.util import _infer_volume_format
    from tal.reconstruct import get_volume_project_rw
    from tal.reconstruct.filters import HFilter
    from tal.config import get_resources

    downscale = get_resources().downscale
    if downscale is not None and downscale > 1:
        data.downscale(downscale)

    pf_filter = HFilter('pf',delta_t = data.delta_t, n_w = data.H.shape[0], 
                        lambda_c = c_wavelength, cycles=wave_cycles)
    # Data extraction
    H = data.H
    # Time extraction
    if data.t_start is not None:
        # Correct delayed data
        t_pad = int(np.round(data.t_start / data.delta_t))
        H = np.pad(H, ((t_pad, 0), (0, 0), (0, 0)), 'minimum')

    V = volume
    if volume is None:
        V = get_volume_project_rw(data, 0.8+np.arange(0, 0.5, 0.01))
    
    volume_format = _infer_volume_format(V, VolumeFormat.UNKNOWN)

    return reconstruct(data, pf_filter, V,
                        volume_format,
                        camera_system)
