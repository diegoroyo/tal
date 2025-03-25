"""
tal.reconstruct.pf_dev
======================

Reconstruction using the phasor fields framework.
See "Non-Line-of-Sight Imaging using Phasor Field Virtual Wave Optics."

This implementation is particularly well suited when:

- The 3D volume that you want to reconstruct is coplanar to the relay wall
and/or
- You want a time-resolved reconstruction, not a time-gated reconstruction
and/or
- You have a projector camera system and want to focus light to multiple points

If that is not your case, you might want to check the tal.reconstruct.pf_dev module.

Does _NOT_ attempt to compensate effects caused by attenuation:
    - cos decay i.e. {sensor|laser}_grid_normals are ignored
    - 1/d^2 decay
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import VolumeFormat, CameraSystem
from typing import Union
import numpy as np


def solve(data: NLOSCaptureData,
          wl_mean: float,
          wl_sigma: float,
          border: str = 'zero',
          volume_xyz: NLOSCaptureData.VolumeXYZType = None,
          volume_format: VolumeFormat = VolumeFormat.UNKNOWN,
          camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT,
          projector_focus: Union[NLOSCaptureData.Array3,
                                 NLOSCaptureData.VolumeXYZType] = None,
          progress: bool = True,
          compensate_invsq: bool = False,
          output_frequencies: bool = None,
          try_optimize_convolutions: bool = True) -> Union[NLOSCaptureData.SingleReconstructionType,
                                                           NLOSCaptureData.ExhaustiveReconstructionType]:
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

    compensate_invsq
        If True, the inverse square falloff of light is compensated for, i.e., objects further away
        from the relay wall will appear brighter in the reconstruction.

    try_optimize_convolutions
        When volume_xyz consists of depth-slices (Z-slices) that are 
            1) coplanar to the XY relay wall, and
            2) sampled at the same rate,
        the computation can be optimized to use less memory and be much faster.
        It is recommended to leave this set this to True. If it is not possible to apply
        this optimization, tal.reconstruct.pf_dev will fall back to the default implementation.
        You can generate these depth-slices with tal.reconstruct.get_volumeXXX functions.
    """
    from tal.reconstruct.util import convert_to_N_3, convert_reconstruction_from_N_3
    if projector_focus is not None:
        projector_focus = np.array(projector_focus)
    H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3, \
        optimize_projector_convolutions, optimize_camera_convolutions = \
        convert_to_N_3(data, volume_xyz, volume_format,
                       projector_focus=projector_focus,
                       try_optimize_convolutions=try_optimize_convolutions)

    if output_frequencies is None:
        output_frequencies = (
            camera_system.is_transient()
            and
            data.H_format.is_fourier_domain()
        )

    from tal.reconstruct.pf_dev.phasor_fields import backproject_pf_multi_frequency
    reconstructed_volume_n3 = backproject_pf_multi_frequency(
        H, laser_grid_xyz, sensor_grid_xyz, volume_xyz_n3, volume_xyz.shape[:-1],
        camera_system, data.t_accounts_first_and_last_bounces,
        data.t_start, data.delta_t, data.is_laser_paired_to_sensor(),
        projector_focus,
        wl_mean, wl_sigma, border,
        optimize_projector_convolutions, optimize_camera_convolutions,
        data.laser_xyz, data.sensor_xyz,
        compensate_invsq=compensate_invsq,
        original_nt=data.scene_info.get('original_nt', None),
        skip_fft_in=data.H_format.is_fourier_domain(),
        skip_fft_out=output_frequencies,
        progress=progress)

    return convert_reconstruction_from_N_3(data, reconstructed_volume_n3,
                                           volume_xyz, volume_format,
                                           camera_system, projector_focus)


def convert_data_to_fourier(data, wl_mean, wl_sigma, border='zero'):
    """
    Converts H to frequency domain, doing a band-pass filter based on wl-mean and wl-sigma.

    Careful, this does not use any padding. If your temporal signal does not have enough zeros
    at the edges, the convolution can have some artifacts.
    """
    import numpy as np
    from tal.config import get_resources
    from tal.reconstruct.pf_dev.phasor_fields import _get_padding, _get_freqs_weights
    from tal.log import log, LogLevel
    assert data.t_start == 0, 'tal.reconstruct.pf_dev: t_start must be 0'

    nt, other = data.H.shape[0], data.H.shape[1:]

    if border is None:
        padding = 0
        H_fft = data.H
    elif border == 'zero':
        padding = _get_padding(wl_sigma, data.delta_t)
        H_fft = np.pad(data.H,
                       ((padding, padding),) + ((0, 0),) * (data.H.ndim - 1),
                       'constant')
    else:
        raise AssertionError(f"Border must be 'zero' or None, not {border}")

    na = np.prod(other)
    nf = nt + 2 * padding
    H_fft = H_fft.reshape(nf, na).astype(np.complex64)

    range_all = np.arange(na)
    get_resources().split_work(
        lambda subrange_a: np.fft.fft(H_fft[:, subrange_a], axis=0),
        data_in=range_all,
        data_out=H_fft,
        slice_dims=(0, 1),
    )

    H_fft = H_fft.reshape(nf, *other)

    freqs, _, freq_min_idx, freq_max_idx = _get_freqs_weights(
        nf, data.delta_t, wl_mean, wl_sigma)
    log(LogLevel.INFO, 'tal.reconstruct.pf_dev: '
        f'Using {len(freqs)} wavelengths from {1 / freqs[-1]:.4f}m to {1 / freqs[0]:.4f}m')

    data.scene_info['original_nt'] = nt
    data.H = H_fft[freq_min_idx:freq_max_idx+1]
    data.H_format = data.H_format.to_fourier_domain()

    return data
