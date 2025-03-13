"""
tal.reconstruct
===============

Functions to reconstruct of hidden scenes given captured data.

Also contains utilities to help in the process:
- Filtering functions
- Volume generation functions

There are multiple implementations of different reconstruction algorithms,
e.g. backprojection (tal.reconstruct.bp), phasor fields (tal.reconstruct.pf and tal.reconstruct.pf_dev),
each of those contains a solve function that takes the captured data and returns the reconstructed volume.
"""

from tal.reconstruct import bp, fbp, pf, pf_dev, fk

from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat
from typing import Union

_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]


def filter_H(data: _Data,
             filter_name: str,
             data_format: HFormat = HFormat.UNKNOWN,
             border: str = 'zero',
             plot_filter: bool = False,
             return_filter: bool = False,
             progress: bool = True,
             **kwargs) -> NLOSCaptureData.HType:
    """
    Filter a captured data signal (H) using specified filter_name
        * data_format should be non-null if not specified through data
        * border sets the behaviour for the edges of the convolution
          - 'erase': the filtered signal has the edges set to zero
          - 'zero': before filtering, pad the signal with zeros
          - 'edge': before filtering, pad the signal with edge values
        * If plot_filter=True, shows a plot of the resulting filter
        * If return_filter=True, returns the filter (K)
          else, returns the filtered signal (H * K)
        * If progress=True, shows a progress bar with the performed step

    Available filters and respective arguments:
    - filter_name='pf': Filter certain frequencies, weighted using a Gaussian on the frequency domain
        FIXME(diego): Update wl_mean/wl_sigma here and scene_defaults.yaml
        * wl_mean: Mean of the Gaussian in the frequency domain
        * wl_sigma: STD of the Gaussian in the frequency domain
        * delta_t: Time interval, must be non-null if not specified through data
        e.g. mean = 3, sigma = 0.5 will filter frequencies of around 3m,
        sigma = 0.5 does not translate to 3 +/- 0.5m, but it is somewhat like that
        for more info. see tal.reconstruct.pf_dev, which outputs the specific range
        of frequencies that are filtered given wl_mean and wl_sigma
    """
    from tal.reconstruct.filters import filter_H_impl
    return filter_H_impl(data, filter_name, data_format, border, plot_filter, return_filter, progress, **kwargs)


def compensate_laser_cos_dsqr(data: NLOSCaptureData):
    """
    Compensate for the cos decay and 1/d^2 decay of the laser, from the physical
    laser device to each illuminated point in the relay wall.

    (Operates in place)
    """
    import numpy as np

    def compensate(H, rw_xyz, rw_n):
        w_i = data.laser_xyz - rw_xyz
        d = np.linalg.norm(w_i)
        w_i = w_i / d
        cos_term = np.dot(w_i, rw_n)
        H /= cos_term / d ** 2

    def compensate_i(nl):
        for i in range(nl):
            compensate(data.H[:, i, ...],
                       data.laser_grid_xyz[i, :],
                       data.laser_grid_normals[i, :])

    def compensate_i_j(nlx, nly):
        for i in range(nlx):
            for j in range(nly):
                compensate(data.H[:, i, j, ...],
                           data.laser_grid_xyz[i, j, :],
                           data.laser_grid_normals[i, j, :])

    if data.H_format == HFormat.T_Lx_Ly_Sx_Sy:
        compensate_i_j(*data.laser_grid_xyz.shape[:2])
    elif data.H_format == HFormat.T_Sx_Sy:
        if data.is_laser_paired_to_sensor():
            compensate_i_j(*data.sensor_grid_xyz.shape[:2])
        else:
            assert data.laser_grid_xyz.size == 3
            compensate(data.H,
                       data.laser_grid_xyz.reshape(3),
                       data.laser_grid_normals.reshape(3))
    elif data.H_format == HFormat.T_Li_Si:
        compensate_i(data.laser_grid_xyz.shape[0])
    elif data.H_format == HFormat.T_Si:
        if data.is_laser_paired_to_sensor():
            compensate_i(data.sensor_grid_xyz.shape[0])
        else:
            assert data.laser_grid_xyz.size == 3
            compensate(data.H,
                       data.laser_grid_xyz.reshape(3),
                       data.laser_grid_normals.reshape(3))
    else:
        raise ValueError(
            f'This function is not implemented for H_format={data.H_format}')


def get_volume_min_max_resolution(minimal_pos, maximal_pos, resolution):
    import numpy as np
    assert np.all(maximal_pos > minimal_pos), \
        'maximal_pos must be greater than minimal_pos'
    e = resolution / 2  # half-voxel
    return np.moveaxis(np.mgrid[minimal_pos[0]+e:maximal_pos[0]:resolution,
                                minimal_pos[1]+e:maximal_pos[1]:resolution,
                                minimal_pos[2]+e:maximal_pos[2]:resolution], 0, -1)


def get_volume_project_rw(data: NLOSCaptureData, depths: Union[float, list]):
    """
    Generate a volume with the same XY coordinates as the sensor grid,
    and the Z coordinates specified by depths.
    """
    import numpy as np
    from tal.enums import GridFormat

    if isinstance(depths, (int, float)):
        depths = [depths]
    if data.sensor_grid_format == GridFormat.X_Y_3:
        sx, sy, _ = data.sensor_grid_xyz.shape
        volume_xyz = np.zeros((sx, sy, len(depths), 3))
        volume_xyz[..., 0] = data.sensor_grid_xyz[..., 0].reshape((sx, sy, 1))
        volume_xyz[..., 1] = data.sensor_grid_xyz[..., 1].reshape((sx, sy, 1))
        volume_xyz[..., 2] = depths
    else:
        raise AssertionError('This function only works with GridFormat.X_Y_3')

    return volume_xyz
