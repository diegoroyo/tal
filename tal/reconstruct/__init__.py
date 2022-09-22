from tal.reconstruct.pf import *
from tal.reconstruct.bp import *
from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import HFormat
from typing import Union

_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]


def filter_H(data: _Data,
             filter_name: str,
             data_format: HFormat = HFormat.UNKNOWN,
             plot_filter: bool = False,
             return_filter: bool = False,
             **kwargs) -> NLOSCaptureData.HType:
    """
    Filter a captured data signal (H) using specified filter_name
        * data_format should be non-null if not specified through data
        * If plot_filter=True, shows a plot of the resulting filter
        * If return_filter=True, returns the filter (K)
          else, returns the filtered signal (H * K)
    Available filters and respective arguments:
    - pf: Filter certain frequencies, weighted using a Gaussian on the frequency domain
        * wl_mean: Mean of the Gaussian in the frequency domain
        * wl_sigma: STD of the Gaussian in the frequency domain
        * delta_t: Time interval, must be non-null if not specified through data
        e.g. mean = 3, sigma = 0.5 will filter frequencies of ~2-4m
    """
    from tal.reconstruct.filters import filter_H_impl
    return filter_H_impl(data, filter_name, data_format, plot_filter, return_filter, **kwargs)


# def get_volume(center, rotation, size, resolution):
#     #TODO: Volume type up to Diego
#     raise NotImplemented('get_volume to be implemented')

def get_volume():
    nx = 256
    ny = 256
    center = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
    dx = 1 / np.sqrt(2)
    dx /= nx
    dy = 1
    dy /= ny
    dz = 1 / np.sqrt(2)
    dz /= nx

    x = np.linspace(center[0] - dx * nx // 2, center[0] + dx * nx // 2, nx)
    y = np.linspace(center[1] - dy * ny // 2, center[1] + dy * ny // 2, ny)
    z = np.linspace(center[2] - dz * nx // 2, center[2] + dz * nx // 2, nx)
    xyz = np.stack((
        np.repeat(x.reshape((nx, 1, 1, 1)), ny, axis=1),
        np.repeat(y.reshape((1, ny, 1, 1)), nx, axis=0),
        np.repeat(z.reshape((nx, 1, 1, 1)), ny, axis=1),
    ), axis=-1)

    return xyz.reshape((-1, 3))
