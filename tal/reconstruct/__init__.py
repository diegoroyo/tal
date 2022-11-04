from tal.reconstruct.pf import *
from tal.reconstruct.bp import *
from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat
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


def get_volume(center, rotation, size, resolution):
    # TODO: Volume type up to Diego
    raise NotImplemented('get_volume to be implemented')
