from tal.reconstruct.bp import *
from tal.reconstruct.pf import *
from tal.reconstruct.pf_dev import *
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
    Available filters and respective arguments:
    - pf: Filter certain frequencies, weighted using a Gaussian on the frequency domain
        * wl_mean: Mean of the Gaussian in the frequency domain
        * wl_sigma: STD of the Gaussian in the frequency domain
        * delta_t: Time interval, must be non-null if not specified through data
        FIXME(diego): is this sentence true? vvv probably not
        e.g. mean = 3, sigma = 0.5 will filter frequencies of ~2-4m
    """
    from tal.reconstruct.filters import filter_H_impl
    return filter_H_impl(data, filter_name, data_format, border, plot_filter, return_filter, **kwargs)


def get_volume_min_max_resolution(minimal_pos, maximal_pos, resolution):
    assert np.all(maximal_pos > minimal_pos), \
        'maximal_pos must be greater than minimal_pos'
    e = resolution / 2  # half-voxel
    return np.moveaxis(np.mgrid[minimal_pos[0]+e:maximal_pos[0]:resolution,
                                minimal_pos[1]+e:maximal_pos[1]:resolution,
                                minimal_pos[2]+e:maximal_pos[2]:resolution], 0, -1)
