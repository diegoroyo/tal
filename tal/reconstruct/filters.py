from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import HFormat
from scipy.ndimage import convolve1d
import numpy as np


def filter_H_impl(data, filter_name, data_format, plot_filter, return_filter, **kwargs):
    if isinstance(data, NLOSCaptureData):
        assert data.H_format != HFormat.UNKNOWN or data_format is not None, \
            'H format must be specified through the NLOSCaptureData object or the data_format argument'
        H_format = (data_format
                    if data_format != HFormat.UNKNOWN
                    else None) or data.H_format
        H = data.H
        delta_t = data.delta_t
    else:
        assert data_format is not None and data_format != HFormat.UNKNOWN, \
            'If data is not a NLOSCaptureData object, H format must be specified through the data_format argument'
        H_format = data_format
        H = data
        delta_t = None

    if H_format == HFormat.T_Sx_Sy:
        nt, nsx, nsy = H.shape
    elif H_format == HFormat.T_Lx_Ly_Sx_Sy:
        nt, nlx, nly, nsx, nsy = H.shape
    else:
        raise AssertionError('Unknown H_format')

    if filter_name == 'pf':
        wl_mean = kwargs.get('wl_mean', None)
        assert wl_mean is not None, \
            'For the "pf" filter, wl_mean must be specified'
        wl_sigma = kwargs.get('wl_sigma', None)
        wl_sigma = wl_sigma or wl_mean / np.sqrt(2)
        delta_t = kwargs.get('delta_t', None) or delta_t
        assert delta_t is not None, \
            'For the "pf" filter, delta_t must be specified through an NLOSCaptureData object or the delta_t argument'

        t_max = delta_t * (nt - 1)
        t = np.linspace(start=0, stop=t_max, num=nt)

        K = np.exp(-(t - t_max / 2) ** 2 / (2 * wl_sigma) ** 2) * \
            np.exp(2j * np.pi * t / wl_mean)
    else:
        raise AssertionError(
            'Unknown filter_name. Check the documentation for available filters')

    if plot_filter:
        import matplotlib.pyplot as plt
        plt.plot(t[:len(K)] - t[len(K) // 2], np.real(K), c='b')
        plt.plot(t[:len(K)] - t[len(K) // 2],
                 np.imag(K), c='b', linestyle='--')
        plt.show()
    if return_filter:
        return K

    HoK = convolve1d(H, K, axis=0, mode='constant', cval=0)
    return HoK
