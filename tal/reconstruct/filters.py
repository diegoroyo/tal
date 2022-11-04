from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat
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
        K_shape = (nt, 1, 1, 1, 1)
    else:
        raise AssertionError('Unknown H_format')

    padding = 0

    if filter_name == 'pf':
        wl_mean = kwargs.get('wl_mean', None)
        assert wl_mean is not None, \
            'For the "pf" filter, wl_mean must be specified'
        wl_sigma = kwargs.get('wl_sigma', None)
        wl_sigma = wl_sigma or wl_mean / np.sqrt(2)
        delta_t = kwargs.get('delta_t', None) or delta_t
        assert delta_t is not None, \
            'For the "pf" filter, delta_t must be specified through an NLOSCaptureData object or the delta_t argument'

        t_max = delta_t * (nt * 2 - 1)
        t = np.linspace(start=0, stop=t_max, num=nt * 2)

        #   vvv Gaussian envelope (x = t - t_max/2, mu = 0, sigma = wl_sigma)
        K = (1 / (wl_sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-((t - t_max / 2) / wl_sigma) ** 2 / 2) * \
            np.exp(2j * np.pi * t / wl_mean)
        #   ^^^ Pulse inside the Gaussian envelope (complex exponential)

        # center at zero (not in freq. domain but fftshift works)
        K = np.fft.fftshift(K)
    else:
        raise AssertionError(
            'Unknown filter_name. Check the documentation for available filters')

    if plot_filter:
        import matplotlib.pyplot as plt
        K_show = np.fft.ifftshift(K)
        plt.plot(t[:len(K_show)] - t[len(K_show) // 2], np.real(K_show), c='b')
        plt.plot(t[:len(K_show)] - t[len(K_show) // 2],
                 np.imag(K_show), c='b', linestyle='--')
        plt.show()
    if return_filter:
        return K

    # pad with identical, inverted signal
    if H_format == HFormat.T_Sx_Sy or H_format == HFormat.T_Lx_Ly_Sx_Sy:
        H = np.resize(H, (nt * 2, *H.shape[1:]))
        H[nt:, ...] = H[:nt, ...][::-1, ...]
        K_shape = (nt * 2,) + (1,) * (H.ndim - 1)
    else:
        raise AssertionError('Unknown H_format')

    H_fft = np.fft.fft(H, axis=0)
    K_fft = np.fft.fft(K)
    H_fft *= K_fft.reshape(K_shape)
    del K_fft
    HoK = np.fft.ifft(H_fft, axis=0)
    del H_fft

    # undo padding
    if H_format == HFormat.T_Sx_Sy or H_format == HFormat.T_Lx_Ly_Sx_Sy:
        return HoK[:nt, ...]
    else:
        raise AssertionError('Unknown H_format')
