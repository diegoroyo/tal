from tal.io.capture_data import NLOSCaptureData
from tal.enums import HFormat
import numpy as np


def filter_H_impl(data, filter_name, data_format, border, plot_filter, return_filter, **kwargs):
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

    assert border in ['erase', 'zero', 'edge'], \
        'border must be one of "erase", "zero" or "edge"'

    nt = H.shape[H_format.time_dim()]
    nt_pad = nt

    if filter_name == 'pf':
        wl_mean = kwargs.get('wl_mean', None)
        assert wl_mean is not None, \
            'For the "pf" filter, wl_mean must be specified'
        wl_sigma = kwargs.get('wl_sigma', None)
        wl_sigma = wl_sigma or wl_mean / np.sqrt(2)
        delta_t = kwargs.get('delta_t', None) or delta_t
        assert delta_t is not None, \
            'For the "pf" filter, delta_t must be specified through an NLOSCaptureData object or the delta_t argument'

        t_6sigma = int(np.round(6 * wl_sigma / delta_t))  # used for padding
        if t_6sigma % 2 == 1:
            t_6sigma += 1  # its easier if padding is even
        if return_filter:
            nt_pad = t_6sigma
        else:
            nt_pad = nt + 2 * (t_6sigma - 1)
        t_max = delta_t * (nt_pad - 1)
        t = np.linspace(start=0, stop=t_max, num=nt_pad)

        #   vvv Gaussian envelope (x = t - t_max/2, mu = 0, sigma = wl_sigma)
        gaussian_envelope = np.exp(-((t - t_max / 2) / wl_sigma) ** 2 / 2)
        K = gaussian_envelope / np.sum(gaussian_envelope) * \
            np.exp(2j * np.pi * t / wl_mean)
        #   ^^^ Pulse inside the Gaussian envelope (complex exponential)

        # center at zero (not in freq. domain but fftshift works)
        K = np.fft.ifftshift(K)
    else:
        raise AssertionError(
            'Unknown filter_name. Check the documentation for available filters')

    if plot_filter:
        import matplotlib.pyplot as plt
        K_show = np.fft.fftshift(K)
        cut = (nt_pad - t_6sigma) // 2
        if cut > 0:
            K_show = K_show[cut:-cut]
        plt.plot(t[:len(K_show)] - t[len(K_show) // 2], np.real(K_show),
                 c='b')
        plt.plot(t[:len(K_show)] - t[len(K_show) // 2], np.imag(K_show),
                 c='b', linestyle='--')
        plt.plot(t[:len(K_show)] - t[len(K_show) // 2], np.abs(K_show),
                 c='r')
        plt.show()
    if return_filter:
        return K

    padding = (nt_pad - nt)
    assert padding % 2 == 0
    padding //= 2

    # pad with edge values
    if H_format.time_dim() == 0:
        mode = None
        if border == 'edge':
            mode = 'edge'
        if border == 'zero' or border == 'erase':
            mode = 'constant'
        if mode:
            H_pad = np.pad(H,
                           ((padding, padding),) +  # first dim (time)
                           ((0, 0),) * (H.ndim - 1),  # other dims
                           mode=mode)
        K_shape = (nt_pad,) + (1,) * (H.ndim - 1)
    else:
        raise AssertionError('Unknown H_format')

    H_fft = np.fft.fft(H_pad, axis=0)
    K_fft = np.fft.fft(K)
    H_fft *= K_fft.reshape(K_shape)
    del K_fft
    HoK = np.fft.ifft(H_fft, axis=0)
    del H_fft

    # undo padding
    if H_format.time_dim() == 0:
        HoK = HoK[padding:-padding, ...]
        if border == 'erase':
            HoK[:padding//2] = 0
            HoK[-padding//2:] = 0
        return HoK
    else:
        raise AssertionError('Unknown H_format')
