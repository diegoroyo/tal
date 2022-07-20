from tal.io.capture_data import NLOSCaptureData
from tal.plot.compare import plot_t_comparison
from tal.plot.xy import plot_xy_grid, plot_xy_interactive
from tal.plot.xy import plot_xt_interactive, plot_yt_interactive
from typing import Union


def xy_grid(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
            size_x: int = 8, size_y: int = 8,
            t_start: int = None, t_end: int = None, t_step: int = 1):
    return plot_xy_grid(data, size_x, size_y, t_start, t_end, t_step)


def xy_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
                   cmap: str = 'hot'):
    return plot_xy_interactive(data, cmap)


def tx_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
                   cmap: str = 'hot'):
    return plot_xt_interactive(data, cmap)


def ty_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
                   cmap: str = 'hot'):
    return plot_yt_interactive(data, cmap)


def t_comparison(x: int = None, y: int = None,
                 t_start: int = None, t_end: int = None,
                 a_min: float = None, a_max: float = None,
                 **kwargs: Union[NLOSCaptureData, NLOSCaptureData.HType]):
    return plot_t_comparison(x, y, t_start, t_end, a_min, a_max, **kwargs)
