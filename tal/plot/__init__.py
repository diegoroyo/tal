from tal.io.capture_data import NLOSCaptureData
from typing import Union, List

_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]
_DataList = Union[List[_Data], _Data]


def xy_grid(data: _Data,
            size_x: int = 8, size_y: int = 8,
            t_start: int = None, t_end: int = None, t_step: int = 1):
    from tal.plot.xy import plot_xy_grid
    return plot_xy_grid(data, size_x, size_y, t_start, t_end, t_step)


def xy_interactive(data: _Data,
                   cmap: str = 'hot'):
    from tal.plot.xy import plot_xy_interactive
    return plot_xy_interactive(data, cmap)


def tx_interactive(data: _Data,
                   cmap: str = 'hot'):
    from tal.plot.xy import plot_xt_interactive
    return plot_xt_interactive(data, cmap)


def ty_interactive(data: _Data,
                   cmap: str = 'hot'):
    from tal.plot.xy import plot_yt_interactive
    return plot_yt_interactive(data, cmap)


def t_comparison(data_list: _DataList,
                 x: int = None, y: int = None,
                 t_start: int = None, t_end: int = None,
                 a_min: float = None, a_max: float = None,
                 labels: List[str] = None):
    from tal.plot.compare import plot_t_comparison
    return plot_t_comparison(data_list, x, y, t_start, t_end, a_min, a_max, labels)
