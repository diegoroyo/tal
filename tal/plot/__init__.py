from tal.io.capture_data import NLOSCaptureData
from typing import Union, List


_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]
_DataList = Union[List[_Data], _Data]


def xy_grid(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
            size_x: int = 8, size_y: int = 8,
            t_start: int = None, t_end: int = None, t_step: int = 1):
    from tal.plot.xy import plot_xy_grid
    return plot_xy_grid(data, size_x, size_y, t_start, t_end, t_step)


def t_comparison(data_list: _DataList,
                 x: int = None, y: int = None,
                 t_start: int = None, t_end: int = None,
                 a_min: float = None, a_max: float = None,
                 labels: List[str] = None):
    from tal.plot.compare import plot_t_comparison
    return plot_t_comparison(data_list, x, y, t_start, t_end, a_min, a_max, labels)


def volume(data: _Data, title: str = '', slider_title: str = 'Time',
           slider_step: float = 0.1, cmap: str = 'hot',
           opacity='sigmoid', backgroundcolor=None):
    from tal.plot.plotter3d import plot3d
    return plot3d(data, title, slider_title, slider_step, cmap, opacity,
                  backgroundcolor)


def xy_interactive(data: _Data, cmap: str = 'hot'):
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, 't')


def tx_interactive(data: _Data, cmap: str = 'hot'):
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, 'y')


def ty_interactive(data: _Data, cmap: str = 'hot'):
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, 'x')


def txy_interactive(data: _Data, cmap: str = 'hot', slice_axis: str = 't'):
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, slice_axis)
