"""
tal.plot
========

Functions for plotting and examining captured data or reconstructions using that data.

They can be called from Python, but also can be called using the command line interface
by running tal plot <function_name> <args> <kwargs> (see tal plot -h).
"""

import numpy as np  # FIXME type hints for function below
from tal.io.capture_data import NLOSCaptureData
from typing import Union, List


_Data = Union[NLOSCaptureData, NLOSCaptureData.HType]
_DataList = Union[List[_Data], _Data]


def amplitude_phase(data: np.ndarray, title: str = ''):
    """
    For a 2D array with complex data, plot amplitude/phase
    """
    from tal.plot.xy import plot_amplitude_phase
    return plot_amplitude_phase(data, title)


def xy_grid(data: _Data,
            size_x: int = 8, size_y: int = 8,
            t_start: int = None, t_end: int = None, t_step: int = 1):
    """
    For a single impulse response H with shape (T, Sx, Sy), plot a grid of temporal slices.

    size_x, size_y
        Size of the grid.

    t_start, t_end, t_step
        Limit the temporal range to [t_start, t_end) with step t_step.
        This operation is done before the grid is created.
    """
    from tal.plot.xy import plot_xy_grid
    return plot_xy_grid(data, size_x, size_y, t_start, t_end, t_step)


def t_comparison(data_list: _DataList,
                 x: int = None, y: int = None,
                 t_start: int = None, t_end: int = None,
                 a_min: float = None, a_max: float = None,
                 normalize: bool = False,
                 labels: List[str] = None):
    """
    For impulse responses H, do an interactive plot of transient pixels.
    This function can be called with multiple H (to compare transient pixels),
    but they must have the same shape.

    Preferred to call using CLI e.g. tal plot t_comparison <file_1> <file_2>
    """
    from tal.plot.compare import plot_t_comparison
    return plot_t_comparison(data_list, x, y, t_start, t_end, a_min, a_max, normalize, labels)


def volume(data: _Data, title: str = '', slider_title: str = 'Time',
           slider_step: float = 0.1, color: str = 'hot',
           opacity: str = 'sigmoid', backgroundcolor: str = None):
    """ 3D volume plot of the data """
    from tal.plot.plotter3d import plot3d
    return plot3d(data, title, slider_title, slider_step, color, opacity,
                  backgroundcolor)


def xy_interactive(data: _Data, cmap: str = 'hot'):
    """ See txy_interactive. Shortcut with slice_axis='t' """
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, 't')


def tx_interactive(data: _Data, cmap: str = 'hot'):
    """ See txy_interactive. Shortcut with slice_axis='y' """
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, 'y')


def ty_interactive(data: _Data, cmap: str = 'hot'):
    """ See txy_interactive. Shortcut with slice_axis='x' """
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, 'x')


def txy_interactive(data: _Data, cmap: str = 'hot', slice_axis: str = 't'):
    """
    For 3D data, plot a 2D slice of the data with the given slice_axis fixed.
    User can interactively change the slice position.
    """
    from tal.plot.xy import plot_txy_interactive
    return plot_txy_interactive(data, cmap, slice_axis)
