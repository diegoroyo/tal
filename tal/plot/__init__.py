from tal.io.capture_data import NLOSCaptureData
from tal.plot.xy import plot_xy_grid, plot_xy_interactive
from typing import Union


def xy_grid(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
            size_x: int = 8, size_y: int = 8,
            t_start: int = None, t_end: int = None, t_step: int = 1):
    return plot_xy_grid(data, size_x, size_y, t_start, t_end, t_step)


def xy_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType]):
    # TODO(pablo): fill
    # maybe use data_2 = None, data_3 = None for multiple etc.
    return plot_xy_interactive(data)
