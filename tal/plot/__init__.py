from tal.io.capture_data import NLOSCaptureData
from tal.plot.xy import plot_xy_grid
from tal.plot.xy import plot_txy_interactive, plot_zxy_interactive
from tal.plot.xy import ByAxis
from typing import Union

def xy_grid(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
            size_x: int = 8, size_y: int = 8,
            t_start: int = None, t_end: int = None, t_step: int = 1):
    return plot_xy_grid(data, size_x, size_y, t_start, t_end, t_step)


def txy_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType], 
                    cmap:str = 'hot', by: ByAxis = ByAxis.T):
    return plot_txy_interactive(data, cmap, by)

def zxy_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType], 
                    cmap:str = 'hot', by: ByAxis = ByAxis.Z):
    return plot_zxy_interactive(data, cmap, by)