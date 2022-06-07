from tal.io.capture_data import NLOSCaptureData
from tal.plot.xy import plot_xy_at_different_t
from typing import Union


def xy_at_different_t(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
                      size_x: int = 8, size_y: int = 8,
                      t_start: int = None, t_end: int = None, t_step: int = 1):
    return plot_xy_at_different_t(data, size_x, size_y, t_start, t_end, t_step)
