from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import HFormat
from tal.util import SPEED_OF_LIGHT
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_xy_grid(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
                 size_x: int, size_y: int,
                 t_start: int, t_end: int, t_step: int):
    def t_to_time(t): return f'Bin #{t_start + t * t_step}'
    if isinstance(data, NLOSCaptureData):
        assert data.H_format == HFormat.T_Sx_Sy, \
            'plot_xy_grid does not support this data format'
        txy = data.H
        if data.t_start is not None and data.delta_t is not None:
            def t_to_time(
                t): return f'Bin #{(t_start or 0) + t * t_step}, {(data.t_start + t * data.delta_t) * 1e12 / SPEED_OF_LIGHT:.0f} ps'
    else:
        assert data.ndim == 3 and data.shape[1] == data.shape[2], \
            'plot_xy_grid does not support this data format'
        txy = data
    txy = txy[t_start:t_end:t_step, ...]
    nt = txy.shape[0]
    step = 1
    plot_size = size_x * size_y
    while nt // step > plot_size:
        step *= 2
    txy_min, txy_max = np.min(txy), np.max(txy)
    fig, axs = plt.subplots(size_y, size_x)

    for i in tqdm(range(plot_size)):
        t_bin = i * step
        image = txy[t_bin]
        row = i // size_x
        col = i % size_x
        mappable = axs[row, col].imshow(image.astype(
            np.float32), cmap='jet', vmin=txy_min, vmax=txy_max)
        fig.colorbar(mappable, ax=axs[row, col])
        axs[row, col].axis('off')
        axs[row, col].set_title(t_to_time((t_start or 0) + t_bin * t_step))

    plt.tight_layout()
    plt.show()


def plot_xy_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType]):
    if isinstance(data, NLOSCaptureData):
        assert data.H_format == HFormat.T_Sx_Sy, \
            'plot_xy_grid does not support this data format'
        txy = data.H
    else:
        assert data.ndim == 3 and data.shape[1] == data.shape[2], \
            'plot_xy_grid does not support this data format'
        txy = data
    # TODO(pablo): fill
