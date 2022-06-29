from nptyping import NDArray
from tal.io.capture_data import NLOSCaptureData
from tal.io.enums import HFormat
from tal.util import SPEED_OF_LIGHT
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from tqdm import tqdm
from enum import Enum

class ByAxis(Enum):
    UNKOWN = 0
    T = 1
    X = 2
    Y = 3
    Z = 4


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

# More general plotting
def plot_3d_interactive_axis(xyz: np.ndarray, focus_slider: np.ndarray,
                                axis: int, title: str, slider_title: str,
                                slider_unit: str, cmap: str = 'hot',
                                xlabel: str = '', ylabel: str = ''):
    assert xyz.ndim == 3, 'Unknown datatype to plot'
    assert axis < 3, f'Data only have 3 dims (given axis={axis})'
    assert xyz.shape[axis] == len(focus_slider), \
            'The slider and the data have different lengths'
    # Move the axis, so the interactive axis is at 0
    xyz_p = np.moveaxis(xyz, axis, 0)
    v_min = np.min(xyz); v_max = np.max(xyz)

    # Plot the first figure
    fig = plt.figure()
    fig.suptitle(title)

    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    img = ax.imshow( xyz_p[0], cmap = cmap, vmin = v_min, vmax = v_max)
    fig.colorbar(img, ax=ax, shrink = 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider = Slider(
            ax = ax_slider,
            label = slider_title,
            valmin = 0,
            valmax = len(focus_slider) - 1,
            valinit = 0,
            valstep = 1.0,
            orientation='horizontal' )
    ax_text = plt.axes([0.25, 0.05, 0.5, 0.03])
    ax_text.axis('off')
    text = ax_text.text(0, 0, f'{focus_slider[0]} {slider_unit}')

    # Figure update
    def update(i):
        idx = int(i)
        img.set_array(xyz_p[idx])
        text.set_text(f'{focus_slider[idx]} {slider_unit}')

    slider.on_changed(update)
    plt.show()


def plot_txy_interactive(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
                         cmap:str = 'hot', by: ByAxis = ByAxis.T):
    if isinstance(data, NLOSCaptureData):
        assert data.H_format == HFormat.T_Sx_Sy, \
            'plot_txy_interactive does not support this data format'
        txy = data.H
        delta_t = data.delta_t
    else:
        assert data.ndim == 3 and data.shape[1] == data.shape[2], \
            'plot_txy_interactive does not support this data format'
        txy = data
        delta_t = None
    
    title = 'Impulse response '
    # Plot parameters
    if by == ByAxis.T:
        axis = 0
        n_it = txy.shape[axis]
        it_v = np.arange(n_it, dtype = np.float32)
        title += 'by time'; slider_title = 'Bins'; slider_unit = 'index'
        if delta_t is not None: it_v*=delta_t; slider_unit = 'ps'
        xlabel = 'x'; ylabel = 'y'
    elif by == ByAxis.Y:
        axis = 1
        n_it = txy.shape[axis]
        it_v = np.arange(n_it, dtype = np.float32)
        title += 'by y'; slider_title = 'Planes'; slider_unit = 'index'
        txy = txy.swapaxes(0,2)
        xlabel = 't'; ylabel = 'x'
    elif by == ByAxis.X:
        axis = 2
        n_it = txy.shape[axis]
        it_v = np.arange(n_it, dtype = np.float32)
        title += 'by x'; slider_title = 'Planes'; slider_unit = 'index'
        txy = txy.swapaxes(0,1)
        xlabel = 't'; ylabel = 'y'
    else:
        raise 'plot_txy_interactive does not support this axis'

    # Plot the data
    return plot_3d_interactive_axis(txy, it_v, axis = axis, 
                                title = title,
                                slider_title = slider_title,
                                slider_unit = slider_unit,
                                cmap = cmap,
                                xlabel = xlabel, ylabel = ylabel)


def plot_zxy_interactive(data: NDArray, cmap:str = 'hot', by: ByAxis = ByAxis.Z):

    assert data.ndim == 3 and data.shape[1] == data.shape[2], \
        'plot_zxy_interactive does not support this data format'
    zxy = data
    
    title = 'Amplitude by '
    # Plot parameters
    if by == ByAxis.Z:
        axis = 0
        n_it = zxy.shape[axis]
        it_v = np.arange(n_it, dtype = np.float32)
        title += 'by z'; slider_title = 'Planes'; slider_unit = 'index'
        xlabel = 'x'; ylabel = 'y'
    elif by == ByAxis.Y:
        axis = 1
        n_it = zxy.shape[axis]
        it_v = np.arange(n_it, dtype = np.float32)
        title += 'by y'; slider_title = 'Planes'; slider_unit = 'index'
        xlabel = 'x'; ylabel = 'z'
    elif by == ByAxis.X:
        axis = 2
        n_it = zxy.shape[axis]
        it_v = np.arange(n_it, dtype = np.float32)
        title += 'by x'; slider_title = 'Planes'; slider_unit = 'index'
        xlabel = 'y'; ylabel = 'z'
    else:
        raise 'plot_zxy_interactive does not support this axis'

    # Plot the data
    return plot_3d_interactive_axis(zxy, it_v, axis = axis, 
                                title = title,
                                slider_title = slider_title,
                                slider_unit = slider_unit,
                                cmap = cmap,
                                xlabel = xlabel, ylabel = ylabel)