from tal.io.capture_data import NLOSCaptureData
from typing import Union
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button
import numpy as np
from tqdm import tqdm
import pyvista as pv


def plot3d(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
           title: str = '', slider_title: str = 'Time',
           slider_step: float = 0.1, cmap='hot',
           opacity='sigmoid', backgroundcolor=None):
    """
    Plot in 3d the data indicated, with a slider if the data has a four 
    dimension
    @param data             : The data to plot. has to be 3 o 4 d
    @param title            : Title of the figure to plot. Default empty
    @param slider_tittle    : Title to print under the slider. Default "Time"
    @param cmap             : Colormap to print the image. Defalut hot
    @param opacity          : Opacity for the image. Default sigmoid
    @param backgroundcolor  : Color of the background
    """
    data_to_plot = data
    if isinstance(data, NLOSCaptureData):
        data_to_plot = data.H

    assert data_to_plot.ndim >= 3 and data_to_plot.ndim <= 4,\
        "Expected 3 or 4 dimension data,"\
        + f" but data has dimension {data_to_plot.ndim}"

    p = pv.Plotter()
    p.add_title(title)
    if backgroundcolor is not None:
        p.background_color = backgroundcolor

    # Normalize the data so it is possible to plot it with the plotter
    norm_data_to_plot = data_to_plot * 1000 / np.max(data_to_plot)

    # Plot with slider if the data has more than 3 dimension
    if norm_data_to_plot.ndim == 4:
        def volume_by_time(slider_value):
            idx = int(slider_value/slider_step)
            p.add_volume(norm_data_to_plot[idx],
                         clim=np.array([0, np.max(data_to_plot)]),
                         cmap=cmap,
                         opacity=opacity,
                         name='data_by_time')

        p.add_slider_widget(volume_by_time,
                            [0, (norm_data_to_plot.shape[0]-1) * slider_step],
                            value=0,
                            title=slider_title)
    else:  # 3 dim
        p.add_volume(norm_data_to_plot,
                     clim=np.array([0, np.max(data_to_plot)]),
                     cmap=cmap,
                     opacity=opacity,
                     name='full_data')

    # Select the initial camera
    p.camera_position = 'yz'
    p.camera.azimuth -= 15
    p.camera.elevation += 30
    # Plot the image
    p.show()
