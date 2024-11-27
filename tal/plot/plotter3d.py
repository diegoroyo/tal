
from tal.io.capture_data import NLOSCaptureData
from typing import Union
import numpy as np
import pyvista as pv


def plot3d(data: Union[NLOSCaptureData, NLOSCaptureData.HType],
           title: str = '', slider_title: str = 'Time',
           slider_step: float = 0.1, color='hot',
           opacity='linear', backgroundcolor=None):
    """
    Plot in 3d the data indicated, with a slider if the data has a four 
    dimension
    @param data             : The data to plot. has to be 3 o 4 d
    @param title            : Title of the figure to plot. Default empty
    @param slider_tittle    : Title to print under the slider. Default "Time"
    @param color            : If string, it use that colormap to plot the 
                              image. It can be a 2d array of shape (N, 4), where
                              N are the number of points. Defalut "hot"
    @param opacity          : Opacity for the image. This argument is ignored
                              if color is an array.
    @param backgroundcolor  : Color of the background
    """
    data_to_plot = data
    if isinstance(data, NLOSCaptureData):
        data_to_plot = data.H

    assert data_to_plot.ndim >= 3 and data_to_plot.ndim <= 4, \
        "Expected 3 or 4 dimension data,"\
        + f" but data has dimension {data_to_plot.ndim}"

    p = pv.Plotter()
    p.add_title(title)
    if backgroundcolor is not None:
        p.background_color = backgroundcolor

    if isinstance(color, np.ndarray):
        cmap = None
        scalars = color
        if scalars.ndim > 2:
            scalars = scalars.reshape(-1, 4)
        mapper = 'gpu'
    else:
        scalars = None
        cmap = color
        mapper = None

    # Normalize the data so it is possible to plot it with the plotter
    norm_data_to_plot = (data_to_plot - np.min(data_to_plot)) * 255 \
        / (np.max(data_to_plot) - np.min(data_to_plot))

    # Plot with slider if the data has more than 3 dimension
    if norm_data_to_plot.ndim == 4:
        assert cmap is not None, "Only colormaps can be used with a slider"

        def volume_by_time(slider_value):
            idx = int(slider_value/slider_step)
            p.add_volume(norm_data_to_plot[idx],
                         clim=[0, 255],
                         scalars=scalars,
                         cmap=cmap,
                         opacity=opacity,
                         mapper = mapper,
                         name='data_by_time')

        p.add_slider_widget(volume_by_time,
                            [0, (norm_data_to_plot.shape[0]-1) * slider_step],
                            value=0,
                            title=slider_title)
    else:  # 3 dim
        p.add_volume(norm_data_to_plot,
                     clim=[0, 255],
                     scalars=scalars,
                     cmap=cmap,
                     opacity=opacity,
                     mapper = mapper,
                     name='full_data')

    # Select the initial camera
    p.camera_position = 'yz'
    p.camera.azimuth -= 15
    p.camera.elevation += 30
    # Plot the image
    p.show()
