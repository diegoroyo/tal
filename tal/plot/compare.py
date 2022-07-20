from tal.io.capture_data import NLOSCaptureData
import matplotlib.pyplot as plt
import numpy as np
import tal
from matplotlib.widgets import Slider, RangeSlider


def plot_t_comparison(data_list, x, y, t_start, t_end, a_min, a_max, labels):
    if isinstance(data_list, np.ndarray) or isinstance(data_list, NLOSCaptureData):
        data_list = [data_list]
    assert all(isinstance(data, np.ndarray) or
               isinstance(data, NLOSCaptureData) for data in data_list), \
        'Incorrect data types'

    def get_H(data):
        if isinstance(data, NLOSCaptureData):
            return data.H
        else:
            return data

    nt, ny, nx = get_H(data_list[0]).shape
    for data in data_list[1:]:
        nt2, ny2, nx2 = get_H(data).shape
        assert nt == nt2 and ny == ny2 and nx == nx2, \
            'Dimensions do not match'

    fig = plt.figure()
    fig.suptitle('Signal comparison')

    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    x_slider_ax = plt.axes([0.25, 0.13, 0.5, 0.03])
    y_slider_ax = plt.axes([0.25, 0.10, 0.5, 0.03])
    t_slider_ax = plt.axes([0.25, 0.07, 0.5, 0.03])
    a_slider_ax = plt.axes([0.25, 0.04, 0.5, 0.03])

    x = x or 0
    y = y or 0
    t_start = t_start or 0
    t_end = t_end or nt - 1
    A_MAX_TOTAL = max(np.max(get_H(data)) for data in data_list)
    a_min = a_min or 0.0
    a_max = a_max or A_MAX_TOTAL

    def update():
        ax.cla()
        x_range = list(range(t_start, t_end, 1))
        for i, data in enumerate(data_list):
            ax.plot(x_range, get_H(data)[t_start:t_end, x, y],
                    label=str(i) if labels is None else labels[i])
            ax.set_ylim(bottom=a_min, top=a_max)
            ax.legend()
        plt.draw()

    x_slider = Slider(
        ax=x_slider_ax, label='x', valmin=0, valmax=nx - 1,
        valinit=x, valstep=1, orientation='horizontal')
    y_slider = Slider(
        ax=y_slider_ax, label='y', valmin=0, valmax=ny - 1,
        valinit=y, valstep=1, orientation='horizontal')
    t_slider = RangeSlider(
        ax=t_slider_ax, label='t', valmin=0, valmax=nt - 1,
        valinit=(t_start, t_end), valstep=1, orientation='horizontal')
    a_slider = RangeSlider(
        ax=a_slider_ax, label='Amplitude', valmin=0, valmax=A_MAX_TOTAL,
        valinit=(a_min, a_max), orientation='horizontal')

    def update_x(new_x):
        nonlocal x
        x = new_x
        update()

    def update_y(new_y):
        nonlocal y
        y = new_y
        update()

    def update_t(new_t):
        nonlocal t_start, t_end
        t_start, t_end = new_t
        update()

    def update_a(new_a):
        nonlocal a_min, a_max
        a_min, a_max = new_a
        update()

    x_slider.on_changed(update_x)
    y_slider.on_changed(update_y)
    t_slider.on_changed(update_t)
    a_slider.on_changed(update_a)
    update()
    plt.show()
