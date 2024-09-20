# (Your)-Transient Auxiliary Library

    bueno el readme ya tal

`y-tal` (or just `tal`) is a Python library with many utilities for people who work in the development of non-line-of-sight imaging techniques. This library provides different tools to generate and analyze data and implementations of different non-line-of-sight reconstruction algorithms. Some parts are also accessible through command line for for ease of use.

Authors: [Diego Royo](https://github.com/diegoroyo), [Pablo Luesia](https://github.com/p-luesia)

# Installation

Working with HDF5 files requires the following packages:

```
sudo apt install libhdf5-dev
```

You will also need the required packages with the included `requirements.txt` file in this repo.

```
pip3 install -r requirements.txt
```

To install `tal` you have the following options:

1) Latest published version (recommended):

```
pip3 install y-tal
```

2) Latest version in GitHub (more features, more unstable):

```
pip3 install git+https://github.com/diegoroyo/tal
```

# Usage:

## Python interface

```python
import tal

data = tal.io.read_capture('capture.hdf5')
tal.plot.xy_interactive(data, cmap='nipy_spectral')
```

## Command line

Some Python functions also have an interface through the command line, including all their parameters:

```
tal plot xy_interactive capture.hdf5 --cmap nipy_spectral
```

```
❯ tal -h
usage: tal [-h] [-v] {config,render,plot} ...

Y-TAL - (Your) Transient Auxiliary Library - v0.10.2

positional arguments:
  {config,render,plot}  Command
    config              Edit the TAL configuration file
    render              Create, edit or execute renders of simulated NLOS scene data captures
    plot                Plot capture data using one of the configured methods

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
```

## `tal render`: Command line tool to render `mitransient` scenes

You will need to have [mitransient](https://github.com/diegoroyo/mitransient) installed in your PC. You can install just via `pip install mitransient` or follow their README for custom installation instructions.

> [!IMPORTANT]
> On the first run of `tal render`, it will ask you if you have installed Mitsuba 2 (mitsuba2-transient-nlos) or 3 (mitransient aka. mitsuba3-transient-nlos), and (only if you have compiled Mitsuba youself) the location of your installation folder: for Mitsuba 2, please use the root folder of the repo (i.e. `/path/to/mitsuba2-transient-nlos`); for Mitsuba 3 you'll have to use the root folder of your custom mitsuba instalation (i.e. `/path/to/mitsuba3`). **If at any time you need to switch from Mitsuba 2 to 3 or vice-versa, or want to switch your installation folder, please use the `tal config` command.**

`mitransient` must be installed in your device. On your first `tal render <scene>` command, it will detect (or ask you) where your installation folder is located, and will execute the necessary `mitsuba` commands and generate the specified scene XML files.

You can find examples for how to render a scene in the [`examples`](https://github.com/diegoroyo/tal/tree/master/examples) folder of this repository. You can always use the `tal render -h` command too:

```
❯ tal render -h
usage: tal render [-h] [-t THREADS] [-s SEED] [-n NICE] [-q] [-r] [--no-steady] [--no-logging]
                  [--no-partial-results]
                  [config_file ...]

positional arguments:
  config_file           Can be:
                        1) Path to a TAL scene config YAML file
                        2) Path to a TAL scene directory (must have a YAML file inside with the same name as the directory)
                        3) 'new <folder_name>' to create a new folder (i.e. tal render new <folder_name>)

optional arguments:
  -h, --help            show this help message and exit
  -t THREADS, --threads THREADS
                        Number of threads
  -s SEED, --seed SEED  Random seed for the sampler. Without setting this value to different values, the
                        same results will be produced everytime.
  -n NICE, --nice NICE  Change +/- in nice factor. Positive values = lower priority. Negative values =
                        higher priority (needs sudo)
  -q, --quiet           Disable progress bars and other verbose outputs
  -r, --dry-run         Do not execute mitsuba, just print out the commands that would be executed
  --no-steady           Disable generation of steady state images
  --no-logging          Disable logging of mitsuba output
  --no-partial-results  Remove the "partial" folder which stores temporal data after creating the final hdf5
                        file (e.g. multiple experiments for confocal/exhaustive)
```

## `tal plot`: Visualize time-resolved capture data

Used to visualize the capture data (maybe generated using `tal render`). It accepts HDF5 files in a format compatible with TAL.

```
❯ tal plot -h
usage: tal plot [-h] [--normalize NORMALIZE] [--opacity OPACITY] [--slider-step SLIDER_STEP]
                [--slider-title SLIDER_TITLE] [--y Y] [--slice-axis SLICE_AXIS] [--t-end T_END]
                [--color COLOR] [--cmap CMAP] [--labels LABELS] [--title TITLE] [--x X] [--a-min A_MIN]
                [--backgroundcolor BACKGROUNDCOLOR] [--size-x SIZE_X] [--t-start T_START] [--t-step T_STEP]
                [--size-y SIZE_Y] [--a-max A_MAX]
                preset [capture_files ...]

positional arguments:
  preset                Plot method. Can be one of:
                            amplitude_phase
                            t_comparison
                            tx_interactive
                            txy_interactive
                            ty_interactive
                            volume
                            xy_grid
                            xy_interactive
  capture_files         One or more paths to capture files

optional arguments:
  -h, --help            show this help message and exit
  --normalize NORMALIZE
  --opacity OPACITY
  --slider-step SLIDER_STEP
  --slider-title SLIDER_TITLE
  --y Y
  --slice-axis SLICE_AXIS
  --t-end T_END
  --color COLOR
  --cmap CMAP
  --labels LABELS
  --title TITLE
  --x X
  --a-min A_MIN
  --backgroundcolor BACKGROUNDCOLOR
  --size-x SIZE_X
  --t-start T_START
  --t-step T_STEP
  --size-y SIZE_Y
  --a-max A_MAX
```

## `tal reconstruct`: Implementation of multiple non-line-of-sight reconstruction algorithms

_NOTE: No command-line version for now_

[You can check the implemented algorithms here](https://github.com/diegoroyo/tal/tree/master/tal/reconstruct). As of Nov. 2023, implemented: backprojection, filtered backprojection, and different phasor-field cameras.

You can find examples for how use the reconstruction algorithms in the [`examples`](https://github.com/diegoroyo/tal/tree/master/examples) folder of this repository. Note that to test the reconstruction algorithms you will need to have a HDF5 capture file. If you don't, please check the `tal render` section or [convert your data to a format usable by `tal`](https://github.com/diegoroyo/tal/blob/master/tal/io/format.py).

### Filtering the impulse response H

```python
import tal

data = tal.io.read_capture('capture.hdf5')
# for more info on the parameters: https://github.com/diegoroyo/tal/blob/master/tal/reconstruct/__init__.py#L25
# NOTE: if you use fbp, pf or pf_dev, you do not need to perform this filtering step
data.H = tal.reconstruct.filter_H(data, filter_name='pf', wl_mean=0.05, wl_sigma=0.05)
```

### Selecting the bounding volume for the reconstruction

```python
import tal
import numpy as np

data = tal.io.read_capture('capture.hdf5')

# Option 1: You can create it manually:
volume_xyz = np.array(...)  # (x, y, z, 3) or (x, y, 3) or (n, 3) shape
# Option 2: Create a volume from two points and a scalar resolution:
volume_xyz = tal.reconstruct.get_volume_min_max_resolution(minimal_pos=np.array([-3, -2, -1]), maximal_pos=np.array([3, 2, 1]), resolution=0.01)
print(volume_xyz.shape)  # (600, 400, 200, 3)
# Option 3: Create a volume coplanar to the relay wall, displaced by a distance d
volume_xyz = tal.reconstruct.get_volume_project_rw(data, depths=[1.0, 1.5, 2.0, 2.5, 3.0])
print(volume_xyz.shape)  # (sx, sy, 5, 3) where sx, sy are SPAD scan dimensions on X and Y axes
```

You can now use `volume_xyz` to specify the reconstruction volume for the `bp`, `fbp` or `pf_dev` reconstruction methods.

### `bp`

Implementation of the backprojection algorithm [Velten2012].

```python
# follow steps above to read data and obtain volume_xyz

H_1 = tal.reconstruct.bp.solve(data,
                               volume_xyz=volume_xyz,
                               camera_system=tal.enums.CameraSystem.DIRECT_LIGHT)
# the camera_system parameter specifies the concrete camera implementation in the phasor-field framework
# by default most papers use the DIRECT_LIGHT equivalent so you probably want to leave it as-is

# visualize your result
tal.plot.amplitude_phase(H_1)                              
```

### `fbp`, `pf_dev`

Filtered backprojection [Velten2012] and the `pf_dev` implementation [Liu2019] of phasor fields accept the same arguments.

```python
# follow steps above to read data and obtain volume_xyz

# you can switch pf_dev and fbp interchangeably
H_1 = tal.reconstruct.pf_dev.solve(data,
                                   wl_mean=0.06, wl_sigma=0.06,
                                   volume_xyz=volume_xyz,
                                   camera_system=tal.enums.CameraSystem.DIRECT_LIGHT)
H_1 = tal.reconstruct.fbp.solve(data,
                                   wl_mean=0.06, wl_sigma=0.06,
                                   volume_xyz=volume_xyz,
                                   camera_system=tal.enums.CameraSystem.DIRECT_LIGHT)
# the wl_mean and wl_sigma parameters set the band pass filter that is the phasor-field-based filter
# the camera_system parameter specifies the concrete camera implementation in the phasor-field framework
# by default most papers use the DIRECT_LIGHT equivalent so you probably want to leave it as-is

# visualize your result
tal.plot.amplitude_phase(H_1)                              
```

### `pf`

An implementation of phasor-field cameras [Liu2019]. See also `pf_dev`.

```python
import tal

data = tal.io.read_capture('capture.hdf5')
V = np.moveaxis(np.mgrid[-1:1.1:0.1, -1:1.1:0.1, 0.5:2.6:0.1], 0, -1).reshape(-1,3)
# Reconstruct the data to the volume V with virtual illumination pulse
# with central wavefactor 6 and 4 cycles
reconstruction = tal.reconstruct.pf.solve(data, 6, 4, V, verbose=3, n_threads=1)
```

### Logging

The verbosity of the output can be controlled through `tal.set_log_level(level)`.
For the available values of `level`, see `tal.LogLevel`. For example, if you only wish to see
progress bars, warnings and more you can use:

```python
import tal

tal.set_log_level(tal.LogLevel.PROGRESS)
# ...rest of your code...
```

Your choice of logging level is also stored in the configuration file, so it is kept between executions.
It also can be changed using `tal config` in the command line.

### `tal.resources`

The `filter_H`, `bp`, `fbp` and `pf_dev` functions/modules support multi-threading.
The number of threads can be set using:

```python
import tal

# Option 1: Scoped
with tal.resources('max'):  # all CPUs
  # ...work...

# Option 2: Set
tal.set_resources(4)  # use 4 CPUs
# ...work...
```

### References

> [Velten2012] Velten, A., Willwacher, T., Gupta, O., Veeraraghavan, A., Bawendi, M. G., & Raskar, R. (2012). Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging. Nature communications, 3(1), 745.

> [Liu2019] Liu, X., Guillén, I., La Manna, M., Nam, J. H., Reza, S. A., Huu Le, T., ... & Velten, A. (2019). Non-line-of-sight imaging using phasor-field virtual wave optics. Nature, 572(7771), 620-623.

### License and citation

`tal` is licensed under the GPL-3.0 license. If you use `tal` in an academic work, we would appreciate if you cited our work:

```
@software{Royo_y-tal,
author = {Royo, Diego and Luesia-Lahoz, Pablo},
license = {GPL-3.0},
title = {{y-tal}},
url = {https://github.com/diegoroyo/tal},
publisher = {GitHub},
doi = {https://doi.org/10.5281/zenodo.11197745},
}
```
