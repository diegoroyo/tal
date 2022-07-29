# (Your)-Transient Auxiliary Library

    bueno el readme ya tal

Utilities library for our work in the development of non-line-of-sight imaging techniques. See [mitsuba2-transient-nlos](https://github.com/diegoroyo/mitsuba2-transient-nlos) for data capture simulation. This library provides different tools to generate and analyze data and implementations of non-line-of-sight reconstruction algorithms. Some parts are also accessible through command line for for ease of use.

Authors: [Diego Royo](https://github.com/diegoroyo), [Pablo Luesia](https://github.com/p-luesia)

## Installation

Latest published version (recommended):

```
pip3 install y-tal
```

Latest version in GitHub (more features, more unstable):

```
pip3 install git+https://github.com/diegoroyo/tal
```

You will need the required packages with the included `requirements.txt` file in this repo.

```
pip3 install -r requirements.txt
```

Usage (python):

```python
import tal

data = tal.io.read_capture('capture.hdf5')
tal.plot.xy_interactive(data, cmap='nipy_spectral')
```

Usage (command line):

```
tal plot xy_interactive capture.hdf5 --cmap nipy_spectral
```

```
❯ tal -h
usage: tal [-h] {render,plot} ...

TAL - Transient Auxiliary Library

positional arguments:
  {render,plot}  Command
    render       Create, edit or execute renders of simulated NLOS scene data captures
    plot         Plot capture data using one of the configured methods

optional arguments:
  -h, --help     show this help message and exit
```

## `tal render`: Command line tool to render mitsuba2-transient-nlos scenes

See: [mitsuba2-transient-nlos](https://github.com/diegoroyo/mitsuba2-transient-nlos) repository.

`mitsuba2-transient-nlos` must be installed in your device. On your first `tal render <scene>` comamnd, it will ask you where your installation folder is located, and will execute the necessary `mitsuba` commands and generate the specified scene XML files.

```
❯ tal render -h
usage: tal render [-h] [-t THREADS] [-n NICE] [-q] [-r] [--no-steady] [--no-logging] [--no-partial-results] [config_file ...]

positional arguments:
  config_file           Can be:
                        1) Path to a TAL scene config YAML file
                        2) Path to a TAL scene directory (must have a YAML file inside with the same name as the directory)
                        3) 'new <folder_name>' to create a new folder (i.e. tal render new <folder_name>)

optional arguments:
  -h, --help            show this help message and exit
  -t THREADS, --threads THREADS
                        Number of threads
  -n NICE, --nice NICE  Change +/- in nice factor. Positive values = lower priority. Negative values = higher priority (needs sudo)
  -q, --quiet           Disable progress bars and other verbose outputs
  -r, --dry-run         Do not execute mitsuba, just print out the commands that would be executed
  --no-steady           Disable generation of steady state images
  --no-logging          Disable logging of mitsuba2 output
  --no-partial-results  Remove the "partial" folder which stores temporal data after creating the final hdf5 file (e.g. multiple experiments for confocal/exhaustive)
```

## `tal plot`: Visualize time-resolved render data

Used to visualize the renders generated using `tal render` i.e. HDF5 files in TAL format.

```
❯ tal plot -h
usage: tal plot [-h] [--t-end T_END] [--t-start T_START] [--y Y] [--size-x SIZE_X] [--x X] [--a-min A_MIN] [--labels LABELS] [--t-step T_STEP] [--cmap CMAP] [--size-y SIZE_Y]
                [--a-max A_MAX]
                preset [capture_files ...]

positional arguments:
  preset             Plot method. Can be one of:
                         t_comparison
                         tx_interactive
                         ty_interactive
                         xy_grid
                         xy_interactive
  capture_files      One or more paths to capture files

optional arguments:
  -h, --help         show this help message and exit
  --t-end T_END
  --t-start T_START
  --y Y
  --size-x SIZE_X
  --x X
  --a-min A_MIN
  --labels LABELS
  --t-step T_STEP
  --cmap CMAP
  --size-y SIZE_Y
  --a-max A_MAX
```

## `tal reconstruct`: Non-line-of-sight reconstruction algorithms implementation

_NOTE: No command-line version for now_

```python
import tal

data = tal.io.read_capture('capture.hdf5')
V = np.moveaxis(np.mgrid[-1:1.1:0.1, -1:1.1:0.1, 0.5:2.6:0.1], 0, -1).reshape(-1,3)
# Reconstruct the data to the volume V with virtual illumination pulse
# with central wavefactor 6 and 4 cycles
reconstruction = tal.reconstruct.pf.solve(data, 6, 4, V, verbose=3, n_threads=1)
```
