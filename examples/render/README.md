# `tal render` tutorials

> [!TIP]
> All tutorials here assume that you have read the first one (Quickstart) just below, only focusing on the things that change with respect to that first tutorial.

- [Quickstart](#quickstart)
- [GPU rendering](#gpu-rendering)
- [Acquisition modalities: single/confocal/exhaustive captures](#acquisition-modalities-singleconfocalexhaustive-captures)
  - [Confocal capture](#confocal-capture)
  - [Exhaustive capture](#exhaustive-capture)
    - [Bunny example](#bunny-example)
    - [Plane example](#plane-example)
- [Rendering in frequency space](#rendering-in-frequency-space)

## Quickstart

Start with `0-z-quickstart`. Each render has two key parts: (i) a parent folder (in this case, `z-single-64x64`), and a YAML file which configures the render, with the same name as the folder. In this case we want to simulate the capture of a Z-shaped object, located at 1 meter from the relay wall, like on the diagram below:

![Z preview](meshes/nlos-z-simple-preview.png)

We will simulate only one laser position at the very center of the relay wall, and capture the time-resolved response at a 64x64 grid of points evenly spaced throughout the relay wall. Now, you can open the `z-single-64x64.yaml` file and read a more thorough explanation of what each parameter does. Look for the commented lines that start with `# *` which will give extra info. You can ignore most of the parameters of the YAML file at this point, some are for advanced topics.

After reading, let's launch the render. Navigate to the folder:
```
cd 0-z-quickstart
```
Then launch the render with `y-tal` as:
```
tal render z-single-64x64
```
This will generate a folder inside `z-single-64x64` with format `YYYYMMDD-HHMMSS`. Inside there are many files, but the most important one is `z-single-64x64.hdf5`, which contains the output file. That's all that's needed to do a render! Now you can use that file on the `examples/reconstruct` tutorials.

## GPU rendering

Rendering on GPU is much faster. For smaller examples it may not matter a lot, but for large capture grids/exhaustive configurations it really speeds things up.

You can find this example on `0-z-quickstart/z-single-64x64-gpu`. If you look at the `z-single-64x64.yaml`, you can find this line:
```yaml
# * See https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html
# * you should have compiled this variant previously
# * See the mitransient repository for more information
mitsuba_variant: llvm_ad_mono
```
The `llvm_*` variants execute on CPU. If you want GPU execution, you should change two things. First, switch to `cuda_*`:
```diff
- mitsuba_variant: llvm_ad_mono
+ mitsuba_variant: cuda_ad_mono
```
And then, when you run `tal`, specify the ID (0, 1, 2, ...) of the GPU that you want to use. In case your system only has one GPU, choose 0:
```
tal render z-single-64x64-gpu -g 0
```

## Acquisition modalities: single/confocal/exhaustive captures

Previously we used 'single' capture as depicted in the figure below. The X represent capture points, and the dots represent illumination points on the relay wall. Now let's look on what we need to do for confocal and exhaustive modalities:

![Z preview](meshes/capture-configurations.png)

### Confocal capture

You can find this example on `1-bunny-confocal/bunny-confocal-128x128`. The main changes are:
```diff
- scan_type: single  # single, confocal or exhaustive
+ scan_type: confocal  # single, confocal or exhaustive
sensor_width: 64
sensor_height: 64
# note these should match sensor_width and sensor_height
- # laser_width: 32
- # laser_height: 32
+ laser_width: 64
+ laser_height: 64

# ...

mesh:
  type: obj
- filename: ../meshes/Z.obj
+ filename: ../meshes/bunny-single-1m.obj
# ...
- displacement_z: 1.0  # using default RW settings this corresponds to depth
+ # *** The bunny is already displaced in the OBJ file, so we don't need to displace it here. 
+ displacement_z: 0.0  # using default RW settings this corresponds to depth
```
Then you can `cd 1-bunny-confocal` (really important) and then you can run (takes ~2-4min):
```
tal render bunny-confocal-128x128
```
Which should generate a `bunny-confocal-128x128.hdf5` file.

> [!INFO]
> Make sure you have `mitransient>=1.3.0` and `y-tal>=0.21.0`, which will greatly speed up render times thanks to the `simultaneous_scan` option that becomes available in the YAML file. Make sure that it's set to `true`, else your renders will take a looong time.

### Exhaustive capture

We provide two examples: one with a bunny, and another with a planar hidden object. We'll use both later.

#### Bunny example

You can find this example on `2-exhaustive/bunny-exhaustive-2x2x64x64`. The main changes are:
```diff
# Used only for scan_type=exhaustive, ignored otherwise
- # integrator_force_equal_illumination_scanning: true
+ integrator_force_equal_illumination_scanning: false
integrator_illumination_scan_fov: 20  # UNUSED

# IMPORTANT: we need to set simultaneous_scan = false, we want to illuminate points one by one.
# Otherwise mitransient will use the illumination_scan_fov strategy
- # simultaneous_scan: true
+ simultaneous_scan: false

- scan_type: single  # single, confocal or exhaustive
+ scan_type: exhaustive  # single, confocal or exhaustive
sensor_width: 64
sensor_height: 64
- # laser_width: 32
- # laser_height: 32
+ laser_width: 2
+ laser_height: 2
# ...
# We also modify a bit the area covered by the lasers.
# These numbers go from [0, 1] where 0 represents the top left part of the relay wall,
# and 1 represents the bottom right part of the relay wall.
# So this just adds a tiny bit of padding, and the laser points are
# evenly spaced inside that padded area 
- # laser_aperture_start_x: 0.0  # 0: left, 1: right of relay wall
- # laser_aperture_start_y: 0.0  # 0: top, 1: bottom of relay wall
- # laser_aperture_end_x: 1.0
- # laser_aperture_end_y: 1.0
+ laser_aperture_start_x: 0.1  # 0: left, 1: right of relay wall
+ laser_aperture_start_y: 0.1  # 0: top, 1: bottom of relay wall
+ laser_aperture_end_x: 0.9
+ laser_aperture_end_y: 0.9

# ...

mesh:
  type: obj
- filename: ../meshes/Z.obj
+ filename: ../meshes/bunny-single-1m.obj
# ...
- displacement_z: 1.0  # using default RW settings this corresponds to depth
+ # *** The bunny is already displaced in the OBJ file, so we don't need to displace it here. 
+ displacement_z: 0.0  # using default RW settings this corresponds to depth
```
Then you can `cd 2-exhaustive` (really important) and then you can run (takes ~2-4min):
```
tal render bunny-confocal-2x2x32x32
```
Which should generate a `bunny-confocal-2x2x32x32.hdf5` file.


#### Plane example

You can find this example on `2-exhaustive/plane-exhaustive-16x16x16x16`. The main changes are:
```diff
# Redundant with `laser_{width|height}`
integrator_force_equal_illumination_scanning: true
# integrator_illumination_scan_fov: 20  # UNUSED
# ...
# Important for speed
simultaneous_scan: true

- scan_type: single  # single, confocal or exhaustive
+ scan_type: exhaustive  # single, confocal or exhaustive
- sensor_width: 64
- sensor_height: 64
+ sensor_width: 16
+ sensor_height: 16
- # laser_width: 32
- # laser_height: 32
+ laser_width: 16  # redundant with `integrator_force_equal_illumination_scanning`
+ laser_height: 16

# ...

mesh:
  type: obj
- filename: ../meshes/Z.obj
+ filename: ../meshes/wall-2m-z0.obj
# ...
displacement_z: 1.0  # using default RW settings this corresponds to depth
```
Then you can `cd 2-exhaustive` (really important) and then you can run (takes ~30s-1min):
```
tal render plane-exhaustive-16x16x16x16
```
Which should generate a `plane-exhaustive-16x16x16x16.hdf5` file.


## Rendering in frequency space

We also provide the option to render directly in frequency space (i.e., instead of simulating the full temporal response I(t), we generate the Fourier transform of that response F{I}(w) for only a subset of frequencies w, which uses less memory). This is especially useful in memory-constrained applications, although it makes things a bit harder to work with.

Since this is designed to work with the phasor field reconstruction method, the frequencies w are defined by the `wl_mean` and `wl_sigma` parameters of the phasor field filtering function.

```diff
- histogram_mode: time  # time or frequency
+ histogram_mode: frequency  # time or frequency
# --- for histogram_mode=frequency
- wl_mean: null
- wl_sigma: null
+ wl_mean: 0.06
+ wl_sigma: 0.06
```

Finally we can launch the render with:

```
tal render z-single-64x64-freq
```

Which will generate a `z-single-64x64-freq.hdf5` file.