# Examples

> [!WARNING]
> Running the `reconstruct` tutorials requires that you have run the corresponding `render` tutorials beforehand (to generate files), or that you have downloaded the corresponding capture `hdf5` files from here: TODO. Check the `render` tutorials first, then move to the `reconstruct` tutorials.

> [!WARNING]
> Our code is mostly designed to work with files that have also been rendered with `y-tal`, however we also support some other file formats (e.g., capture files from the phasor field/phasor field diffraction papers). To see if we support your capture files, check the `tal.io.read_capture` function (try to read it, see if it works, and if it does not, feel free to write a pull request for it, you'll need to write your conversion function in `tal.io.format.py`)

We provide the following tutorials:

* `render`: Tutorials on how to simulate a NLOS scene using `y-tal`, choosing parameters in the YAML file.
* `reconstruct` Tutorials on how to use our reconstruction algorithms.