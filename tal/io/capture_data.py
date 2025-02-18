from enum import Enum
import os
import h5py
import yaml
import numpy as np
import inspect
from typing import Union, get_type_hints
from nptyping import NDArray, Shape
from tal.io.format import convert_dict, detect_dict_format
from tal.enums import FileFormat, HFormat, GridFormat, VolumeFormat
from tal.log import log, LogLevel


class LazyDataset:
    def __init__(self, dataset):
        self.read = False
        self.dataset = dataset

    def __get__(self, instance, owner):
        if not self.read:
            self.dataset = self.dataset[()]
            self.read = True
        return self.dataset

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def read_hdf5(filename: str, skip_H: bool = False) -> dict:
    raw_data = h5py.File(filename, 'r')

    def parse(key, value):
        if skip_H and key == 'H':
            value = None
        elif isinstance(value, h5py.Empty):
            value = None
        elif isinstance(value, h5py.Group):
            value = {k: parse(k, v) for k, v in value.items()}
        else:
            if isinstance(value, h5py.Dataset):
                value = value[()]
                # FIXME(diego): unused, need to figure out how
                # to call do stuff like value.size or
                # isinstance(value, bytes) without reading the
                # whole dataset
                # value = LazyDataset(value)
            if isinstance(value, np.ndarray) and value.size == 1:
                value = np.squeeze(value)[()]

        # parse enums and strings
        if key in get_type_hints(NLOSCaptureData):
            key_class = get_type_hints(NLOSCaptureData)[key]
            if inspect.isclass(key_class) and issubclass(key_class, Enum):
                if isinstance(value, h5py.Empty):
                    value = key_class(0)
                else:
                    value = key_class(value)
        elif isinstance(value, bytes):
            # FIXME(diego): not working?
            value = yaml.safe_load(value)
        return value

    raw_data = dict((key, parse(key, raw_data.get(key)))
                    for key in raw_data.keys())
    return raw_data


def write_hdf5(filename: str, capture_data: dict, compression_level: int = False):
    assert 0 <= compression_level <= 9, 'Compression level must be an integer between 0 and 9'
    file = h5py.File(filename, 'w')
    for key, value in capture_data.items():
        if value is None:
            file[key] = h5py.Empty(float)
        elif isinstance(value, dict):
            file[key] = yaml.dump(value, Dumper=yaml.CDumper)
        elif isinstance(value, Enum):
            dt = h5py.enum_dtype(dict((item.name, item.value)
                                 for item in value.__class__), basetype='i')
            ds = file.create_dataset(key, (1,), dtype=dt)
            ds[0] = value.value
        elif compression_level > 0 and key == 'H':
            file.create_dataset(key, data=value, compression='gzip', compression_opts=compression_level)
        else:
            file[key] = value
    file.close()


class NLOSCaptureData:
    """
    Contains data for a NLOS capture scene. Typically, a laser illuminates a planar wall with a delta light pulse.
    An ultrafast camera captures the reflected light at that wall.

    The scene is described by the following parameters:

    H
        Multi-dimensional (>=2 dimensions) impulse response function.
        Contains time-resolved illumination for different laser and sensor positions.
        See H_format for the number of dimensions and what each one contains.

    H_format
        Enum describing what each dimension contains.
        e.g. HFormat.T_Lx_Ly_Sx_Sy means that there is temporal (T) data for a 2D grid
             of laser positions (Lx, Ly) and a 2D grid of sensor positions (Sx, Sy)
        e.g. HFormat.T_Sx_Sy means that there is temporal (T) data for a 2D grid of sensor positions.
             In this case, see is_confocal() to differentiate between the case of only one laser point
             and the case of a 2D grid of laser positions co-located with the sensor positions.
             You can also check sensor_grid_xyz and laser_grid_xyz, but is_confocal() is easier.

    sensor_xyz, laser_xyz
        3D coordinates of the sensor and laser positions.
        Note that it is the origin of the laser itself, and _not_ the point where the laser is focused at.

    sensor_grid_xyz, laser_grid_xyz
        Positions of laser and sensor points at the wall.
        See {sensor|laser}_grid_format for the number of dimensions.

    sensor_grid_normals, laser_grid_normals
        Normal vectors of the laser and sensor points at the wall.
        See {sensor|laser}_grid_format for the number of dimensions.

    sensor_grid_format, laser_grid_format
        Enum describing what each dimension of {sensor|laser}_grid_{xyz|normals} contains.
        See tal.enums.GridFormat for details.

    delta_t
        Temporal resolution of H i.e. distance between two time bins.
        Measured in meters (see https://en.wikipedia.org/wiki/Optical_path_length).

    t_start
        Initial instant of H, used to define a different time origin.
        Measured in meters (see https://en.wikipedia.org/wiki/Optical_path_length).

    t_accounts_first_and_last_bounces
        Whether the impulse response accounts for the time-of-flight between
        the laser origin and the wall (first bounce),
        and the wall and the sensor origin (last bounce).

    scene_info
        YAML-encoded string. Contains additional information about the scene. Implemented keys:
        - 'original_format': str (e.g. 'HDF5_ZNLOS' if imported from ZNLOS dataset instead of tal render)
        - 'config': dict (original scene_config.yaml when generated using tal render)
        - 'args': dict (original args passed when generated using tal render)
        - 'volume': dict (if available e.g. for 'HDF5_ZNLOS' datasets, *not for tal render*)
            - 'center': Array3 (center of volume)
            - 'rotation': Array3 (rotation of volume formatted as per Z-NLOS - probably unused for now)
            - 'size': Array3 (size of volume - length of each XYZ side)
            - 'xyz': MatrixN3 (if ever volume_format is VolumeFormat.N_3, points will be stored here)
        - 'ground_truth': dict (if available, e.g. generated by tal render).
            Remember that TAL uses a XYZ coordinate system where the relay wall is a plane at Z = 0,
            and Z > 0 is the depth axis away from the relay wall.
            All ground truth values use (X, Y) coordinates for the relay wall plane
            - 'format': GroundTruthFormat (for now it is GroundTruthFormat.X_Y always)
            - 'depth': GroundTruthDepthType (TensorXY)
                Example: depth[i, j] stores the distance to the hidden geometry,
                    from the relay wall to the hit point.
                It has size 1024 x 1024. depth[0, 0] corresponds to the top-left
                    of the relay wall, depth[-1, -1] corresponds to the bottom-right of the relay wall.
                If it does not hit the hidden geometry, depth[i, j] = -1
                If it hits, depth[i, j] stores the distance from the relay wall to the hit point.
            - 'normals': GroundTruthNormalsType (TensorXY3)
                Similar to depth, but stores the normal of the hidden geometry at the hit point.
    """

    #
    # Type aliases
    #
    Float = np.float32
    TensorTSxSy = NDArray[Shape['T, Sx, Sy'], Float]
    TensorTLxLySxSy = NDArray[Shape['T, Lx, Ly, Sx, Sy'], Float]
    HType = Union[TensorTSxSy, TensorTLxLySxSy]
    MatrixN3 = NDArray[Shape['*, 3'], Float]
    TensorXY3 = NDArray[Shape['X, Y, 3'], Float]
    LaserGridType = Union[MatrixN3, TensorXY3]
    SensorGridType = Union[MatrixN3, TensorXY3]
    TensorXYZ3 = NDArray[Shape['X, Y, Z, 3'], Float]
    VolumeXYZType = Union[MatrixN3, TensorXY3, TensorXYZ3]
    Array3 = NDArray[Shape['3'], Float]

    # reconstruction types
    TensorN = NDArray[Shape['N'], Float]
    TensorXY = NDArray[Shape['X, Y'], Float]
    TensorXYZ = NDArray[Shape['X, Y, Z'], Float]
    SingleReconstructionType = Union[TensorN, TensorXY, TensorXYZ]
    TensorNN = NDArray[Shape['N, N'], Float]
    TensorXYXY = NDArray[Shape['X, Y, X, Y'], Float]
    TensorXYZXYZ = NDArray[Shape['X, Y, Z, X, Y, Z'], Float]
    ExhaustiveReconstructionType = Union[TensorNN, TensorXYXY, TensorXYZXYZ]
    GroundTruthDepthType = TensorXY
    GroundTruthNormalsType = TensorXY3

    #
    # Actual capture data (ignore _start and _end)
    #
    _start: None = None  # used in as_dict()
    H: HType = None
    H_format: HFormat = None
    sensor_xyz: Array3 = None
    sensor_grid_xyz: SensorGridType = None
    sensor_grid_normals: SensorGridType = None
    sensor_grid_format: GridFormat = None
    laser_xyz: Array3 = None
    laser_grid_xyz: LaserGridType = None
    laser_grid_normals: LaserGridType = None
    laser_grid_format: GridFormat = None
    # volume_xyz : VolumeXYZType = None  # deprecated
    volume_format: VolumeFormat = None  # deprecated
    delta_t: Float = None
    t_start: Float = None
    t_accounts_first_and_last_bounces: bool = None
    scene_info: dict = None  # additional information
    _end: None = None  # used in as_dict()

    def __get_dict_keys(self):
        variables = list(self.__class__.__dict__.keys())
        variables = variables[variables.index(
            '_start') + 1: variables.index('_end')]
        return variables

    def __init__(self, filename: str = None, file_format: FileFormat = FileFormat.AUTODETECT, skip_H: bool = False):
        if filename is None:
            return

        # Read raw data and check its current format
        final_path = filename
        while os.path.islink(final_path):
            aux_filename = final_path
            assert os.path.exists(aux_filename), \
                f'File {aux_filename} is a broken link'
            # Access the original data in case it is a link
            log(LogLevel.DEBUG, f'{aux_filename} is a link. Following link...')
            aux_filename = os.readlink(aux_filename)
            final_path = os.path.join(
                os.path.dirname(final_path), aux_filename)

        assert os.path.isfile(final_path), f'Does not exist: {final_path}'
        raw_data = read_hdf5(final_path, skip_H)
        if file_format == FileFormat.AUTODETECT:
            file_format = detect_dict_format(raw_data)

        # Convert data to HDF5_TAL format
        if file_format != FileFormat.HDF5_TAL:
            raw_data = convert_dict(
                raw_data, format_to=FileFormat.HDF5_TAL)

        own_dict_keys = self.__get_dict_keys()
        for key, value in raw_data.items():
            if key not in own_dict_keys:
                raise AssertionError(f'raw_data contains unknown key: {key}')
            if key == 'scene_info' and not isinstance(value, h5py.Empty):
                value = yaml.load(value, Loader=yaml.CLoader)

            setattr(self, key, value)

    def is_laser_paired_to_sensor(self):
        """
        Returns True if each sensor point only has one laser point associated.
        This is the case for confocal capture or custom configurations.
        In single or exhaustive configuration, this does not happen as each
        laser/sensor point can have multiple sensor/laser points associated.
        Some reconstructions algorithms require this information to work properly.
        """
        if self.H_format in [HFormat.T_Lx_Ly_Sx_Sy, HFormat.T_Li_Si]:
            return False
        elif self.H_format in [HFormat.T_Sx_Sy, HFormat.T_Si]:
            return (self.sensor_grid_format == self.laser_grid_format
                    and
                    self.sensor_grid_xyz.shape[:-1] ==
                    self.laser_grid_xyz.shape[:-1])
        else:
            raise AssertionError('Invalid H_format')

    def is_confocal(self):
        """ Returns True if H contains confocal data (i.e. Sx and Sy represent both laser and sensor coordinates) """
        if self.H_format in [HFormat.T_Lx_Ly_Sx_Sy, HFormat.T_Li_Si]:
            return False
        elif self.H_format in [HFormat.T_Sx_Sy, HFormat.T_Si]:
            return np.allclose(self.sensor_grid_xyz, self.laser_grid_xyz)
        else:
            raise AssertionError('Invalid H_format')

    def as_dict(self):
        """ Returns a dict containing all the data in this object """
        dict_keys = self.__get_dict_keys()
        return dict((key, getattr(self, key)) for key in dict_keys)

    def downscale(self, downscale: int):
        """
        Updates the data in this object to reduce the number of laser and sensor positions by the given factor

        e.g. Consider data with H_format = (T, Sx, Sy) and H.shape = (4096, 256, 256).
             Calling with downscale = 2 will reduce H.shape to (4096, 128, 128).
        """
        assert downscale > 1, 'downscale must be > 1'
        assert self.H_format in [HFormat.T_Sx_Sy], \
            'Only implemented for HFormat.T_Sx_Sy'
        
        nt, nsx, nsy = self.H.shape
        if self.is_confocal():
            self.laser_grid_xyz = self.laser_grid_xyz.reshape(
                (nsx // downscale, downscale, nsy // downscale, downscale, 3)).mean(axis=(1, 3))
        self.H = self.H.reshape(
            (nt, nsx // downscale, downscale, nsy // downscale, downscale)).sum(axis=(2, 4))
        self.sensor_grid_xyz = self.sensor_grid_xyz.reshape(
            (nsx // downscale, downscale, nsy // downscale, downscale, 3)).mean(axis=(1, 3))
        
        log(LogLevel.INFO,
            f'Downscaled from {nsx}x{nsy} to {nsx // downscale}x{nsy // downscale}')

    def get_single_subdata_from_laser_point(self, *args):
        """
        If your data has more than one illumination point, you can use this function
        to obtain the data for a single illumination point.

        The args specify the indices of the laser point in the same format as
        the impulse response H or the laser positions laser_grid_xyz.

        e.g. If H_format = (T, Lx, Ly, Sx, Sy), then args = (5, 10) will return
             the impulse response at (T, 5, 10, Sx, Sy).
        """
        n_indices = len(args)

        if n_indices == 1:
            assert self.H_format == HFormat.T_Li_Si and self.laser_grid_format == GridFormat.N_3, \
                'Number of indices must match H_format and laser_grid_format'
            il = args[0]
            assert 0 <= il and \
                il < self.laser_grid_xyz.shape[0] and \
                il < self.laser_grid_normals.shape[0] and \
                il < self.H.shape[1], \
                'Index out of bounds'
            slices = np.index_exp[il:il+1]
        elif n_indices == 2:
            assert self.H_format == HFormat.T_Lx_Ly_Sx_Sy and self.laser_grid_format == GridFormat.X_Y_3, \
                'Number of indices must match H_format and laser_grid_format'
            ilx, ily = args
            assert 0 <= ilx and 0 <= ily and \
                ilx < self.laser_grid_xyz.shape[0] and \
                ily < self.laser_grid_xyz.shape[1] and \
                ilx < self.laser_grid_normals.shape[0] and \
                ily < self.laser_grid_normals.shape[1] and \
                ilx < self.H.shape[1] and \
                ily < self.H.shape[2], \
                'Index out of bounds'
            slices = np.index_exp[ilx:ilx+1, ily:ily+1]
        else:
            raise AssertionError('Invalid number of indices')

        from copy import deepcopy
        data_out = deepcopy(self)
        # cannot use * operator directly inside slice [...]
        # as it is not supported until python 3.11
        # for now it's rewritten by wrapping everything inside a tuple
        data_out.H = data_out.H[(slice(None), *slices, ...)]
        data_out.laser_grid_xyz = \
            data_out.laser_grid_xyz[(*slices, ...)]
        data_out.laser_grid_normals = \
            data_out.laser_grid_normals[(*slices, ...)]

        return data_out

    def get_subset_of_laser_points_data(self, *args):
        """
        If your data has more than one illumination point, you can use this function
        to obtain the data for a smaller number of illumination points.

        The args specify the indices of the laser points in the same format as
        the impulse response H or the laser positions laser_grid_xyz.

        e.g. If H_format = (T, Lx, Ly, Sx, Sy), then args = ([5,6], [10]) will return
             the impulse response at (T, 5, 10, Sx, Sy) and (T, 6, 10, Sx, Sy).

        """
        n_indices = len(args)

        if n_indices == 1:
            assert self.H_format == HFormat.T_Li_Si and self.laser_grid_format == GridFormat.N_3, \
                'Number of indices must match H_format and laser_grid_format'
            il = np.array(args[0])
            assert np.all(0 <= il) and \
                np.all(il < self.laser_grid_xyz.shape[0]) and \
                np.all(il < self.laser_grid_normals.shape[0]) and \
                np.all(il < self.H.shape[1]), \
                'Index out of bounds'
            slices = np.index_exp[il]
        elif n_indices == 2:
            assert self.H_format == HFormat.T_Lx_Ly_Sx_Sy and self.laser_grid_format == GridFormat.X_Y_3, \
                'Number of indices must match H_format and laser_grid_format'
            ilx, ily = np.array(args[0]), np.array(args[1])
            assert np.all(0 <= ilx) and np.all(0 <= ily) and \
                np.all(ilx < self.laser_grid_xyz.shape[0]) and \
                np.all(ily < self.laser_grid_xyz.shape[1]) and \
                np.all(ilx < self.laser_grid_normals.shape[0]) and \
                np.all(ily < self.laser_grid_normals.shape[1]) and \
                np.all(ilx < self.H.shape[1]) and \
                np.all(ily < self.H.shape[2]), \
                'Index out of bounds'
            slices = np.ix_(ilx, ily)
        else:
            raise AssertionError('Invalid number of indices')

        from copy import deepcopy
        data_out = deepcopy(self)
        # cannot use * operator directly inside slice [...]
        # as it is not supported until python 3.11
        # for now it's rewritten by wrapping everything inside a tuple
        data_out.H = data_out.H[(slice(None), *slices, ...)]
        data_out.laser_grid_xyz = \
            data_out.laser_grid_xyz[(*slices, ...)]
        data_out.laser_grid_normals = \
            data_out.laser_grid_normals[(*slices, ...)]

        return data_out