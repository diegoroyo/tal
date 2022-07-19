

from enum import Enum
import os
import h5py
import yaml
import numpy as np
import inspect
from typing import Union, get_type_hints
from nptyping import NDArray, Shape
from tal.io.format import convert_dict, detect_dict_format
from tal.io.enums import FileFormat, HFormat, GridFormat, VolumeFormat


def read_hdf5(filename: str) -> dict:
    raw_data = h5py.File(filename, 'r')

    def parse(key, value):
        if isinstance(value, h5py.Empty):
            value = None
        else:
            if isinstance(value, h5py.Dataset):
                value = value[()]
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
            value = yaml.safe_load(value)
        return value
    # FIXME(diego): performing np.array() of each element is pretty slow
    # if the data is not going to be read (e.g. converting sensor_grid_xyz to
    # an array is not necessary if you are only going to read H)
    # however, the overhead is pretty small
    raw_data = dict((key, parse(key, raw_data.get(key)))
                    for key in raw_data.keys())
    return raw_data


def write_hdf5(capture_data: dict, filename: str):
    file = h5py.File(filename, 'w')
    for key, value in capture_data.items():
        if value is None:
            file[key] = h5py.Empty(float)
        elif isinstance(value, dict):
            file[key] = yaml.dump(value)
        elif isinstance(value, Enum):
            dt = h5py.enum_dtype(dict((item.name, item.value)
                                 for item in value.__class__), basetype='i')
            ds = file.create_dataset(key, (1,), dtype=dt)
            ds[0] = value.value
        else:
            file[key] = value
    file.close()


class NLOSCaptureData:

    """
    Type aliases
    """
    Float = np.float32
    TensorTSxSy = NDArray[Shape['T, Sx, Sy'], Float]
    TensorTLxLySxSy = NDArray[Shape['T, Lx, Ly, Sx, Sy'], Float]
    HType = Union[TensorTSxSy, TensorTLxLySxSy]
    MatrixN3 = NDArray[Shape['*, 3'], Float]
    TensorXY3 = NDArray[Shape['X, Y, 3'], Float]
    LaserGridType = Union[MatrixN3, TensorXY3]
    SensorGridType = Union[MatrixN3, TensorXY3]
    TensorXYZ3 = NDArray[Shape['X, Y, Z, 3'], Float]
    VolumeType = Union[MatrixN3, TensorXYZ3]
    Array3 = NDArray[Shape['3'], Float]

    """ Capture data start (ignore _start) """
    _start: None = None  # used in as_dict()
    """
    TODO(diego): add docs
    """
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
    # volume_xyz : VolumeType = None  # implemented as a property, see below
    volume_format: VolumeFormat = None
    delta_t: Float = None
    t_start: Float = None
    t_accounts_first_and_last_bounces: bool = None
    scene_info: dict = None  # additional information
    """
    Implemented scene_info keys:
    - 'original_format': str (e.g. 'HDF5_ZNLOS')
    - 'config': dict (original scene_config.yaml when generated using TAL)
    - 'args': dict (original args passed when generated using TAL)
    - 'volume': dict
        - 'center': Array3 (center of volume)
        - 'rotation': Array3 (rotation of volume formatted as per Z-NLOS - probably unused for now)
        - 'size': Array3 (size of volume - length of each XYZ side)
        - 'xyz': MatrixN3 (if ever volume_format is VolumeFormat.N_3, points will be stored here)
    """
    """ Capture data end (ignore _end) """
    _end: None = None  # used in as_dict()

    def __get_dict_keys(self):
        variables = list(self.__class__.__dict__.keys())
        variables = variables[variables.index(
            '_start') + 1: variables.index('_end')]
        return variables

    def __init__(self, filename: str = None, file_format: FileFormat = FileFormat.AUTODETECT):
        if filename is None:
            return

        # Read raw data and check its current format
        assert os.path.isfile(filename), f'Does not exist: {filename}'
        raw_data = read_hdf5(filename)
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
            setattr(self, key, value)

    def is_confocal(self):
        if self.H_format == HFormat.T_Lx_Ly_Sx_Sy:
            return False
        elif self.H_format == HFormat.T_Sx_Sy:
            return self.laser_grid_xyz.shape == self.sensor_grid_xyz.shape
        else:
            raise AssertionError('Invalid H_format')

    def as_dict(self):
        dict_keys = self.__get_dict_keys()
        return dict((key, getattr(self, key)) for key in dict_keys)

    @property
    def volume_xyz(self):
        try:
            return self.scene_info['volume']['xyz']
        except KeyError:
            pass

        try:
            # TODO(diego): return (X, Y, Z, 3) tensor using
            # self.scene_info['volume']['center' | 'rotation' | 'size']
            raise NotImplementedError('volume_xyz is not implemented yet')
        except KeyError:
            return None
