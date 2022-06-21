from enum import Enum


class FileFormat(Enum):
    AUTODETECT = 0
    HDF5_ZNLOS = 1
    HDF5_NLOS_DIRAC = 2
    HDF5_TAL = 3


class HFormat(Enum):
    UNKNOWN = 0
    T_Sx_Sy = 1  # confocal or not
    T_Lx_Ly_Sx_Sy = 2


class GridFormat(Enum):
    UNKNOWN = 0
    N_3 = 1
    X_Y_3 = 2


class VolumeFormat(Enum):
    UNKNOWN = 0
    N_3 = 1
    X_Y_Z_3 = 2
