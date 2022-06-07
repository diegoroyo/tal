from enum import Enum


class FileFormat(Enum):
    AUTODETECT = 0
    HDF5_ZNLOS = 1
    HDF5_NLOS_DIRAC = 2
    HDF5_TAL = 3


class HFormat(Enum):
    T_Sx_Sy = 0  # confocal or not
    T_Lx_Ly_Sx_Sy = 1


class GridFormat(Enum):
    N_3 = 0
    X_Y_3 = 1


class VolumeFormat(Enum):
    N_3 = 0
    X_Y_Z_3 = 1
