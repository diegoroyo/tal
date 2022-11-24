from enum import Enum


class FileFormat(Enum):
    AUTODETECT = 0
    HDF5_ZNLOS = 1
    HDF5_NLOS_DIRAC = 2
    HDF5_TAL = 3
    MAT_PHASOR_FIELDS = 4


class HFormat(Enum):
    UNKNOWN = 0
    T_Sx_Sy = 1  # confocal or not
    T_Lx_Ly_Sx_Sy = 2
    T_Si = 3  # confocal or not
    T_Si_Li = 4


class GridFormat(Enum):
    UNKNOWN = 0
    N_3 = 1
    X_Y_3 = 2


class VolumeFormat(Enum):
    UNKNOWN = 0
    N_3 = 1
    X_Y_Z_3 = 2
    X_Y_3 = 3


class CameraSystem(Enum):
    STEADY = 0  # focused light
    # PHOTO_CAMERA = 1  # NYI (single-freq imaging)
    TRANSIENT = 2  # pulsed point light
    CONFOCAL_TIME_GATED = 3  # pulsed focused light

    def bp_accounts_for_d_2(self) -> bool:
        return self in [CameraSystem.STEADY, CameraSystem.CONFOCAL_TIME_GATED]

    def is_transient(self) -> bool:
        return self in [CameraSystem.TRANSIENT, CameraSystem.CONFOCAL_TIME_GATED]
