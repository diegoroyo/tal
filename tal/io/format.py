from tal.io.enums import FileFormat, GridFormat, HFormat, VolumeFormat
import numpy as np


def __convert_dict_znlos_to_tal(capture_data: dict) -> dict:
    H = capture_data['data']
    if H.ndim == 7:
        # Exhaustive/single - convert to (colors, bounces, t, lx, ly, sx, sy)
        if H.shape[4] == 6:
            # assume (ly, lx, sy, sx, bounces, t, colors)
            H = np.moveaxis(H, list(range(7)), (4, 3, 6, 5, 1, 2, 0))
        elif H.shape[2] == 6:
            # assume (colors, t, bounces, sx, sy, lx, ly)
            H = np.moveaxis(H, list(range(7)), (0, 2, 1, 6, 5, 4, 3))
        else:
            raise AssertionError(
                'Conversion not detected for this H.ndim = 7 case')
    elif H.ndim == 5:
        # Confocal - convert to (colors, bounces, t, sx, sy)
        if H.shape[0] == H.shape[1]:
            # assume (sy, sx, bounces, t, colors)
            H = np.moveaxis(H, list(range(5)), (3, 4, 1, 2, 0))
        elif H.shape[3] == H.shape[4]:
            # assume (colors, t, bounces, sx, sy)
            H = np.moveaxis(H, list(range(5)), (0, 2, 1, 4, 3))
        else:
            raise AssertionError(
                'Conversion not detected for this H.ndim = 5 case')
    else:
        raise AssertionError('Conversion not implemented for H.ndim != 5 or 7')
    # sum colors and bounces dims (t, lx, ly, sx, sy)
    H = np.sum(H, axis=(0, 1))
    # remove (1, 1) dims (e.g. laser in single capture 1x1x256x256)
    H = np.squeeze(H)
    H_format = HFormat.T_Sx_Sy if H.ndim == 3 else HFormat.T_Lx_Ly_Sx_Sy

    def conv_to_xy3(arr):
        if arr.shape[0] == 3:
            return np.transpose(arr)
        else:
            return arr

    def parse_volume_size(volume_size):
        volume_size = np.array(volume_size, dtype=np.float32)
        if volume_size.size == 1:
            volume_size = np.repeat(volume_size, 3)
        return volume_size

    return {
        'H': H,
        'H_format': H_format,
        'sensor_xyz': capture_data['cameraPosition'].reshape(3),
        'sensor_grid_xyz': conv_to_xy3(capture_data['cameraGridPositions']),
        'sensor_grid_normals': conv_to_xy3(capture_data['cameraGridNormals']),
        'sensor_grid_format': GridFormat.X_Y_3,
        'laser_xyz': capture_data['laserPosition'].reshape(3),
        'laser_grid_xyz': conv_to_xy3(capture_data['laserGridPositions']),
        'laser_grid_normals': conv_to_xy3(capture_data['laserGridNormals']),
        'laser_grid_format': GridFormat.X_Y_3,
        'volume_format': VolumeFormat.X_Y_Z_3,
        'delta_t': capture_data['deltaT'],
        't_start': capture_data['t0'],
        'scene_info': {
            'original_format': 'HDF5_ZNLOS',
            'volume': {
                'center': capture_data['hiddenVolumePosition'].reshape(3),
                'rotation': capture_data['hiddenVolumeRotation'].reshape(3),
                'size': parse_volume_size(capture_data['hiddenVolumeSize']),
            }
        },
    }


def __convert_dict_dirac_to_tal(capture_data: dict) -> dict:
    raise NotImplementedError(
        'Conversion from HDF5_NLOS_DIRAC not implemented')


def __convert_dict_tal_to_znlos(capture_data: dict) -> dict:
    raise NotImplementedError('Conversion to HDF5_ZNLOS not implemented')


def __convert_dict_tal_to_dirac(capture_data: dict) -> dict:
    raise NotImplementedError('Conversion to HDF5_NLOS_DIRAC not implemented')


def detect_dict_format(raw_data: dict) -> FileFormat:
    if 'data' in raw_data:
        return FileFormat.HDF5_ZNLOS
    elif 'data_t' in raw_data:
        return FileFormat.HDF5_NLOS_DIRAC
    elif 'H' in raw_data:
        return FileFormat.HDF5_TAL
    else:
        raise AssertionError('Unable to detect capture data file format')


def convert_dict(capture_data: dict,
                 format_to: FileFormat) -> dict:
    """
    Convert raw data from one format to another
    """
    # convert to HDF5_TAL
    file_format = detect_dict_format(capture_data)
    if file_format == FileFormat.HDF5_TAL:
        capture_data_tal = capture_data
    elif file_format == FileFormat.HDF5_ZNLOS:
        capture_data_tal = __convert_dict_znlos_to_tal(capture_data)
    elif file_format == FileFormat.HDF5_NLOS_DIRAC:
        capture_data_tal = __convert_dict_dirac_to_tal(capture_data)
    else:
        raise AssertionError(
            'convert_dict not implemented for this file format')

    # convert from HDF5_TAL to output format
    if format_to == FileFormat.HDF5_TAL:
        return capture_data_tal
    elif format_to == FileFormat.HDF5_ZNLOS:
        return __convert_dict_tal_to_znlos(capture_data_tal)
    elif format_to == FileFormat.HDF5_NLOS_DIRAC:
        return __convert_dict_tal_to_dirac(capture_data_tal)
    else:
        raise AssertionError(
            'convert_dict not implemented for this file format')
