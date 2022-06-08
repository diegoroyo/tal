import tal
import os


def test(filename):
    print(f'Results for {filename}:')
    capture = tal.io.read_capture(
        os.path.join(os.path.split(__file__)[0], filename))
    print(f'H: {capture.H.shape}')
    print(f'H_format: {capture.H_format}')
    print(f'sensor_xyz: {capture.sensor_xyz}')
    print(f'sensor_grid_xyz: {capture.sensor_grid_xyz.shape}')
    print(f'sensor_grid_normals: {capture.sensor_grid_normals.shape}')
    print(f'sensor_grid_format: {capture.sensor_grid_format}')
    print(f'laser_xyz: {capture.laser_xyz}')
    print(f'laser_grid_xyz: {capture.laser_grid_xyz.shape}')
    print(f'laser_grid_normals: {capture.laser_grid_normals.shape}')
    print(f'laser_grid_format: {capture.laser_grid_format}')
    print(f'volume_format: {capture.volume_format}')
    # NOTE: idk why but delta_t upgrades to float64 here - its still stored as float32
    print(f'delta_t: {capture.delta_t}')
    print(f't_start: {capture.t_start}')
    print(
        f't_accounts_first_last_bounces: {capture.t_accounts_first_last_bounces}')
    print(f'scene_info: {capture.scene_info}')
    print()


test('data/NLOSDIRAC_nlosletters_nooffset.hdf5')
test('data/ZNLOS_Zdiffuse_withoffset.hdf5')
test('data/ZNLOS_Z_05_Exhaustive.hdf5')
test('data/ZNLOS_Z_05_Single.hdf5')
test('data/ZNLOS_Z_10_Confocal.hdf5')
test('data/ZNLOS_Z_10_Single.hdf5')
