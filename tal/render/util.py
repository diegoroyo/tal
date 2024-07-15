def import_mitsuba_backend():
    from tal.config import ask_for_config, Config
    mitsuba_version = ask_for_config(Config.MITSUBA_VERSION, force_ask=False)
    if mitsuba_version == '2':
        import tal.render.mitsuba2_transient_nlos as mitsuba_backend
    elif mitsuba_version == '3':
        import tal.render.mitsuba3_transient_nlos as mitsuba_backend
    else:
        raise AssertionError(
            f'Invalid MITSUBA_VERSION={mitsuba_version}, must be one of (2, 3)')
    return mitsuba_backend


def get_grid_xyz(nx, ny, rw_scale_x, rw_scale_y, ax0=0, ax1=1, ay0=0, ay1=1):
    import numpy as np
    px0 = -rw_scale_x + 2 * rw_scale_x * ax0
    px1 = rw_scale_x - 2 * rw_scale_x * (1 - ax1)
    py0 = -rw_scale_y + 2 * rw_scale_y * ay0
    py1 = rw_scale_y - 2 * rw_scale_y * (1 - ay1)
    xg = np.stack(
        (np.linspace(px0, px1, num=2*nx + 1)[1::2],)*ny, axis=1)
    yg = np.stack(
        (np.linspace(py0, py1, num=2*ny + 1)[1::2],)*nx, axis=0)
    assert xg.shape[0] == yg.shape[0] == nx and xg.shape[1] == yg.shape[1] == ny, \
        'Incorrect shapes'
    return np.stack([xg, yg, np.zeros((nx, ny))], axis=-1).astype(np.float32)


def expand_xy_dims(vec, x, y):
    assert len(vec) == 3
    return vec.reshape(1, 1, 3).repeat(x, axis=0).repeat(y, axis=1)
