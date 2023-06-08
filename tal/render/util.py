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
