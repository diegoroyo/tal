import pytest

_EXPERIMENT_FOLDER = 'examples/render-reconstruct'


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(_EXPERIMENT_FOLDER)


def test00_Z_single_llvm():
    import tal
    from tal.render import render_nlos_scene
    import numpy as np
    import argparse
    import os

    args = {
        'threads': os.cpu_count() - 1,
        'seed': 0,
        'nice': 0,
        'gpus': None,
        'dry_run': False,
        'do_steady_renders': True,
        'do_ground_truth_renders': True,
        'do_logging': True,
        'keep_partial_results': True,
    }
    args = argparse.Namespace(**args)
    config_path = 'nlos-z'

    hdf5_path = render_nlos_scene(config_path, args)

    data = tal.io.read_capture(hdf5_path)

    assert data.H.shape == (320, 64, 64)
    assert np.isclose(data.H.sum(), 280.32675)


def test00_Z_single_cuda():
    import tal
    from tal.render import render_nlos_scene
    import numpy as np
    import argparse
    import os

    args = {
        'threads': os.cpu_count() - 1,
        'seed': 0,
        'nice': 0,
        'gpus': [0],
        'dry_run': False,
        'do_steady_renders': True,
        'do_ground_truth_renders': True,
        'do_logging': True,
        'keep_partial_results': True,
    }
    args = argparse.Namespace(**args)
    config_path = 'nlos-z-gpu'

    hdf5_path = render_nlos_scene(config_path, args)

    data = tal.io.read_capture(hdf5_path)

    assert data.H.shape == (320, 64, 64)
    assert np.isclose(data.H.sum(), 280.32675)
