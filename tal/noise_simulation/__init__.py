"""
tal.noise_simulation
======

Functions for simulating the noise caused by different sources (temporal jitter, SPAD, dark counts...) into an already simulated capture data.

This is a private module, it is recommended to use the command line interface instead of calling these functions directly.

See tal noise_simulation -h for more information.
"""

def simulate_noise(capture_data_path, config_path, args):
    """
    It is recommended to use the command line interface instead of calling this function directly.

    """
    from tal.noise_simulation import noise_simulation
    return noise_simulation.simulate_noise(capture_data_path, config_path, args)
