from tal.io.capture_data import NLOSCaptureData
import tal.reconstruct.pf.pf_solver as pf
from tal.reconstruct.pf.rsd_kernel import RSD_kernel
import numpy as np


def solve(data: NLOSCaptureData, wavefactor: float, wave_cycles: float,
          volume: NLOSCaptureData.VolumeType = None, res_in_freq=False,
          verbose: int = 1, n_threads: int = 1):
    """
    Reconstruct a NLOS captured data with a gaussian pulse in the volume
    using phasor fields and the RSD propagation.
    @param data         : NLOS capture data at the relay wall
    @param wavefactor   : Central wavefactor of the virtual illumination pulse
                          used for the reconstruction. The wavelength will be 
                          wavefactor*delta_x (distance between relay wall 
                          sensor points)
    @param wave_cycles  : Number of central wavelength cycles of the virtual
                          illumination pulse
    @param volume       : Desired volume in the 3d space to reconstruct from 
                          the relay wall
    @param res_in_freq  : If True, return the result by frequencies, without 
                          convining the used frequencies into the time domain.
                          If False, return the result in time domain, evaluated
                          in t=0
    @param verbose      : It can be used to select from 0 to 3 levels of 
                          information printed by screen. Iff 0 prints nothing,
                          if 3 prints the whole information
    @param n_threads    : Number of threads to use for the reconstruction. By
                          default is set to 1
    @return             : Reconstruction from the relay wall data to the given
                          volume, in the given form.    
    """
    # Data extraction
    H = data.H
    # Time extraction
    if data.t_start is not None:
        # Correct delayed data
        t_pad = int(np.round(data.t_start / data.delta_t))
        H = np.pad(H, ((t_pad, 0), (0, 0), (0, 0)), 'minimum')
    T = data.delta_t*np.arange(H.shape[0])

    # TODO correct volume
    V = volume
    S = data.sensor_grid_xyz
    L = np.zeros((1, 3), float)

    # FIXME: Minimun distance assumes ordered points
    reshaped_S = S.reshape(-1, 3)
    wavelength = wavefactor*np.linalg.norm(reshaped_S[0] - reshaped_S[1])

    return pf.reconstruct(H, T, S, L, V, wavelength, wave_cycles, res_in_freq,
                          n_threads, verbose)
