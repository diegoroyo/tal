from tal.io.capture_data import NLOSCaptureData
import tal.reconstruct.pf.pf_solver as pf
from tal.reconstruct.pf.rsd_kernel import RSD_kernel
import numpy as np

def solve(data: NLOSCaptureData, wavefactor: float, wave_cycles: float,
         volume: NLOSCaptureData.VolumeType = None, verbose: int = 1, 
         n_threads: int = 1):
    # Data extraction
    H = data.H
    # Time extraction
    if data.t_start is not None:
        # Correct delayed data
        t_pad = int(np.round(data.t_start / data.delta_t))
        H = np.pad(H, ((t_pad,0),(0,0),(0,0)),'minimum')
    T = data.delta_t*np.arange(H.shape[0])
    
    # TODO correct volume
    V = volume
    S = data.sensor_grid_xyz
    L = np.zeros((1,3), float)
    # FIXME: Minimun distance assumes ordered points
    reshaped_S = S.reshape(-1,3)
    wavelength = wavefactor*np.linalg.norm(reshaped_S[0] - reshaped_S[1])


    return pf.reconstruct(H, T, S, L, V, wavelength, wave_cycles, True,
                         n_threads, verbose)