from tal.io.capture_data import NLOSCaptureData
import tal.reconstruct.pf.pf_solver as pf
from tal.reconstruct.pf.rsd_kernel import RSD_kernel
import numpy as np

def solve(data: NLOSCaptureData, volume: NLOSCaptureData.VolumeType = None):
    # IMPLEMENTING
    H = data.H
    T = data.delta_t*np.arange(H.shape[0])
    return pf.reconstruct(H, )