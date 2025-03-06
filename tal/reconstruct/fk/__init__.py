"""
tal.reconstruct.fk
===================

Reconstruction using the fk-migration algorithm.
See "Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration"

This implementation is an alternative to the other bp, fbp and pf/pf_dev 
approaches. 

WARNING: the fk-migration demands a lot of memory usage. If you think you might
get memory errors, try downscaling the y-tal data or trim the latest temporal
data.
"""

from tal.io.capture_data import NLOSCaptureData
import numpy as np
from scipy.interpolate import interpn


def solve(data: NLOSCaptureData) -> NLOSCaptureData.SingleReconstructionType:
    """
    See module description of tal.reconstruct.fbp

    data
        See tal.io.read_capture
    """
    assert data.is_confocal(), \
        "Data must be confocal to use fk-migration with y-tal"
    # TODO: implement non confocal approach from "Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration"
    if data.is_confocal():
        # Padding of the data
        N = data.sensor_grid_xyz.shape[0]
        M = data.H.shape[0]
        width = data.sensor_grid_xyz[-1,-1,0]
        range = data.delta_t*M

        data = data.H

        z,y,x = np.mgrid[-M:M, -N:N, -N:N]
        x = x/N
        y = y/N
        z = z/M
        
        grid_z = np.tile(np.linspace(0, 1, M)[:, np.newaxis, np.newaxis],
                        (1, N, N))
        aux_data = np.sqrt((data*grid_z)**2)

        t_data = np.zeros((2*M, 2*N, 2*N))
        t_data[:M, :N, :N] = aux_data
        # FFT
        f_data = np.fft.fftshift(np.fft.fftn(t_data))

        # Stolt trick
        sqrt_term = np.sqrt((N*range/(M*width*4))**2 * (x**2 + y**2) + z**2)
        f_vol = interpn((z[:,0,0],y[0,:,0],x[0,0,:]),
                        f_data, 
                        np.moveaxis(np.array([sqrt_term, y, x]), 0,-1),
                        method = 'linear',
                        bounds_error = False,
                        fill_value=0)
        f_vol *= z>0
        f_vol *= np.abs(z) / np.max(sqrt_term)

        # IFFT
        t_vol = np.fft.ifftn(np.fft.ifftshift(f_vol))
        t_vol = np.abs(t_vol)**2

        return t_vol[:M, :N, :N]
