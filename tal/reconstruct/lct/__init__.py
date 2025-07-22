"""
tal.reconstruct.lct
===================

Reconstruction using the Light Cone Transform (LCT) algorithm.
See "Confocal non-line-of-sight imaging based on the light-cone transform", 
O'Toole et al. (2018).

This implementation is an alternative to the other bp, fbp and pf/pf_dev 
approaches. 

WARNING: the fk-migration demands a lot of memory usage. If you think you might
get memory errors, try downscaling the y-tal data or trim the latest temporal
data.
"""

from tal.io.capture_data import NLOSCaptureData

def solve(data: NLOSCaptureData, diffuse_material: bool = False, backprojection : bool = False, snr: float = 8e-1) -> NLOSCaptureData.SingleReconstructionType:
    """
    See module description of tal.reconstruct.fbp

    data
        See tal.io.read_capture
    """
    assert data.is_confocal(), \
        "Data must be confocal to use LCT with y-tal"

    if data.is_confocal():
         from tal.reconstruct.lct.lct import resolve_lct
         return resolve_lct(data.H, data.sensor_grid_xyz, data.delta_t, diffuse_material, backprojection, snr)