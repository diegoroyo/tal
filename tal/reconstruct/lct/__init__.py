"""
tal.reconstruct.lct
===================

Reconstruction using the Light Cone Transform (LCT) algorithm.
See "Confocal non-line-of-sight imaging based on the light-cone transform", 
O'Toole et al. (2018).

This implementation is an alternative to the other bp, fbp and pf/pf_dev 
approaches.
"""

from tal.io.capture_data import NLOSCaptureData


def solve(data: NLOSCaptureData,
          diffuse_material: bool = False,
          backprojection: bool = False,
          snr: float = 8e-1) -> NLOSCaptureData.SingleReconstructionType:
    """
    Reconstruction using the Light Cone Transform (LCT) algorithm.

    data
        See tal.io.read_capture

    diffuse_material
        If True, the material is considered diffuse, which affects the radiometric scaling.

    backprojection
        If True, the reconstruction is done using backprojection.

    snr
        The signal-to-noise ratio for the reconstruction, used in the Wiener filter.
    """
    assert data.is_confocal(), \
        "Data must be confocal to use LCT with y-tal"

    from tal.reconstruct.lct.lct import resolve_lct
    return resolve_lct(data.H, data.sensor_grid_xyz, data.delta_t, diffuse_material, backprojection, snr)
