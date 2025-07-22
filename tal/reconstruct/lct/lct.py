import numpy as np
from scipy.interpolate import interpn
from scipy.sparse import lil_matrix, spdiags
from scipy.fft import fftn, ifftn
from tal.io.capture_data import NLOSCaptureData


def resolve_lct(H_0, sensor_grid_xyz, delta_t: float, diffuse_material: bool, backprojection: bool, snr: float) -> NLOSCaptureData.SingleReconstructionType:
    """
    Resolve the Light Cone Transform (LCT) reconstruction.
    H_0
        The input data in the time domain, shaped as (nt, nx, ny).
    sensor_grid_xyz
        The sensor grid coordinates, shaped as (nx, ny, 3).
    delta_t
        The time step between samples.
    diffuse_material
        Whether the material is diffuse or not.
    backprojection
        If True, the reconstruction is done using backprojection.
    snr
        The signal-to-noise ratio for the reconstruction.
    Returns
        The reconstructed volume in the time domain, shaped as (nt, nx, ny).
    """
    nt, nx, ny = H_0.shape
    width = sensor_grid_xyz[-1,-1,0]
    range = delta_t * nt

    # Define NLOS blur kernel
    inverse_psf = define_psf(H_0.shape, width / range, backprojection, snr)

    # Define transform operators
    mtx, mtxi = resampling_operator(nt)

    # Define volume representing voxel distance from wall
    grid_z = np.tile(np.linspace(0, 1, nt)[:, np.newaxis, np.newaxis], (1, nx, ny))

    # Scale radiometric component
    if diffuse_material:
        H_1 = H_0 * (grid_z ** 4)
    else:
        H_1 = H_0 * (grid_z ** 2)

    # Forward transform
    H_1 = H_1.reshape(nt, nx * ny)
    H_1 = mtx @ H_1
    H_1 = H_1.reshape(nt, nx, ny)

    f_H_1 = np.zeros((2 * nt, 2 * nx, 2 * ny), dtype=np.complex128)  
    f_H_1[:nt, :nx, :ny] = H_1

    # Apply the inverse PSF and transform to the time domain again
    f_H_1 = np.fft.ifftn(np.fft.fftn(f_H_1) * inverse_psf)
    H_1 = f_H_1[:nt, :nx, :ny]
    H_1 = H_1.reshape(nt, nx * ny)
    H_1 = mtxi @ H_1
    H_1 = np.abs(H_1.reshape(nt, nx, ny)).astype(np.float64)

    return H_1


def define_psf(shape: tuple, slope: float, backprojection: bool, snr: float) -> np.ndarray:
    """Compute blur kernel"""
    x = np.linspace(-1, 1, 2 * shape[1])    # (2 * nt, 2 * nx, 2 * ny)
    y = np.linspace(-1, 1, 2 * shape[2])
    z = np.linspace(0, 2, 2 * shape[0])
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')

    # Define PSF
    psf = np.abs(((4 * slope) ** 2) * (grid_x ** 2 + grid_y ** 2) - grid_z)
    psf = (psf == np.min(psf, axis=0, keepdims=True)).astype(float)
    psf = psf / np.sum(psf[:, shape[1], shape[2]])      # Normalize along z-axis
    psf = psf / np.linalg.norm(psf)
    psf = np.roll(psf, shape[1:], axis=(1, 2))          # Shift to center

    # FFT
    fpsf = fftn(psf)
    if not backprojection:
        invpsf = np.conj(fpsf) / (np.abs(fpsf) ** 2 + 1 / snr)
    else:
        invpsf = np.conj(fpsf)

    return invpsf


def resampling_operator(M):
    """Define resampling operators"""
    mtx = lil_matrix((M ** 2, M))           # We need an sparse matrix for efficiency; the final mtx and mtxi are shaped as (M, M) though

    x = np.arange(1, M ** 2 + 1)
    rows = x - 1
    cols = np.ceil(np.sqrt(x)) - 1
    mtx[rows, cols.astype(int)] = 1
    mtx = spdiags(1 / np.sqrt(x), 0, M ** 2, M ** 2) @ mtx
    mtxi = mtx.T

    K = int(np.round(np.log2(M)))
    for k in range(K):                      # Merge every two rows and columns until we have M rows and columns
        mtx = 0.5 * (mtx[::2] + mtx[1::2])
        mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])

    return mtx, mtxi