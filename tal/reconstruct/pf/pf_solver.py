"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   pf_solver.py
Description :   Solves the NLOS problem using the phasor fields approximation,
                applying Rayleigh-Sommerfeld propagation to fill the given 
                voxelization
"""
import numpy as np
from numba import njit
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from .rsd_kernel import RSD_kernel

from tal.reconstruct.pf.propagator import RSD_parallel_propagator, RSD_propagator



def pulse(delta_t: float, n_w: int, lambda_c: float, cycles: float):
    """
    Generate a gaussian pulse in frequency space. It assume the data 
    bins are equally space in time. Returns the pulse in frequency,
    the wavelengths of the frequencies extracted from the data, and
    the indices to contain the 99.74% of the frequencies given the 
    centered frequency with lambda_c wavelength, repeated cycles times
    @param  delta_t     : Time spacing between bins
    @param  n_w         : Number of frequencies extracted with numpy fft
    @param  lambda_c    : Central wavelength of the pulse
    @param  cycles      : Number of pulses to consider of the data
    @return             : The values of the frequency pulse, the wave-
                            lengths of the frequencies to return and the 
                            indices to contain the 99.74% distribution of the
                            frequencies

    """
    sigma = cycles*lambda_c / 6             # Gaussian sigma
    w_c = 1 / lambda_c                      # Central frequency
    w = np.fft.fftfreq(n_w, delta_t)        # All frequencies
    # To control division by 0
    w[0] = 1e-30
    wavelengths = 1 / w                     # All wavelengths
    # Correct 0 and inf values
    wavelengths[0] = np.inf; w[0] = 0.0
    # Frequency distances to central frequency in rad/s
    delta_w = 2*np.pi*(w - w_c)
    # Gaussian pulse for reconstruction
    freq_pulse = np.exp(-sigma**2*delta_w**2/2)*np.sqrt(2*np.pi)*sigma

    # Central freq and sigma to k up to numpy fft
    central_k = delta_t * n_w * w_c
    sigma_k = delta_t * n_w /(2*np.pi*sigma)

    # Interest indicies in range [mu-3sigma, mu+3sigma]
    min_k = np.round(central_k - 3*sigma_k)
    max_k = np.round(central_k + 3*sigma_k)
    # Indices in the gaussian pulse
    indices = np.arange(min_k, max_k+1, dtype=int) 

    return (freq_pulse, wavelengths, indices)
    

def H_to_fH(H: np.ndarray, t_bins: np.ndarray,  lambda_c: float,
            cycles: float):
    """
    Transforms the data H in time, to the desired Fourier space frequencies
    @param H        : The impulse response of the scene
    @param t_bins   : Time of each temporal bin
    @param lambda_c : The wavelength of the central frequency for the virtual
                      illumination pulse
    @param cycles   : Number of cycles of the virtual illumination pulse
    """
    if cycles == 0:
        # Fourier term function
        fourier_term = np.exp( -2 * np.pi * t_bins / lambda_c * 1j)
        fH = np.array([np.sum(H*fourier_term.reshape((-1,)+(1,)*H.ndim),
                             axis = 0)])
        wv = np.array([lambda_c])
        f_pulse = np.ones(1)
        significant_idx = None
    else:
        fH_all = np.fft.fft(H, axis = 0)
        f_pulse, wv_all, significant_idx = pulse(t_bins[1], fH_all.shape[0],
                                        lambda_c, cycles)
        wv = wv_all[significant_idx]
        fH = fH_all[significant_idx]
    
    return (fH, wv, f_pulse, significant_idx)


def fI_to_I(fI: np.ndarray, f_pulse: np.ndarray, sig_idx: np.ndarray,
        n_threads: int = 1, 
        desc: str = "Fourier to time reconstruction by planes"):
    """
    Transforms the volume reconstruction I in Fourier domain to time domain
    executed in parallel
    @param fI       : A volume reconstruction of the scene by frequency domain
    @param f_pulse  : Pulse used for the reconstruction in Fourier domain
    @param sig_idx  : Iff not None, indices that correspond in the f_pulse to
                      the most significant values of the pulse
    @param  desc    : Description to show in the progress bar. If none, it 
                      shows no progress bar 
    @return         : The volume reconstruction of the scene in time domain, 
                      evaluated in t=0
    """
    if sig_idx is None:    # Single frequency
        # No need to convert the data
        return fI[0]
    else:                   # Multiple frequency
        with Pool(n_threads) as p:
            fIz2Iz_partial = partial(__fIz2Iz,
                                    f_pulse = f_pulse, 
                                    significant_idx = sig_idx)
            return np.array(list(tqdm(p.imap(fIz2Iz_partial, fI.swapaxes(0,1)),
                                        desc=desc,
                                        disable = desc is None,
                                        unit = "plane",
                                        total = fI.shape[1])))


def propagate_parallel_planes(fH: np.ndarray, P: np.ndarray, V: np.ndarray,
                        wl_v: np.ndarray, desc: str = 'Reconstructing planes',
                        n_threads: int = 1):
    """
    Propagate the impulse response in Fourier space to the plane V of
    the voxelization, whith wavelengths wl. The first axis of fH and wl
    match for each frequency used.
    @param fH           : The impulse response in the relay wall in
                          Fourier domain. The first axis represent each
                          frequency wl
    @param P            : Matrix of points of a plane surface
    @param V            : 3d volume formed by parallel planes to P
    @param wl_v         : Array of wavelength frequencies for each fH 
                          component in the first axis
    @param desc         : Description to show in the progress bar. If none,
                          it shows no progress bar 
    @param n_threads    : Number of threads to use for the propagation
    @return             : A RSD propagation given fH values, distributed
                          as P, to the 3d parallel volume V, with 
                          wavelengths wl
    """
    # Creates the Kernel object
    K = RSD_kernel(V, P, wl_v, None)
    K.pre_proc()
    K_shape = K.kernel_shape()
    # Spatial 2d fourier to fast convolutions
    ffH = np.fft.fft2(fH, s = K_shape)

    nw = len(wl_v)      # Number of frequencies
    ffH_dims = ffH.ndim - 1    # fH spatial dimensions. The first one is freq
    ffh_3d = (ffH_dims == 3)
    # Reshape for the transformation. It takes into account fH is 3D
    ffH_reshp = ffH.reshape((nw, V.shape[0] * ffh_3d + (not ffh_3d)) \
                            + K_shape)

    # Arguments to run in parallel
    pool_prop_i = partial(__f_propagate_plane_i, K, ffH_reshp, ffh_3d,
                            (fH.shape[1], fH.shape[2]))

    # Propagate to each plane of V in parallel
    with ThreadPoolExecutor(max_workers = n_threads) as executor:
        conv_H_K = np.array(
                        list(tqdm( 
                            executor.map(pool_prop_i, range(V.shape[0])),
                                    desc = desc,
                                    disable = desc is None,
                                    unit = "plane",
                                    total = V.shape[0]
                            )))
    # Return the real convolved part by frequencies at the first axis
    return conv_H_K.swapaxes(0,1)
    

def propagate_points(fHz: np.ndarray, P: np.ndarray, V: np.ndarray,
                    wl_v: np.ndarray, desc: str = 'Reconstructing points'):
    """
    Propagate the impulse response in Fourier space from fHz, with points P, to
    the V points.
    @param fHz  : Impulse response values
    @param P    : Impulse response positions in the 3d space
    @param V    : Choosen coordinates to propagate
    @param wl_v : Vector of all the wavelengths contained in fHz
    @return     : The propagation by frequencies to the plane
    """
    assert V.ndim == 2, 'propagate_points V array of 3d points'
    assert P.ndim == 2, 'propagate_points P array of 3d points'

    propagated = np.zeros((fHz.shape[0], V.shape[0]), dtype = np.complex128)
    for hd_v in tqdm(range(V.shape[0]), 
                    desc = desc,
                    disable = desc is None,
                    unit='point', 
                    total = V.shape[0]):
        # Calculate the distances
        K_d= np.sqrt(np.sum((V[hd_v, np.newaxis,:] - P[np.newaxis, :, :])**2, axis = -1))
        # Calculate the RSD kernel
        K_rsd = np.zeros((len(wl_v),) + K_d.shape, dtype = np.complex128)
        for hd_w, wl in enumerate(wl_v):
            K_rsd[hd_w] = RSD_kernel.RSD_kernel_w(K_d, wl)
        # Return the RSD propagation from Hz plane
        propagated[:, hd_v] = np.sum(fHz[:, np.newaxis, :]*K_rsd, axis = -1)[:, 0]

    return propagated


def propagate_plane(Vz: np.ndarray, P: np.ndarray, fHz: np.ndarray,
                    wl_v: np.ndarray):
    """
    Propagate the impulse response in Fourier space from fHz, with points P, to
    the Vz plane, with wl_v frecuencies.
    @param Vz   : Choosen plane coordinates to propagate
    @param P    : Impulse response positions in the 3d space
    @param fHz  : Impulse response values
    @param wl_v : Vector of all the wavelengths contained in fHz
    @return     : The propagation by frequencies to the plane
    """
    # Calculate the distances
    K_d= np.sqrt(np.sum((Vz[..., np.newaxis,:] - P)**2, axis = -1))
    # Calculate the RSD kernel
    K_rsd = np.zeros((len(wl_v),) + K_d.shape, dtype = np.complex128)
    for hd_w, wl in enumerate(wl_v):
        K_rsd[hd_w] = RSD_kernel.RSD_kernel_w(K_d, wl)
    # Return the RSD propagation from Hz plane
    return np.sum(fHz*K_rsd, axis = -1)


def propagate(fH: np.ndarray, P: np.ndarray, V: np.ndarray,
                wl_v: np.ndarray, n_threads: int = 1, 
                desc: str = "Reconstructing planes"):
    """
    Propagate the impulse response in Fourier space to the volume V
    voxelization, whith wavelengths wl. The first axis of fH and wl
    match for each frequency used.
    @param fH           : The impulse response in the relay wall in Fourier
                          domain. The first axis represent each frequency wl.
                          It can be a 2d plane , or a 3d space (semi-propagate)
    @param P            : Array of points of a plane surface
    @param V            : Array of reconstruction points
    @param wl_v         : Array of wavelength frequencies for each fH component
                          in the first axis
    @param desc         : Description to show in the progress bar. If none,
                          it shows no progress bar 
    @param n_threads    : Number of threads to use with the propagation. 
                          By default is 1.
    @return             : A RSD propagation given fH values, distributed as P,
                          to the 3d volume V, with wavelengths wl
    """  
    # Parallel processing execution function
    pool_propagate_plane = partial(__propagate_plane_i, V, P, fH_reshp,
                                    wl_v, fH_3d)
    # For each plane in V
    with ThreadPoolExecutor(max_workers = n_threads) as executor:
        fI = np.array( list(
                tqdm(executor.map(pool_propagate_plane, range(V.shape[0])), 
                            desc = desc,
                            disable = desc is None,
                            unit = "plane",
                            total = V.shape[0]
                )))
    return np.array(fI).swapaxes(0,1)


def reconstruct( H:  np.ndarray, t_bins:  np.ndarray, S:  np.ndarray,
                 L:  np.ndarray,  V: np.ndarray, lambda_c: float = 6,
                 cycles: float = 4, f_results: bool = False,
                 n_threads: int = 1, verbose: int = 0):
    """
    Returns a NLOS solver object based on Phasor Fields
    @param H            : Array with the impulse response of the scene by time,
                          camera points and laser points
    @param t_bins       : Temporal stamp for each bin 
    @param S            : Coordinates for each sensor point in z y x in the 
                          relay wall
    @param L            : Coordinates for each light source point in z y x in
                          the relay wall
    @param V            : Voxelization of the reconstructions
    @param lambda_c     : Central wavelength for the virtual illumination 
                          gaussian pulse for the reconstruction
    @param cycles       : Number of cycles of the virtual illumination pulse
    @param f_results    : Iff true, returns the data in the fourier domain 
                          (with the first axis for each frequency). It returns
                          the result in time domain 0 otherwise
    @param n_threads    : Number of threads to use on the reconstruction
    @param verbose      : Set the level of verbose:
                            - 0: prints nothing
                            - 1: informs about the start of the proceedings
                            - 2: 1 and gives information about the pulse, and
                                 shows progress bars for the reconstructions
    @return             : A Phasor Fields approach reconstruction of a NLOS 
                          problem
    """
    assert (V.ndim == 4 or V.ndim == 2) and V.shape[-1] == 3, \
            "reconstruct does not support V data"
    assert (S.ndim == 3 or S.ndim == 2) and S.shape[-1] == 3,\
            "reconstruct does not support this S data"
    assert (L.ndim == 3 or L.ndim == 2) and L.shape[-1] == 3,\
            "reconstruct does not support this L data"

    # Reshape all the data to match all the data
    S_r = S; L_r = L
    if S_r.ndim == 2: S_r = S_r.reshape(-1, 1, 3)
    if L_r.ndim == 2: L_r = L_r.reshape(-1, 1, 3)
    H_r = __H_format(H, S, L)


    # Time domain impulse response to Fourier
    __v_print(f"Generating virtual illumination pulse:\n" + 
              f"\tCentral wavelength: {lambda_c} m\n" + 
              f"\t{cycles} cycles\n...", 2, verbose)

    f_H, wv, f_pulse, sig_idx = H_to_fH(H_r, t_bins, lambda_c, cycles)

    __v_print(f"Done. {len(wv)} frequencies to use", 2, verbose)
    
    __v_print("Generating propagator from sensors...", 2, verbose)
    if S_r.shape == V[0].shape and __parallel(S_r, V[0]):
        propagator_S = RSD_parallel_propagator(S_r, V, wv)
        __v_print("Parallel RSD propagator", 2, verbose)
    else:
        propagator_S = RSD_propagator()
        __v_print("Points RSD propagator", 2, verbose)
    __v_print("Done", 2, verbose)
    __v_print("Generating propagator from lights...", 2, verbose)
    if L_r.shape == V[0].shape and __parallel(L_r, V[0]):
        propagator_L = RSD_parallel_propagator(L_r, V, wv)
        __v_print("Parallel RSD propagator", 2, verbose)
    else:
        propagator_L = RSD_propagator()
        __v_print("Points RSD propagator", 2, verbose)
    __v_print("Done", 2, verbose)
    __v_print(f"Propagating with {n_threads} threads", 2, verbose)

    for V_z in tqdm(V, disable = verbose != 2, unit='plane', total = V.shape[0]):
        fI_s = propagator_S.propagate(f_H, S_r, V_z, wv, P_axis=(1,2))
        print(fI_s.shape)
        fI = propagator_L.propagate(fI_s, L_r, V_z, wv, P_axis=(1,2), V_axis=(3,4))
        print(fI.shape)

    adfadfae

    # Propagate from sensors
    if S.ndim == 3 and V.ndim == 4 and __parallel(S, V[0]): # Parallel planes
        desc = __v_desc("Reconstructing from Sensors", 2, verbose)
        fI_s = propagate_parallel_planes(f_H, S_r, V, wv, n_threads=n_threads,
                                        desc = desc)
    else:                       # Propagate as sparse points
        fI_s = propagate(f_H, S, V, wv, n_threads=n_threads,
                    desc = __v_desc("Reconstructing from Sensors", 2, verbose))

    del f_H

    __v_print("Done\nPropagating from Light...", 1, verbose) 

    # Propagate from light source
    if L.ndim == 3 and V.ndim == 4 and __parallel(L, V[0]): # Parallel planes
        desc = __v_desc("Reconstructing from Lights", 2, verbose)
        fI = propagate_parallel_planes(fI_s, L_r, V, wv, n_threads=n_threads,
                                        desc = desc)
    else:                   # Propagate as sparse points
        fI = propagate(fI_s, L, V, wv, n_threads=n_threads,
                    desc = __v_desc("Reconstructing from Lights", 2, verbose))
    
    __v_print("Done", 1, verbose) 

    # Release space
    del fI_s

    if f_results:  
        # Data in frequency domain
        return fI
    else:
        __v_print("Transforming reconstruction to time space...", 1, verbose)

        # Data to time domain
        I = fI_to_I(fI, f_pulse, sig_idx, n_threads, 
                desc = __v_desc("Reconstruction to time domain", 2, verbose))

        __v_print("Done", 1, verbose)
        return I


###############################################################################
#                       Auxiliary methods and functions                       #
###############################################################################

# Prints only iff threshold =< given_verbose
def __v_print(msg, threshold, given_verbose):
    if threshold <= given_verbose:
        print(msg)


# Return the given text iff threshold =< given_verbose. Else it returns None
def __v_desc(desc, threshold, given_verbose):
    if threshold <= given_verbose:
        return desc
    else:
        None

# Auxiliary function to propagate planes using parallel process
def __propagate_plane_i(V, P, fHz, wl_v, fH_3d, hd_z):
    return propagate_plane(V[hd_z], P, fHz[:, hd_z*fH_3d], wl_v)


# Auxiliar function to be used with i_map, to propagate the Kernel to the 
# plane hd_z, with data ffH, and return the corner matrix of shape <<shape>>
def __f_propagate_plane_i(K, ffH, ffH_3d, shape,  hd_z):
    return K.f_propagate_i(ffH[:, hd_z*ffH_3d], hd_z)\
            [:, -shape[0]:, -shape[1]:]

# Auxiliary function to transform fourier to prime planes, 
# given the pulse parameters
def __fIz2Iz(fIz: np.ndarray, f_pulse: np.ndarray, 
             significant_idx: np.ndarray):
    """
    Transforms the plane reconstruction I in Fourier domain to time domain
    @param fIz              : A plane reconstruction of the scene by frequency
                              domain
    @param f_pulse          : Virtual ilumination pulse in frequency space
    @param significant_idx  : Indices with enough significance in the virtual
                              illumination pulse to consider, in the numpy fft
                              of the original signal
    @return                 : The plane reconstruction of the scene in time 
                              domain, evaluated in t=0
    """
    # Shape of the 3d volume
    (_, ny, nx) = fIz.shape
    # All frequencies to use
    nw_all = f_pulse.shape[0]
    # All no indicated frequencies values are 0
    all_freq = np.zeros((nw_all, ny, nx), dtype = np.complex128)

    # Fill the significant data weighted with the illumination
    # pulse value
    all_freq[significant_idx] = fIz*f_pulse[significant_idx,
                                            np.newaxis, np.newaxis]
    # Inverse FFT to return the data, evaluated at time 0
    return np.fft.ifft(all_freq, axis = 0)[0]

# Update the format of H to the dimensions of S and L
def __H_format(H: np.ndarray, S: np.ndarray, L: np.ndarray):
    n_bins_shape = (H.shape[0], )

    rec_S = S.ndim == 3
    if rec_S:
        S_shape = (S.shape[0], S.shape[1])
    else:
        S_shape = (S.shape[0], 1)

    rec_L = L.ndim == 3
    if rec_L:
        L_shape = (L.shape[0], L.shape[1])
    else:
        L_shape = (L.shape[0], 1)

    return H.reshape(n_bins_shape + S_shape + L_shape)


# Return true if the two planes are parallel
def __parallel(S1, S2):
        # Vectors in plane 1
        p1 = S1[0,0]; p2 = S1[0,-1]; p3 = S1[-1,0]
        v1 = p3 - p1
        v2 = p2 - p1
        # Normal to the plane
        n1 = np.cross(v1,v2)

        # Vectors in plane 2
        p1 = S2[0,0]; p2 = S2[0,-1]; p3 = S2[-1,0]
        v1 = p3 - p1
        v2 = p2 - p1
        # Normal to the plane
        n2 = np.cross(v1,v2)

        # Return similar to 0 with error
        return np.linalg.norm(np.cross(n1, n2)) <= 1e-8