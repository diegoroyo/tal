"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   pf_solver.py
Description :   Solves the NLOS problem using the phasor fields approximation,
                applying Rayleigh-Sommerfeld propagation to fill the given 
                voxelization
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from typing import Union, Tuple

from tal.reconstruct.pf.propagator import Propagator, RSD_propagator
from tal.reconstruct.pf.propagator import RSD_parallel_propagator


def pulse(delta_t: float, n_w: int, lambda_c: float, cycles: float
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    wavelengths[0] = np.inf
    w[0] = 0.0
    # Frequency distances to central frequency in rad/s
    delta_w = 2*np.pi*(w - w_c)
    # Gaussian pulse for reconstruction
    freq_pulse = np.exp(-sigma**2*delta_w**2/2)*np.sqrt(2*np.pi)*sigma

    # Central freq and sigma to k up to numpy fft
    central_k = delta_t * n_w * w_c
    sigma_k = delta_t * n_w / (2*np.pi*sigma)

    # Interest indicies in range [mu-3sigma, mu+3sigma]
    min_k = np.round(central_k - 3*sigma_k)
    max_k = np.round(central_k + 3*sigma_k)
    # Indices in the gaussian pulse
    indices = np.arange(min_k, max_k+1, dtype=int)

    return (freq_pulse, wavelengths, indices)


def H_to_fH(H: np.ndarray, t_bins: np.ndarray,  lambda_c: float,
            cycles: float
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                        Union[np.ndarray, None]]:
    """
    Transforms the data H in time, to the desired Fourier space frequencies
    @param H        : The impulse response of the scene
    @param t_bins   : Time of each temporal bin
    @param lambda_c : The wavelength of the central frequency for the virtual
                      illumination pulse
    @param cycles   : Number of cycles of the virtual illumination pulse
    @return         : Tuple containting first the impulse response in fourier
                      domain for those non zero values, second the wavelengths
                      associated with each fourier component, third the pulse
                      value, and fourth the indices of the pulse according the
                      fft module of numpy
    """
    if cycles == 0:
        # Fourier term function
        fourier_term = np.exp(-2 * np.pi * t_bins / lambda_c * 1j)
        fH = np.array([np.sum(H*fourier_term.reshape((-1,)+(1,)*(H.ndim - 1)),
                              axis=0)])
        wl = np.array([lambda_c])
        f_pulse = np.ones(1)
        significant_idx = None
    else:
        fH_all = np.fft.fft(H, axis=0)
        f_pulse, wv_all, significant_idx = pulse(t_bins[1], fH_all.shape[0],
                                                 lambda_c, cycles)
        wl = wv_all[significant_idx]
        fH = fH_all[significant_idx]

    return (fH, wl, f_pulse, significant_idx)


# Auxiliary function to transform fourier to prime planes,
# given the pulse parameters
def fIz2Iz(fIz: np.ndarray, f_pulse: np.ndarray, significant_idx: np.ndarray, 
            idx_t : np.ndarray = np.array([0])) -> np.ndarray:
    """
    Transforms the plane reconstruction I in Fourier domain to time domain
    @param fIz              : A plane reconstruction of the scene by frequency
                              domain
    @param f_pulse          : Virtual ilumination pulse in frequency space
    @param significant_idx  : Indices with enough significance in the virtual
                              illumination pulse to consider, in the numpy fft
                              of the original signal
    @param idx_t            : Evaluation in fft indices of the time
    @return                 : The plane reconstruction of the scene in time 
                              domain, evaluated at the time given by idx_t
    """
    # Shape of the volume
    fIz_shape = tuple(np.array(fIz.shape)[1:])
    # All frequencies to use
    nw_all = f_pulse.shape[0]
    nw_sig_max = np.max(significant_idx)
    # All no indicated frequencies values are 0
    all_freq = np.zeros((nw_sig_max + 1), dtype=np.complex128)
    Iz = np.zeros(idx_t.shape + fIz_shape, dtype=np.complex128)

    # Index along points of fIz
    for it in np.ndindex(fIz_shape):
        exp_it = (np.s_[:],) + it
        fIp = fIz[exp_it]
        # Fill the significant data weighted with the illumination
        # pulse value
        all_freq[significant_idx] = fIp*f_pulse[significant_idx]
        # Inverse FFT to return the data, evaluated at time 0
        Iz[exp_it] = np.fft.ifft(all_freq, n=nw_all)[idx_t]

    return Iz


def fI_to_I(fI: np.ndarray, f_pulse: np.ndarray, sig_idx: np.ndarray,
            idx_t : np.ndarray = np.array([0]), n_threads: int = 1,
            desc: str = "Fourier to time reconstruction"
            ) -> np.ndarray:
    """
    Transforms the volume reconstruction I in Fourier domain to time domain
    executed in parallel
    @param fI           : A volume reconstruction of the scene by frequency 
                          domain (in the first axis)
    @param f_pulse      : Pulse used for the reconstruction in Fourier domain
    @param sig_idx      : Iff not None, indices that correspond in the f_pulse
                         to the most significant values of the pulse
    @param idx_t        : Evaluation in fft indices of the time
    @param n_threads    : Number of threads to use for the function
    @param desc         : Description to show in the progress bar. If None, it
                          shows no progress bar 
    @return             : The volume reconstruction of the scene in time 
                          domain, evaluated in t. By default t=[0]
    """
    assert fI.ndim >= 2 and fI.ndim <= 4, "This data type is not supported"
    if sig_idx is None:    # Single frequency
        # No need to convert the data
        return fI[0]
    else:                   # Multiple frequency
        assert fI.shape[0] == len(sig_idx), "Different number of frequencies"
        with ProcessPoolExecutor(max_workers=n_threads) as ex:
            fIz2Iz_partial = partial(fIz2Iz,
                                     f_pulse=f_pulse,
                                     significant_idx=sig_idx,
                                     idx_t=idx_t)
            units = {1: 'points', 2: 'rows', 3: 'planes'}
            return np.array(
                    list(
                        tqdm(
                            ex.map(fIz2Iz_partial, fI.swapaxes(0, 1)),
                                    desc=desc,
                                    disable=desc is None,
                                    unit=units.get(fI.ndim - 1, ' '),
                                    total=fI.shape[1]))).swapaxes(0,1)


def propagator(P: np.ndarray, V: np.ndarray, wl: np.ndarray) -> Propagator:
    """
    Returns the proper RSD propagator given P, V and wl
    @param P    : Origin points to performance the propagator. It can be an 
                  array or a matrix of 3d coordinates
    @param V    : Destination points to propagate. It can be an array, a
                  matrix or a 3d volume of 3d coordinates
    @param wl   : Array of wavelengths to propagate in metters
    @return     : Propagator object
    """
    assert P.ndim >= 2 and P.ndim <= 3 and P.shape[-1] == 3, \
        "Unsupported data format"
    assert V.ndim >= 2 and V.ndim <= 4 and V.shape[-1] == 3, \
        "Unsupported data format"
    planes = True
    if P.ndim == 3:
        if V.ndim == 4:
            target_plane = V[0]
        elif V.ndim == 3:
            target_plane = V
        else:
            planes = False
    else:
        planes = False

    if planes and target_plane.shape == P.shape \
            and target_plane.shape >= (2, 2, 3) and P.shape >= (2, 2, 3) \
            and __parallel(target_plane, P):
        # Check the planes are in front
        v1 = target_plane[0, 0] - P[0, 0]
        v2 = target_plane[-1, 0] - P[-1, 0]
        v3 = target_plane[0, -1] - P[0, -1]
        v4 = target_plane[-1, -1] - P[-1, -1]
        if np.allclose(v1, v2) and np.allclose(v3, v2) and np.allclose(v3, v4):
            # Parallel propagator
            return RSD_parallel_propagator(P, V, wl)

    return RSD_propagator()


def reconstruct(H:  np.ndarray, t_bins:  np.ndarray, S:  np.ndarray,
                L:  np.ndarray,  V: np.ndarray, lambda_c: float = 6,
                cycles: float = 4, res_in_freq: bool = False,
                n_threads: int = 1, verbose: int = 0) -> np.ndarray:
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
    @param res_in_freq  : Iff true, returns the data in the fourier domain 
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
    assert (V.ndim <= 4 and V.ndim >= 2) and V.shape[-1] == 3, \
        "reconstruct does not support V data"
    assert (S.ndim == 3 or S.ndim == 2) and S.shape[-1] == 3,\
        "reconstruct does not support this S data"
    assert (L.ndim == 3 or L.ndim == 2) and L.shape[-1] == 3,\
        "reconstruct does not support this L data"

    # Reshape all the data to match all the data
    S_r = S
    L_r = L
    if S_r.ndim == 2:
        S_r = S_r.reshape(-1, 1, 3)
    if L_r.ndim == 2:
        L_r = L_r.reshape(-1, 1, 3)
    H_r = __H_format(H, S, L)

    units = {1:'points', 2: 'rows', 3:'planes'}
    unit = units.get(V.ndim-1, ' ')

    # Time domain impulse response to Fourier
    __v_print(f"Generating virtual illumination pulse:\n" +
              f"\tCentral wavelength: {lambda_c} m\n" +
              f"\t{cycles} cycles\n...", 1, verbose)

    f_H, wv, f_pulse, sig_idx = H_to_fH(H_r, t_bins, lambda_c, cycles)
    del H_r

    __v_print(f"Done. {len(wv)} frequencies to use", 1, verbose)

    __v_print("Generating propagator from sensors...", 2, verbose)
    propagator_S = propagator(S_r, V, wv)
    __v_print("Done", 2, verbose)
    __v_print("Generating propagator from lights...", 2, verbose)
    propagator_L = propagator(L_r, V, wv)
    __v_print("Done", 2, verbose)
    __v_print(f"Propagating with {n_threads} threads...", 1, verbose)

    # Propagate using multithreading
    propagate_partial = partial(__propagate, propagator_S, propagator_L, f_H,
                                S_r, L_r, wv)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        fI = np.array(
            list(tqdm(
                executor.map(propagate_partial, V),
                desc=unit + ' reconstructed',
                disable=verbose < 3,
                unit=unit,
                total=V.shape[0]
            )))
    del S_r, L_r, f_H
    __v_print("Done", 1, verbose)

    if not res_in_freq and sig_idx is not None:     # Result in time domain
        __v_print(f"Transforming from Fourier to time domain with {n_threads}"
                  + " threads...", 1, verbose)
        if verbose >= 3:
            desc = unit + ' to time domain'
        else:
            desc = None
        # Transform the data to Fourier domain
        I = fI_to_I(fI.swapaxes(0,1), f_pulse, sig_idx, n_threads = n_threads, 
                    desc = desc)[0]
        __v_print("Done", 1, verbose)
        return I
    elif not res_in_freq and sig_idx is None:
        return fI[:, 0, ...]
    else:
        return fI


###############################################################################
#                       Auxiliary methods and functions                       #
###############################################################################

# Propagate given the propagators, the impulse response in fourier
# f_H, the sensor points S, the light points L, the target coordinates V_z,
# with frecuency wavelengths wv
def __propagate(propagator_S, propagator_L, f_H, S, L, wv, V):
    # Propagate from sensors
    fI_s = propagator_S.propagate(f_H, S, V, wv, P_axis=(1, 2))
    # Propagate from Lights
    v_axis = tuple(np.arange(-V.ndim+1, 0))
    if len(v_axis) == 0:
        v_axis = (-1,)
    fI = propagator_L.propagate(fI_s, L, V, wv, P_axis=(1, 2), V_axis=v_axis)
    return fI


# Prints only iff threshold =< given_verbose
def __v_print(msg, threshold, given_verbose):
    if threshold <= given_verbose:
        print(msg)


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
    p1 = S1[0, 0]
    p2 = S1[0, -1]
    p3 = S1[-1, 0]
    v1 = p3 - p1
    v2 = p2 - p1
    # Normal to the plane
    n1 = np.cross(v1, v2)

    # Vectors in plane 2
    p1 = S2[0, 0]
    p2 = S2[0, -1]
    p3 = S2[-1, 0]
    v1 = p3 - p1
    v2 = p2 - p1
    # Normal to the plane
    n2 = np.cross(v1, v2)

    # Return similar to 0 with error
    return np.linalg.norm(np.cross(n1, n2)) <= 1e-8
