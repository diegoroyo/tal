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
    

def parallel_FFT2d(K: np.ndarray, t: int, s = None, 
                    desc: str = "FFT by plane"):
    """
    Given a list of planes K, it performance in parallel the FFT in 2D
    with t threads
    @param  K       : Array of planes to performance the 2D FFT
    @param  t       : Number of threads to use for the calculus
    @param  s       : Shape of each transformed axis. Int or sequence of ints
    @param  desc    : Description to show in the progress bar. If none, it 
                      shows no progress bar 
    @return         : K planes in Fourier domain
    """
    with Pool(t) as p:
        fft2_shaped = partial(np.fft.fft2, s=s)
        return np.array(
            list(
                tqdm(
                    p.imap(fft2_shaped, K),
                    desc = desc,
                    disable = desc is None,
                    unit = "Planes",
                    total = K.shape[0]
            )))


def parallel_iFFT2d(K: np.ndarray, t: int, s = None, 
                    desc = "iFFT by plane"):
    """
    Given a list of planes K, it performance in parallel the iFFT in 2D
    with t threads
    @param  K       : Array of planes to performance the 2D iFFT
    @param  t       : Number of threads to use for the calculus
    @param  s       : Shape of each transformed axis. Int or sequence of ints
    @param  desc    : Description to show in the progress bar. If none, it 
                      shows no progress bar 
    @return         : K planes in primal domain
    """
    with Pool(t) as p:
        ifft2_shaped = partial(np.fft.ifft2, s=s)
        return np.array(
            list(
                tqdm(
                    p.imap(ifft2_shaped, K),
                    desc = desc,
                    disable = desc is None,
                    unit = "Planes",
                    total = K.shape[0]
            )))


@njit(parallel=True)
def conv_fRSD(fK_rsd: np.ndarray, H_f: np.ndarray):
    """
    Given a plane, and the propagator kernels for the frequencies in 
    Fourier domain, it convolve with the impulse response H in Fourier
    domain
    @param fK_rsd   : RSD propagator kernels in fourier domain to a single
                        plane
    @param H_f      : Impulse response in fourier domain. The frequencies
                        match in the first axis with fK_rsd first axis
    @return         : Convolution in Fourier domain of the RSD propagator's
                        plane
    """
    return fK_rsd * H_f


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
        fH = np.array([np.sum(H*fourier_term[:, np.newaxis, np.newaxis],
                             axis = 0)])
        wv = np.array([lambda_c])
        aux_data = None
    else:
        fH_all = np.fft.fft(H, axis = 0)
        f_pulse, wv_all, significant_idx = pulse(t_bins[1], fH_all.shape[0],
                                        lambda_c, cycles)
        wv = wv_all[significant_idx]
        fH = fH_all[significant_idx]
        aux_data = (f_pulse, significant_idx)
    
    return (fH, wv, aux_data)


def fIz2Iz(fIz: np.ndarray, f_pulse: np.ndarray, significant_idx: np.ndarray):
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


def fI2I(fI: np.ndarray, aux_data, n_threads: int = 1, 
        desc: str = "Fourier to time reconstruction by planes"):
    """
    Transforms the volume reconstruction I in Fourier domain to time domain
    executed in parallel
    @param fI       : A volume reconstruction of the scene by frequency domain
    @param aux_data : Auxiliary data obtained at the fourier transformation
                      of the impulse response. Tuple with pulse and indices
    @param  desc    : Description to show in the progress bar. If none, it 
                      shows no progress bar 
    @return         : The volume reconstruction of the scene in time domain, 
                      evaluated in t=0
    """
    if aux_data is None:    # Single frequency
        # No need to convert the data
        return fI[0]
    else:                   # Multiple frequency
        f_pulse, significant_idx = aux_data
        with Pool(n_threads) as p:
            fIz2Iz_partial = partial(fIz2Iz,
                                    f_pulse = f_pulse, 
                                    significant_idx = significant_idx)
            return np.array(list(tqdm(p.imap(fIz2Iz_partial, fI.swapaxes(0,1)),
                                        desc=desc,
                                        disable = desc is None,
                                        unit = "plane",
                                        total = fI.shape[1])))

# Auxiliar function to be used with i_map, to propagate the Kernel to the 
# plane hd_z, with data ffH, and return the corner matrix of shape <<shape>>
def __f_propagate_plane_i(K, ffH, ffH_3d, shape,  hd_z):
    return K.f_propagate_i(ffH[:, hd_z*ffH_3d], hd_z)\
            [:, -shape[0]:, -shape[1]:]


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
    K_d= np.sqrt(np.sum((Vz - P)**2, axis = -1))
    # Calculate the RSD kernel
    K_rsd = np.zeros((len(wl_v),) + K_d.shape, dtype = np.complex128)
    for hd_w, wl in enumerate(wl_v):
        K_rsd[hd_w] = RSD_kernel.RSD_kernel_w(K_d, wl)
    # Return the RSD propagation from Hz plane
    return np.sum(fHz*K_rsd, axis = -1)


# Auxiliary function to propagate planes using parallel process
def __propagate_plane_i(V, P, fHz, wl_v, fH_3d, hd_z):
    return propagate_plane(V[hd_z], P, fHz[:, hd_z*fH_3d], wl_v)


def propagate(fH: np.ndarray, P: np.ndarray, V: np.ndarray,
                wl_v: np.ndarray, n_threads: int = 1, 
                desc: str = "Reconstructing planes"):
    """
    Propagate the impulse response in Fourier space to the volume V
    voxelization, whith wavelengths wl. The first axis of fH and wl
    match for each frequency used.
    @param fH           : The impulse response in the relay wall in Fourier
                          domain. The first axis represent each frequency wl.
                          It can be a 2d plane, or a 3d space (semi-propagate)
    @param P            : Array of points of a plane surface
    @param V            : 3d volume
    @param wl_v         : Array of wavelength frequencies for each fH component
                          in the first axis
    @param desc         : Description to show in the progress bar. If none,
                          it shows no progress bar 
    @param n_threads    : Number of threads to use with the propagation. 
                          By default is 1.
    @return             : A RSD propagation given fH values, distributed as P,
                          to the 3d volume V, with wavelengths wl
    """  
    nP = P.shape[0]     # Number of points to propagate
    nw = len(wl_v)      # Number of frequencies
    # Shapes calculations
    fH_dims = fH.ndim - 1       # fH spatial dimensions. The first one is freq
    fH_3d = fH_dims == 3
    # Reshape for the transformation. It takes into account fH is 3D
    fH_reshp = fH.reshape((nw, V.shape[0] * fH_3d + (not fH_3d),
                            V.shape[1], V.shape[2], nP))
    # Broadcast for easier calculations
    P_b = np.broadcast_to(P,(V.shape[1], V.shape[2], P.shape[0], P.shape[1]))
    V_b = V[..., np.newaxis, :]

    # Parallel processing execution function
    pool_propagate_plane = partial(__propagate_plane_i, V_b, P_b, fH_reshp,
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


def reconstruct( H:  np.ndarray, t_bins:  np.ndarray, S:  np.ndarray,
                 L:  np.ndarray,  V: np.ndarray, lambda_c: float = 6,
                 cycles: float = 4, S_rec_shape: tuple = None, 
                 L_rec_shape: tuple = None, f_results: bool = False,
                 n_threads: int = 1, verbose: int = 0):
    """
    Returns a NLOS reconstruction with Phasor Fields
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
    @param S_rec_shape  : If not None, it cointains a tuple with the x y z 
                          shape of a rectangle the sensors form
    @param L_rec_shape  : If not None, it cointains a tuple with the x y z
                          shape of a rectangle the light sources form
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

    # Generate iterators of the given data as rectangle
    S_r = None
    L_r = None
    if not S_rec_shape is None:
        S_r = S.reshape(S_rec_shape + (3,))
    if not L_rec_shape is None:
        L_r = L.reshape(L_rec_shape + (3,))

    # Time domain impulse response to Fourier
    __v_print(f"Generating virtual illumination pulse:\n" + 
              f"\tCentral wavelength: {lambda_c} m\n" + 
              f"\t{cycles} cycles\n...", 2, verbose)

    f_H, wv, aux_pulse_data = H_to_fH(H, t_bins, lambda_c, cycles)

    __v_print(f"Done. {len(wv)} frequencies to use", 2, verbose)
    
    __v_print("Propagating from Sensors...", 1, verbose)

    # Propagate from sensors
    if not S_r is None:     # Parallel rectangle
        desc = __v_desc("Reconstructing from Sensors", 2, verbose)
        fI_s = propagate_parallel_planes(f_H, S_r, V, wv, n_threads=n_threads,
                                        desc = desc)
    else:                   # Array of points
        fI_s = propagate(f_H, S, V, wv, n_threads=n_threads,
                    desc = __v_desc("Reconstructing from Sensors", 2, verbose))

    __v_print("Done\nPropagating from Light...", 1, verbose) 

    # Propagate from light source
    if not L_r is None:     # Parallel rectangle
        desc = __v_desc("Reconstructing from Lights", 2, verbose)
        fI = propagate_parallel_planes(fI_s, L_r, V, wv, n_threads=n_threads,
                                        desc = desc)
    else:                   # Array of points
        fI = propagate(fI_s, L, V, wv, n_threads=n_threads,
                    desc = __v_desc("Reconstructing from Lights", 2, verbose))
    
    __v_print("Done", 1, verbose) 

    # Release space
    del fI_s, f_H

    if f_results:  
        # Data in frequency domain
        return fI
    else:
        __v_print("Transforming reconstruction to time space...", 1, verbose)

        # Data to time domain
        I = fI2I(fI, aux_pulse_data, n_threads, 
                desc = __v_desc("Reconstruction to time domain", 2, verbose))

        __v_print("Done", 1, verbose)
        return I