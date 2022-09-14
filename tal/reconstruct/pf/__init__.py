from tal.io.capture_data import NLOSCaptureData
from tal.reconstruct.pf.propagator import Propagator
from typing import Any, Tuple
import numpy as np


def to_Fourier(data: NLOSCaptureData,
               wavefactor: float,
               pulse_cycles: float) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Given the data, and the pulse defined by the wavefactor and cycles, return
    the data in the Fourier domain and the wavelengths associated to each 
    component
    @param data         : NLOS captured data
    @param wavefactor   : Wavefactor of the central frequency of the virtual
                          gaussian illumination pulse. The wavelength value is
                          wavefactor times sensor spacing of the relay wall
    @param pulse_cycles : Cycles of the virtual gaussian illumination pulse.
                          It defines the number of frequencies to use. If it 
                          is set to 0, it will use only a frequency. Otherwise
                          the lower the more frequencies
    @return             : A tuple, with the first element being the different
                          Fourier components of H given the pulse, the second
                          the different wavelengths associated to each 
                          component, and the third auxiliary information to
                          performance the inverse operation
    """
    from tal.reconstruct.pf.pf_solver import H_to_fH
    # Data extraction
    H = data.H
    # Time extraction
    if data.t_start is not None:
        # Correct delayed data
        t_pad = int(np.round(data.t_start / data.delta_t))
        H = np.pad(H, ((t_pad, 0), (0, 0), (0, 0)), 'minimum')
    T = data.delta_t*np.arange(H.shape[0])
    # Calculate the wavelength of the central frequency of the pulse
    # FIXME: Minimun distance assumes ordered points
    reshaped_S = data.sensor_grid_xyz.reshape(-1, 3)
    lambda_c = wavefactor * np.linalg.norm(reshaped_S[0] - reshaped_S[1])
    # Transform to Fourier domain and extract the wavelengths and auxiliary
    # parameters
    fH, wl, f_pulse, sig_idx = H_to_fH(H, T, lambda_c, pulse_cycles)
    return fH, wl, (f_pulse, sig_idx, data.delta_t)


def to_time(fourier_comp: np.ndarray,
            aux_param: Any,
            n_threads: int = 1,
            t_eval: np.ndarray = None) -> np.ndarray:
    """
    Given the Fourier components it transforms the data to time domain, and
    returns the evaluations at the given time
    @param fourier_comp : Fourier components data to transform to time domain
    @param aux_param    : Auxiliary parameters used to transform it back to
                          the original time domain
    @param n_threads    : Number of threads to use for the transformation
    @param t_eval       : Array of time values to evaluate the transformation
    @return             : Array of results of the evaluations for the given
                          time stamps at the data in time. If there is only a
                          frequency, it return the first element of the 
                          array
    """
    from tal.reconstruct.pf.pf_solver import fI_to_I
    # Extract the auxiliary params
    f_pulse, sig_idx, delta_t = aux_param
    # Approximate the time stamps to the fft index
    t_eval = t_eval or np.array([0.0])
    t_eval_it = int(np.round(t_eval / delta_t))
    # Return the result in time domain
    return fI_to_I(fourier_comp, f_pulse, sig_idx, t_eval_it, n_threads)


def get_propagators(data: NLOSCaptureData,
                    voxels: np.ndarray,
                    wl: np.ndarray) -> Tuple[Propagator, Propagator]:
    """
    Return the propagators from the sensor points, and from the light sources
    of the data to the voxels
    @param data     : NLOS captured data 
    @param voxels   : Target voxels to obtain the propagators
    @param wl       : Array of wavelengths that will propagate
    @return         : Tuple of propagators, being the first one the sensor
                      points to voxels propagator, and the second the light
                      sources to the voxels propagator
    """
    from tal.reconstruct.pf.pf_solver import propagator
    S = data.sensor_grid_xyz
    L = data.laser_grid_xyz
    return (propagator(S, voxels, wl), propagator(L, voxels, wl))


def solve(data: NLOSCaptureData, wavefactor: float, wave_cycles: float,
          volume: NLOSCaptureData.VolumeType = None, res_in_freq=False,
          verbose: int = 1, n_threads: int = 1) -> np.ndarray:
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
    from tal.reconstruct.pf.pf_solver import reconstruct
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

    return reconstruct(H, T, S, L, V, wavelength, wave_cycles, res_in_freq,
                       n_threads, verbose)
