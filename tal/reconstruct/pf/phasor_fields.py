"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   pf_solver.py
Description :   Solves the NLOS problem using the phasor fields approximation,
                applying Rayleigh-Sommerfeld propagation to fill the given 
                voxelization
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import CameraSystem, VolumeFormat
from tal.reconstruct.filters import HFilter
from tal.log import log, LogLevel
from tal.config import get_resources

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
  
from tal.reconstruct.pf.propagator import Propagator, PropParams


__epsilon = 1e-5
c = 299_792_458

def reconstruct(data: NLOSCaptureData, filter: HFilter, V: np.ndarray,
                volume_format: VolumeFormat,
                camera_system: CameraSystem,
                Vp:np.ndarray = None,
                by_point = False,
                verbose: int = 0) -> np.ndarray:
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

    # Estimate if reconstruct forced by point
    force_by_point = False
    V_prime = V
    if volume_format == VolumeFormat.X_Y_Z_3:
        force_by_point = by_point

    if force_by_point:
        V_prime = V.reshape(-1, 3)

    # Define the reconstruction parameters
    propParams = PropParams(V_prime, Vp, volume_format, camera_system,
                             force_by_point, data.is_confocal())
    # Check the filter exists
    if not filter.is_configured():
        log(LogLevel.INFO, "Filter is not configured. Using central wavelength of 10 cm and 5 cycles")
        filter.gaussian_package_filter(data.delta_t, data.H.shape[0], 0.1, 5)

    log(LogLevel.DEBUG, "Applying the filter")
    fH = filter.apply(data, True)
    pf_propagator = Propagator(filter, propParams)
    
    # Configure the propagator
    pf_propagator.configure(data)

    # Multiprocessing
    cpu_processes = get_resources().cpu_processes
    if cpu_processes is None or cpu_processes == 1:
        return pf_propagator.propagate_v(fH, np.arange(pf_propagator.iter_size()))
    
    else:
        # Prepare the reconstruction memory
        t_shape = ()
        if camera_system in [CameraSystem.CONFOCAL_TIME_GATED, 
                            CameraSystem.PROJECTOR_CAMERA, 
                            CameraSystem.TRANSIENT, 
                            CameraSystem.PROJECTOR_ONLY]:
            t_shape = (data.H.shape[0],)

        I = np.zeros(t_shape + V.shape[:-1], dtype = np.complex128)
        partial_propagation = partial(pf_propagator.apply, fH)
        with tqdm(total=pf_propagator.iter_size()) as pbar:
            # let's give it some more threads:
            with ThreadPoolExecutor(max_workers=get_resources().cpu_processes) as executor:
                futures = {executor.submit(partial_propagation, arg): arg for arg in np.arange(pf_propagator.iter_size())}
                for future in as_completed(futures):
                    iter = futures[future]
                    I[..., iter] = future.result()
                    pbar.update(1)
                    
        return I