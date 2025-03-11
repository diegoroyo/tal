"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   propagator.py
Description :   Interface of propagation to use with Phasor Fields solver, and
                RSD implementations for Parallel and non Parallel planes
"""
from tal.io.capture_data import NLOSCaptureData
from tal.enums import CameraSystem, VolumeFormat
from tal.reconstruct.util import can_parallel_convolution
from tal.reconstruct.filters import HFilter
from tal.log import log, LogLevel

import pyfftw.interfaces.numpy_fft as np_fft

import numpy as np

from warnings import warn

import pyfftw

class PropParams(object):
    """
    Parameters for the propagation with the Phasor Fields.
    """
    def __init__(self, V:np.ndarray, Vp:np.ndarray|None = None,
                 volume_format: VolumeFormat = VolumeFormat.N_3, 
                 camera_system: CameraSystem = CameraSystem.DIRECT_LIGHT,
                 force_by_point: bool = False, confocal_capture: bool = False):
        """
        Parameters for the propagation of Phasor Fields
        :param V                : Volume definition for the Phasor Fields 
                                  propagation.
        :param Vp               : If proyection camera_system is 
                                  tal.enum.CameraSystem.PROJECTOR_X or 
                                  tal.enum.CameraSystem.PROJECTOR_ONLY, 
                                  propagates the illumination only to this
                                  points. Only tal.enum.VolumeFormat.N_3 
                                  supported.
        :param volume_format    : Define the format of V. See doc in 
                                  tal.enum.VolumeFormat.
        :param camera_system    : Define the camera format for the propagation.
                                  See doc in tal.enum.CameraSystem
        :param force_by_point   : Iff true, only point to point propagation 
                                  will be used.
        :param confocal_capture : Indicate if the data is confocal or not.
        """
        self.V = V
        self.Vp = Vp
        self.volume_format = volume_format
        self.camera_system = camera_system
        self.force_by_point = force_by_point
        self.confocal_capture = confocal_capture

class Propagator(object):
    """
    Propagator for Phasor Fields
    """
    def __init__(self, filter: HFilter, params: PropParams):
        self.params = params
        self.filter = filter

        self.__iterator_x = None
        self.confocal_capture = False
        

    def configure(self, data: NLOSCaptureData):
        log(LogLevel.INFO, "tal.reconstruct.pf.phasor_fields: Configuring the PF propagator.")
        if data.is_confocal() and not self.confocal_capture:
            warn('Detected confocal capture, but not indicated in the parameters. Parameters will be modified')
            self.params.confocal_capture = True 

        # Can the propagation be done by planar convolutions?
        s_parallel = can_parallel_convolution(data, self.params.V, 'sensor') \
                        and not self.params.force_by_point
        l_parallel = can_parallel_convolution(data, self.params.V, 'laser') \
                        and not self.params.force_by_point
        
        # Set the target coordinates
        if s_parallel or l_parallel:
            self.__iterator_x = np.moveaxis(self.params.V, 2, 0)
            iterator_z = self.__iterator_x[:,0,0,2]
            if s_parallel:
                self.__iterator_s = iterator_z
            else:
                self.__iterator_s = self.__iterator_x
            if l_parallel:
                self.__iterator_l = iterator_z
            else:
                self.__iterator_l = self.__iterator_x
        else:
            self.__iterator_x = self.params.V.reshape(-1,3)
            self.__iterator_s = self.__iterator_x
            self.__iterator_l = self.__iterator_x

        # Configure the propagators
        # Sensor propagators
        self.propagator_S = PropagatorCore.dummy()  # Null propagator
        self.S_prop_axes = None
        if self.params.camera_system != CameraSystem.PROJECTOR_ONLY:
            twice_prop = self.params.confocal_capture \
                            and self.params.camera_system in \
                                    [CameraSystem.CONFOCAL_TIME_GATED,
                                     CameraSystem.DIRECT_LIGHT]
            self.propagator_S = PropagatorCore(data.sensor_grid_xyz,
                                            self.filter.omega,
                                            self.__iterator_s,
                                            by_point = not s_parallel,
                                            target_shape = self.__iterator_x.shape[1:3],
                                            twice = twice_prop)
            self.S_prop_axes = (1,2)


        # Laser propagators
        self.propagator_L = PropagatorCore.dummy()
        self.L_prop_axes = None
        if self.params.camera_system in [CameraSystem.PROJECTOR_CAMERA,
                                         CameraSystem.PROJECTOR_CAMERA_T0,
                                         CameraSystem.PROJECTOR_ONLY]:
            # Propagate to Vp points
            # TODO: this is ineficent. Find better way to store it
            proy_dim = self.params.Vp.ndim
            tile_shape = (self.__iterator_x.shape[0],) + (1,)*proy_dim
            self.__iterator_l = np.tile(self.params.Vp, tile_shape)
            self.propagator_L = PropagatorCore(data.laser_grid_xyz,
                                                self.filter.omega,
                                                self.__iterator_l,
                                                by_point = True)
            self.L_prop_axes = (-2, -1)
                
        elif self.params.camera_system in [CameraSystem.DIRECT_LIGHT,
                                           CameraSystem.CONFOCAL_TIME_GATED] \
             and not self.params.confocal_capture:
            
            self.propagator_L = PropagatorCore(data.laser_grid_xyz,
                                            self.filter.omega,
                                            self.__iterator_l,
                                            by_point = not l_parallel,
                                            target_shape = self.__iterator_x.shape[1:3],
                                            twice = False)
            self.L_prop_axes = (-2, -1)
            
        log(LogLevel.DEBUG, f"tal.reconstruct.pf.phasor_fields.configure: Using sensor propagator {self.propagator_S}")
        log(LogLevel.DEBUG, f"tal.reconstruct.pf.phasor_fields.configure: Using laser propagator {self.propagator_L}")
        #### Configure the return signal
        self.store_time = True
        if self.params.camera_system in [CameraSystem.DIRECT_LIGHT, 
                                         CameraSystem.PROJECTOR_CAMERA_T0, 
                                         CameraSystem.TRANSIENT_T0]:
            log(LogLevel.DEBUG, "tal.reconstruct.pf.phasor_fields.configure: Reconstruction at t=0")
            # Return the reconstruction at t=0
            self.store_time = False
            self.to_time = self.t0
        else:
            log(LogLevel.DEBUG, "tal.reconstruct.pf.phasor_fields.configure: Reconstruction with complete time domain.")
            # Return the reconstruction with the temporal signal
            self.to_time = self.expand_fourier
            

    def propagate_v(self, fH, iter_v:np.nditer):
        # Apply the propagation
        expected_shape = self.__iterator_x.shape[1:-1] + (iter_v.shape[0],)
        if self.store_time:
            expected_shape = (self.filter.n_w,) + expected_shape
     
        result = np.zeros(expected_shape, dtype=np.complex128)
        for i, iter in enumerate(iter_v):
            result[..., i] = self.apply(fH, iter)

        return result

    def iter_size(self):
        return self.__iterator_x.shape[0]

    def static_proyect(self, fH, data):
        # Static proyection to a fix set of points
        if self.proyected_fH is None:  
            axis = np.arange(-(data.laser_grid_xyz.ndim - 1), 0)
            self.proyected_fH = np.sum(
                                    fH * Propagator.RSD_prop(data.laser_grid_xyz,
                                                             self.params.Vp,
                                                             self.filter.omega),
                                        axis = axis)
        return self.proyected_fH
        
    def expand_fourier(self, fV):
        # Expand the fourier indices given the filter
        full_fV = np.zeros((self.filter.n_w,) + fV.shape[1:], dtype=np.complex128)
        full_fV[self.filter.indices] = fV
        return np.fft.ifft(full_fV, axis = 0)      

    def t0(self, fV):
        return np.sum(fV, axis = 0)     


    def apply(self, fH, i):
        return self.to_time(
                    self.propagator_L.apply(
                                    self.propagator_S.apply(fH, i, self.S_prop_axes),
                            i, axes = self.L_prop_axes))


class PropagatorCore:
    """
    Given a grid or a list of points, it returns a propagation function that 
    can be later applied to the iterator i
    """
    def __init__(self, o_coords: np.ndarray, ang_freqs: np.ndarray,
                 target: np.ndarray, by_point: bool = False, 
                 target_shape: tuple = None, twice: bool = False):
        """
        Return a RSD propagation, by point or planar, given the origin 
        coordinates and the target cooordinates for the given frequencies.
        - o_coords      : (m,n,3) or (m,3) array of the origin coordinates
        - ang_freqs     : (f) array with the angular frequencies value
        - target        : (m,n,3) or (m,3) or (d) array of targets. It might be a
                          tensor of coordinates or an array of depths.
        - by_point      : If false, it return a planar RSD.
        - target_shape  : If by_point is False, it set this shape for the 
                          reconstruction target with planar RSD
        - twice         : If true, it assumes the propagation twice (as in 
                          confocal captures)
        """
        self.omega = ang_freqs
        self._prop_func = None  
        if by_point:
            self._o_coords = o_coords
            self._t_iterator = target
            if twice:
                self._propagator = self.RSD_prop_sq
            else:
                self._propagator = self.RSD_prop

            self._prop_func = self.__by_point_prop
        else:
            assert o_coords.ndim == 3 and o_coords.shape[-1] == 3,\
                    f"Unknown o_coords format of shape {o_coords.shape}"
            self._t_iterator = target
            self._zero_depth_RSD_coords = \
                        PropagatorCore.__zero_RSD_kernel_coords(o_coords,
                                                                target_shape)
            if twice:
                self._propagator = self.RSD_kernel_sq
            else:
                self._propagator = self.RSD_kernel

            self._prop_func = self.__planar_prop

 

    def __planar_prop(self, fH:np.ndarray, i:int, axes:tuple = (1,2)):
        """
        Given a fourier domain input, it propagates to the propagator at index i
        the given axes by planes
        - fH: (f, m, n) array, with f being the number of frequencies
        - i : Integer to select the propagator index
        - axes: Axes to apply fH propagation
        """
        K = self._propagator(i)
        return np.fft.ifft2(
                    np.fft.fft2(fH,s = K.shape[1:], axes = axes)\
                    *np.fft.fft2(K,axes = (1,2))
                )[:, -fH.shape[axes[0]]:, -fH.shape[axes[1]]:]

    def __by_point_prop(self, fH:np.ndarray, i:int, axes:tuple=None):
        """
        Given a fourier domain input, it propagates to the propagator at index i
        the given axes by planes
        - fH: (f, m, n) array, with f being the number of frequencies
        - i : Integer to select the propagator index
        - axes: Unused argument
        """
        sum_axes = tuple(np.arange(self._o_coords.ndim - 1))
        return np.sum(fH * self._propagator(i), axis = sum_axes)


    def RSD_kernel(self, i:int):
        """
        Given the coordinates at z distance = 0 from a grid, it extracts the 
        RSD kernel to propagate at a plane at z at omega frequencies
        """
        dist = np.linalg.norm(self._zero_depth_RSD_coords\
                                 + np.array([0,0,self._t_iterator[i]]), 
                              axis = -1)
        omega = self.omega.reshape((-1,) + (1,) * dist.ndim)
        return np.exp(omega*1j * dist)/dist
    

    def RSD_kernel_sq(self, i:int):
        """
        Given the coordinates at z distance = 0 from a grid, it extracts the 
        RSD kernel to propagate at a plane at z at omega frequencies squared
        """
        dist = np.linalg.norm(self._zero_depth_RSD_coords\
                                 + np.array([0,0,self._t_iterator[i]]), 
                              axis = -1)
        omega = self.omega.reshape((-1,) + (1,) * dist.ndim)
        return (np.exp(omega*1j * dist)/dist) ** 2
    

    def RSD_prop(self, i: int):
        """
        Given a grid and a list of points x, it returns the propagator with omega 
        frequencies
        """
        ndim = self._o_coords.ndim - 1
        pre_x_shape = self._t_iterator[i].shape
        rshp_x = self._t_iterator[i].reshape((1,)*(ndim + 1) + pre_x_shape)
        dist = np.linalg.norm(rshp_x -  self._o_coords, axis = -1)
        omega_rshp = self.omega.reshape((-1,) + (1,) * ndim)
        return np.exp(omega_rshp*1j * dist)/dist
    

    def RSD_prop_sq(self, i: int):
        """
        Given a grid and a list of points x, it returns the propagator with omega 
        frequencies squared
        """
        ndim = self._o_coords.ndim - 1
        pre_x_shape = self._t_iterator[i].shape
        rshp_x = self.t_iterator[i].reshape((1,)*ndim + pre_x_shape)
        dist = np.linalg.norm(rshp_x -  self.o_coords, axis = -1)
        omega_rshp = self.omega.reshape((-1,) + (1,) * dist.ndim)
        return (np.exp(omega_rshp*1j * dist)/dist)**2
    

    @staticmethod
    def __zero_RSD_kernel_coords(coords, t_shape = None):
        """
        Given coords, it calculate the coordinates for the kernel at distance 0.
        :param coords : The reference coordinates
        :param t_shape: The target shape (it must be bigger than coord.shape)
        :return       : Numpy.ndarray 
        """
        assert coords.ndim == 3 and coords.shape[2] == 3, "Unknown coordinates"
        # Shape of the coordinates
        ni, nj, _ = coords.shape
        # Estimate the delta between planes
        dist_i = np.linalg.norm(coords[0,0] - coords[-1,0])
        delta_i = dist_i/ni
        dist_j = np.linalg.norm(coords[0,0] - coords[0,-1])
        delta_j = dist_j/nj

        # Target shape, to use bigger kernels
        if t_shape is None:
            target_shape = (ni*2-1, nj*2-1)
        else: 
            target_shape = (t_shape[0]*2-1, t_shape[1]*2-1) 

        # The coordinates of the kernel at distance 0
        vi = np.linspace(-delta_i*(target_shape[0]//2) - delta_i, 
                         delta_i*(target_shape[0]//2) + delta_i,
                         target_shape[0])
        vj = np.linspace(-delta_j*(target_shape[1]//2) - delta_j, 
                         delta_j*(target_shape[1]//2) + delta_j,
                         target_shape[1])

        z_RSD_coords = np.moveaxis(np.array(np.meshgrid(vi, vj, [0], 
                                            indexing = 'ij') ), 0, -1)[:,:,0,:]
        return z_RSD_coords
    

    def prop_info(self):
        info_msg = ""
        if self._prop_func == self.__planar_prop:
            info_msg += "Planar RSD propagator"
        elif self._prop_func == self.__by_point_prop:
            info_msg += "By point RSD propagator"
        else:
            info_msg += "Not defined propagator"
        
        if self._propagator in [self.RSD_kernel_sq, self.RSD_kernel_sq]:
            info_msg += " squared."
        else:
            info_msg += "."
        return info_msg
    
    def __str__(self):
        return self.prop_info()

    def apply(self, fH, i, axes):
        # Apply the propagator
        return self._prop_func(fH, i, axes)
    
    @staticmethod
    def dummy():
        a = PropagatorCore(None, None, None, by_point = True)
        a._prop_func = lambda fH, i, axes: fH
        return a
