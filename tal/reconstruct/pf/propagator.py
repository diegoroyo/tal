"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   propagator.py
Description :   Interface of propagation to use with Phasor Fields solver, and
                RSD implementations for Parallel and non Parallel planes
"""

import numpy as np
from collections import defaultdict
from tal.reconstruct.pf.rsd_kernel import RSD_kernel
from typing import Tuple, Union

class Propagator(object):
    """
    Propagation interface
    """
    def propagate(self, fH: np.ndarray, P: np.ndarray,  
                    V: np.ndarray,  wl_v: np.ndarray,
                    axis: Union[int, Tuple[int]] = None) -> np.ndarray:
        """
        Propagate from P to V the value fH, where fH is the impulse response at
        P in the Fourier domain, and wl_v is the array of wavelengths for each
        fH component
        @param self : Propagator subclass implementation
        @param fH   : Impulse response in the Fourier domain
        @param P    : Coordinates of the fH samples
        @param V    : Coordinates to propagate fH
        @param wl_v : Array of wavelengths for each fH Fourier component
        @param axis : Axis of fH which represents P. If None, axis is assumed
                      to be the last one
        @return     : The propagation of fH to the V coordinates
        """
        raise NotImplementedError('Subclasses should implement propagate')


    @staticmethod
    def __reshape_data(fH: np.ndarray, P: np.ndarray, V: np.ndarray, 
                       axis: Union[int, Tuple[int]]):
        """
        Reshape the data to the needed form for the propagation implementations
        """
        raise NotImplementedError('Subclasses should implement reshape_data')
    

    @staticmethod
    def __reshape_result(fI: np.ndarray, fH:np.ndarray, V: np.ndarray,
                         axis: Union[int, Tuple[int]]):
        """
        Reshape the result to match the input data
        """
        raise NotImplementedError('Subclasses should implement reshape_result')


    @staticmethod
    def check_shapes(fH: np.ndarray, P: np.ndarray,  
                        V: np.ndarray,  wl_v: np.ndarray,
                        axis: Union[int, Tuple[int]]):
        """
        Check the shapes of all the input data
        """
        # Reshape to arrays all the data
        V_shape = np.array(V.shape)
        P_shape = np.array(P.shape)
        fH_shape = np.array(fH.shape)

        assert np.all(fH_shape[np.array(axis)] == P_shape[:-2]), \
                "fH and P are not the same shape"
        assert wl_v.shape[0] == fH.shape[0], \
            'number of frequencies does not match in the first axis of fH'
        assert V_shape[-1] == 3, "propagate do not support this data format"
        assert P_shape[-1] == 3, "propagate do not support this data format"

        return (V_shape, P_shape, fH_shape)
        

    @staticmethod
    def axis_value(axis):
        """
        Return a value given axis. Iff None returns axis -1
        """
        if axis is None:
            return (-1,)
        else:
            return axis


class RSD_propagator(Propagator):
    """
    Propagation using Rayleigh-Sommerfeld Diffraction
    """

    def __init__(self):
        super().__init__()


    def propagate(self, fH: np.ndarray, P: np.ndarray,  
                    V: np.ndarray,  wl_v: np.ndarray,
                    axis: Union[int, Tuple[int]] = None) -> np.ndarray:
        """
        Overrides propagate. It uses a RSD propagation. It collapses the 
        given axis, and add new ones of size V
        """
        print(V.shape)
        # Extract the axis and check the shapes
        l_axis = RSD_propagator.axis_value(axis)
        RSD_propagator.check_shapes(fH, P, V, wl_v, l_axis)
        # Reshape the data to the needed form
        V_a, P_a, fH_a = RSD_propagator.__reshape_data(fH, P, V, l_axis)
        # calculate distances from P to all V
        dist = np.linalg.norm(P_a - V_a, axis = -1)
        # RSD kernel calculation
        RSD_k = RSD_kernel.RSD_kernel_w(dist, wl_v)
        # Propagate the light, and focus on the voxel
        I_a = np.sum(fH_a[..., np.newaxis]*RSD_k, axis = -2)
        I = RSD_propagator.__reshape_result(I_a, fH_a, V, l_axis)

        return I


    @staticmethod
    def __reshape_data(fH: np.ndarray, P: np.ndarray, V: np.ndarray, 
                    l_axis: Union[int, Tuple[int]]):
        # Reshape all the data to an array of 3d coordinates
        V_a = V.reshape(1, -1, 3)
        P_a = P.reshape(-1, 1, 3)
        fH_new_shape = np.append(np.delete(fH.shape, l_axis), [P_a.shape[0]])
        # Reshape fH to match the P 
        fH_a = fH
        for axe in l_axis:
            fH_a = np.moveaxis(fH_a, axe, -1)
        fH_a = fH_a.reshape(fH_new_shape)
        fH_a = fH_a.swapaxes(0,-2)
        return V_a, P_a, fH_a


    @staticmethod
    def __reshape_result(I_a: np.ndarray, fH: np.ndarray, V: np.ndarray, 
                        axis: Union[int, Tuple[int]]):
        # Reshape to the corresponding shape. By axis: frequency, fH
        # uncorresponding to P shape, V shape
        I_shape = np.append(np.array(fH.shape)[:-1], np.array(V.shape)[:-1])
        I = I_a.reshape(I_shape)
        return I.swapaxes(0,-2)


class RSD_parallel_propagator(Propagator):
    """
    Propagation using Rayleigh-Sommerfeld Diffraction for parallel planes
    """

    def __init__(self, P: np.ndarray, V: np.ndarray, w_a: np.ndarray):
        super().__init__()
        # Creates the kernel. It check parallel planes
        self.K_rsd = RSD_kernel(V, P, w_a)
        assert P.ndim == 3, "P has to contain the 3d coordinates in a plane"
        assert V.ndim == 3 or V.ndim == 4, \
            "V has to contain the 3d coordinates as volume or plane"
        self.__P = P
        self.__V = V
        if self.__V.ndim == 4:
            self.__V_z_idx = {}
            for hdz, V_z in enumerate(self.__V):
                self.__V_z_idx[str(V_z[0,0])] = hdz
        self.__w_a = w_a
        self.__fH = None

    # Reorder the results to match the given data
    def __reorder_result(self, fI, axis):
        fI_rs = np.moveaxis(fI, -3, 0)
        fI_rs = np.moveaxis(fI_rs, [-2, -1], axis)


    def set_fH(self, fH, axis: Union[int, Tuple[int]] = None):
        """
        Set the new impulse response value to performance the propagation
        """
        if axis is None:
            l_axis = (-1, -2)
        else:
            l_axis = axis
        self.__fH = fH
        ffH = np.fft.fft2(fH, s=self.K_rsd.kernel_shape(), axes = l_axis)
        # Reorder the data to do easier the operations
        ffH = np.moveaxis(ffH, axis, [-2, -1])
        self.ffH = np.moveaxis(ffH, 0, -3)
    

    def propagate(self, fH: np.ndarray, P: np.ndarray,  
                V: np.ndarray,  wl_v: np.ndarray,
                axis: Union[int, Tuple[int]] = None) -> np.ndarray:
        """
        Override of propagate in Propagate. Optimized for parallel planes
        """
        # Check the data and types
        assert P is self.__P, "P is not the same used for initialization"
        assert wl_v is self.__w_a, \
            "wl_v is not the same used for initialization"
        if V.ndim == 3 and self.__V.ndim == 4:
            assert V.base is self.__V, \
                "V is not the part of the one used for initialization"
            idx_plane = [self.__V_z_idx[str(V[0,0])]]
        elif V.ndim == 3 and self.__V.ndim == 3:
            assert V is self.__V, \
                "V is not the same used for initialization"
            idx_plane = [0]
        elif V.ndim == 4:
            assert V is self.__V, \
                "V is not the same used for initialization"
            idx_plane = range(V.shape[0])
        else:
            raise "propagate does not support this data"
        
        # Check the axis 
        if axis is None:
            l_axis = [-2,-1]
        else:
            l_axis = axis


        if self.__fH is None or self.__fH is not fH:
            self.set_fH(fH, axis)

        # Space to store results
        if V.ndim == 3:
            fI_shape = (1, len(wl_v), V.shape[0], V.shape[1])
        else:   # V.ndim == 4
            fI_shape = (V.shape[0], len(wl_v), V.shape[1], V.shape[2])
        fI = np.zeros(fI_shape, dtype = np.complex128)

        # Propagate with the RSD kernel
        for i in idx_plane:
            print(self.K_rsd.f_propagate_i(self.ffH, i).shape)
            fI[i] = self.K_rsd.f_propagate_i(self.ffH, i)
        
        return self.__reorder_result(fI, l_axis)



