"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   propagator.py
Description :   Interface of propagation to use with Phasor Fields solver, and
                RSD implementations for Parallel and non Parallel planes
"""

import numpy as np
from threading import Lock
from tal.reconstruct.pf.rsd_kernel import RSD_kernel
from typing import Tuple, Union


class Propagator(object):
    """
    Propagation interface
    """

    def propagate(self, fH: np.ndarray, P: np.ndarray,
                  V: np.ndarray,  wl_v: np.ndarray,
                  P_axis: Union[int, Tuple[int]] = None,
                  V_axis: Union[int, Tuple[int]] = None) -> np.ndarray:
        """
        Propagate from P to V the value fH, where fH is the impulse response at
        P in the Fourier domain, and wl_v is the array of wavelengths for each
        fH component
        @param self     : Propagator subclass implementation
        @param fH       : Impulse response in the Fourier domain
        @param P        : Coordinates of the fH samples
        @param V        : Coordinates to propagate fH
        @param wl_v     : Array of wavelengths for each fH Fourier component
        @param P_axis   : Axis of fH which represents P. If None, axis is 
                          assumed to be the last one
        @param V_axis   : Axis of fH which represents V. If None, it is assumed
                          there is no axis representing V. It use the axis to
                          partially propagations (e.g. propagation from sensors
                          but not from light sources)
        @return         : The propagation of fH to the V coordinates
        """
        raise NotImplementedError('Subclasses should implement propagate')

    @staticmethod
    def reshape_data(fH: np.ndarray, P: np.ndarray, V: np.ndarray,
                     axis: Union[int, Tuple[int]]):
        """
        Reshape the data to the needed form for the propagation implementations
        """
        raise NotImplementedError('Subclasses should implement reshape_data')

    # Reorder the results to match the given data

    @staticmethod
    def reshape_result(fI: np.ndarray, fH: np.ndarray, V: np.ndarray,
                       w_axis: int,
                       P_axis: Union[int, Tuple[int]],
                       V_axis: Union[int, Tuple[int]]):

        # Reshape to the corresponding shape. By axis: frequency, fH
        # uncorresponding to P shape, V shape
        V_l_axis = V_axis
        if V_l_axis is None:
            I_shape = tuple(np.delete(np.array(fH.shape), P_axis)) \
                + tuple(np.array(V.shape)[:-1])
            V_l_axis = tuple(np.arange(- V.ndim + 1, 0))
        else:
            I_shape = tuple(np.delete(np.array(fH.shape), P_axis + V_axis)) \
                + tuple(np.array(V.shape)[:-1])
        # Move the axis of frequencies to the beginning
        fI_aux = np.moveaxis(fI, w_axis, 0)
        # Reshape the data
        I = fI_aux.reshape(I_shape)
        return I

    @staticmethod
    def check_shapes(fH: np.ndarray, P: np.ndarray,
                     V: np.ndarray,  wl_v: np.ndarray,
                     P_axis: Union[int, Tuple[int]],
                     V_axis: Union[int, Tuple[int]]):
        """
        Check the shapes of all the input data
        """
        # Reshape to arrays all the data
        V_shape = np.array(V.shape)
        P_shape = np.array(P.shape)
        fH_shape = np.array(fH.shape)

        assert np.all(fH_shape[np.array(P_axis)] == P_shape[:-1]), \
            "fH and P are not the same shape"
        if V_axis is not None:
            assert np.all(fH_shape[np.array(V_axis)] == V_shape[:-1]), \
                "Indicated axis for V in H, but shapes are not the same"
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
                  P_axis: Union[int, Tuple[int]] = None,
                  V_axis: Union[int, Tuple[int]] = None) -> np.ndarray:
        """
        Overrides propagate. It uses a RSD propagation. It collapses the 
        given axis, and add new ones of size V
        """
        # Extract the axis and check the shapes
        P_l_axis = RSD_propagator.axis_value(P_axis)
        RSD_propagator.check_shapes(fH, P, V, wl_v, P_l_axis, V_axis)
        # Reshape the data to the needed form
        V_a, P_a, fH_a = RSD_propagator.reshape_data(
            fH, P, V, P_l_axis, V_axis)
        # calculate distances from P to all V
        dist = np.linalg.norm(P_a - V_a, axis=-1)
        # RSD kernel calculation
        RSD_k = RSD_kernel.RSD_kernel_w(dist, wl_v)
        # Propagate the light, and focus on the voxel
        I_a = np.sum(fH_a*RSD_k, axis=-2)
        I = RSD_propagator.reshape_result(I_a, fH, V, -2, P_l_axis, V_axis)

        return I

    @staticmethod
    def reshape_data(fH: np.ndarray, P: np.ndarray, V: np.ndarray,
                     P_axis: Union[int, Tuple[int]],
                     V_axis: Union[int, Tuple[int]]):
        # Reshape all the data to an array of 3d coordinates
        V_a = V.reshape(1, -1, 3)
        P_a = P.reshape(-1, 1, 3)
        if V_axis is not None:
            fH_new_shape = np.append(np.delete(fH.shape, P_axis + V_axis),
                                     [P_a.shape[0], V_a.shape[1]])
        else:
            fH_new_shape = np.append(np.delete(fH.shape, P_axis),
                                     [P_a.shape[0], 1])
        # Reshape fH to match the P
        fH_a = fH
        fH_a = np.moveaxis(fH_a, P_axis, np.arange(-len(P_axis), 0))

        # Reshape fH to match the V
        if V_axis is not None:
            fH_a = np.moveaxis(fH_a, V_axis, np.arange(-len(V_axis), 0))

        fH_a = fH_a.reshape(fH_new_shape)
        fH_a = np.moveaxis(fH_a, 0, -3)    # move the axis of frequencies
        return V_a, P_a, fH_a


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
                self.__V_z_idx[str(V_z[0, 0])] = hdz
        self.__w_a = w_a
        self.__fH = None
        self.lock = Lock()

    def set_fH(self, fH, P_axis, V_axis):
        """
        Set the new impulse response value to performance the propagation
        """
        self.__fH = fH
        ffH = np.fft.fft2(fH, s=self.K_rsd.kernel_shape(), axes=P_axis)
        # Reorder the data to do easier the operations
        ffH = np.moveaxis(ffH, P_axis, [-2, -1])
        # Reorder the frecuencies
        self.ffH = np.moveaxis(ffH, 0, -3)
        if V_axis is not None:
            ffH = np.moveaxis(ffH, V_axis, np.arange(-len(V_axis), 0))

    def propagate(self, fH: np.ndarray, P: np.ndarray,
                  V: np.ndarray,  wl_v: np.ndarray,
                  P_axis: Union[int, Tuple[int]] = None,
                  V_axis: Union[int, Tuple[int]] = None) -> np.ndarray:
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
            idx_plane = [self.__V_z_idx[str(V[0, 0])]]
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
        if P_axis is None:
            P_l_axis = [-2, -1]
        else:
            P_l_axis = P_axis

        with self.lock:
            if self.__fH is None or self.__fH is not fH:
                self.set_fH(fH, P_l_axis, V_axis)

        # Space to store results
        fI_pre_shape = tuple(np.array(self.ffH.shape)[:-3])
        if V.ndim == 3:
            fI_shape = fI_pre_shape + (len(wl_v), 1, V.shape[0], V.shape[1])
        else:   # V.ndim == 4
            fI_shape = fI_pre_shape + (len(wl_v), V.shape[0],
                                       V.shape[1], V.shape[2])
        fI = np.zeros(fI_shape, dtype=np.complex128)

        # Propagate with the RSD kernel
        for idx_fI, plane_id in enumerate(idx_plane):
            K = self.K_rsd.get_f_RSD_kernel_i(plane_id)
            fI_padded = np.fft.ifft2(K * self.ffH)
            fI[..., idx_fI, :, :] = fI_padded[...,
                                              -fI_shape[-2]:, -fI_shape[-1]:]

        # Squeeze unnedded axes
        if V.ndim == 3:
            fI = np.squeeze(fI, axis=-3)

        return RSD_parallel_propagator.reshape_result(fI, fH, V, -V.ndim,
                                                      P_l_axis, V_axis)
