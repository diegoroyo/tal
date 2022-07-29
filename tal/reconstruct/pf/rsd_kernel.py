"""
Author      :   Pablo Luesia Lahoz
E-Mail      :   pluesia at unizar.es
File        :   rsd_kernel.py
Description :   Class for a RSD kernel generator, and propagator, in order
                to perform fast propagation from plane to parallel plane
"""

import numpy as np


class RSD_kernel(object):
    """
    Class implementation for a RSD kernel. It creates a kernel for performance
    fast propagation between paralel planes. 
    """

    def __init__(self, V: np.ndarray, P: np.ndarray, lambda_v: np.ndarray):
        """
        Initialize the class of the kernel. The plane P and the planes in V has
        to be parallel.
        @param V            : Volume in 3d to propagate, formed by parallel
                              planes. The first axis are the planes, the second
                              are the rows, the third the columns, and the 
                              forth the coordinates of the points in the volume.
        @param P            : Points of the plane origin to propagate. The
                              first axis are the rows, the second the columns,
                              and the third the coordinates of the points in 
                              the plane.
        @param lambda_v     : Arrays of wavelengths to performance the RSD
                              propagation in metters.
        :return             : An instance of RSD_kernel class.
        """
        # Check the planes shapes are the same
        assert P.ndim == 3 and V.ndim == 4, \
            "P has to be a plane, and V a volume. Both with 3d coordinates"
        assert P.shape == V[0].shape, "V and P shapes are not the same"
        # Check the planes are parallel
        # First V plane normal
        normal_V, d_V = RSD_kernel.__as_plane_eq(V[0, 0, 0],
                                                 V[0, -1, 0],
                                                 V[0, -1, -1])
        normal_P, d_P = RSD_kernel.__as_plane_eq(P[0, 0], P[-1, 0], P[-1, -1])
        # Check if planes are parallel (with an small error to 0)
        assert np.linalg.norm(np.cross(normal_V, normal_P)) == 0, \
            "V and P are not parallel"

        # Volume to propagate
        self._V = V
        self._n_V = normal_V
        self._d_V = d_V
        # Origin plane to propagate
        self._P = P
        self._n_P = normal_P
        self._d_P = d_P
        # Wavelengths to propagate
        self._wl_v = lambda_v
        # Preprocess values. It is None until data is preprocessed
        self._preprocess = (self.__K_at_P(), self.__vecs_P_to_V_planes())

    ###########################################################################
    #                             PRIVATE METHODS                             #
    ###########################################################################

    # Given 3 points it return the plane equation

    def __as_plane_eq(p1, p2, p3):
        # Vectors in plane
        v1 = p3 - p1
        v2 = p2 - p1
        # Normal to the plane
        n = np.cross(v1, v2)
        d = np.dot(n, p3)
        return n, d

    # Generate a coordinates matrix kernel for the plane P (at distance 0)

    def __K_at_P(self):
        # Precalculate kernel vectors
        (nx, ny, _) = self._P.shape
        # Kernel shape
        rx = 2*nx - 1
        ry = 2*ny - 1
        # Delta of the points in i and j axis of the rectangle
        di = self._P[0, 1] - self._P[0, 0]
        dj = self._P[1, 0] - self._P[0, 0]
        # Generate the central row, and from it the matrix
        K_row = np.linspace(start=-di * (nx - 1),
                            stop=di * (nx - 1),
                            num=rx, dtype=np.float64)
        K_base = np.linspace(start=K_row - dj * (ny - 1),
                             stop=K_row + dj * (ny - 1),
                             num=ry, dtype=np.float64)
        return K_base

    # Return an array of vectors from P to the different V planes

    def __vecs_P_to_V_planes(self):
        n_V_m = np.linalg.norm(self._n_V)   # Modulus of V first plane normal
        # Distance from P to V first plane
        d_P2V = np.abs(np.sum(self._n_V * self._P[0, 0]) + self._d_V) / n_V_m
        # Unitary normal vector of V planes
        n_V_i = self._n_V / n_V_m
        # number of V planes
        rz = self._V.shape[0]
        # Distance between V planes
        d_V2V = np.linalg.norm(self._V[0, 0, 0] - self._V[-1, 0, 0]) / (rz - 1)
        # Shortest vector from first V plane to the V planes
        n_V2V = np.linspace(start=n_V_i*0,            # Vector to itself
                            stop=n_V_i * d_V2V * rz,  # Vector furthest plane
                            num=rz, dtype=np.float64)
        # Add the vector from P to V
        return n_V2V + n_V_i*d_P2V

    # Return a spatial kernel with the vecotrs from P to the i-th plane in V.
    def __spatial_i_kernel(self, i):
        K_base, v_P2V = self._preprocess
        return K_base + v_P2V[i]

    ###########################################################################
    #                             PUBLIC METHODS                              #
    ###########################################################################
    def kernel_shape(self):
        """
        Returns the shape of the Kernel
        """
        (K_base, _) = self._preprocess
        return (K_base.shape[0], K_base.shape[1])

    def get_f_RSD_kernel_i(self, i: int):
        """
        Returns the RSD kernel in the 2d fourier domain of the 
        @param i    : Index of the V plane
        :return     : The plane kernel for i-th plane of V in fourier domain
        """
        s_K = self.__spatial_i_kernel(i)
        d_K = np.linalg.norm(s_K, axis=-1)
        rsd_K = RSD_kernel.RSD_kernel_w(d_K, self._wl_v)
        return np.fft.fft2(rsd_K)

    @staticmethod
    def RSD_kernel_w(K_d: np.ndarray, lambda_w: float):
        """
        Given a rectangular distance spatial kernel K_d, it returns the 
        Rayleigh-Sommerfled propagator kernel for the frequency with wavelength
        lambda_w.
        @param K_d      : Rectangular distance spatial kernel
        @param lambda_w : Wavelength to performance the transformation
        @return         : Complex matrix representing the kernel
        """
        # Broadcastable lambda_w to match K_d
        lambda_w_br = lambda_w.reshape((-1,) + (1,) * K_d.ndim)
        return np.exp(2*np.pi*1j/lambda_w_br * K_d)/K_d
