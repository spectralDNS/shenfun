#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

import numpy as np
cimport cython
cimport numpy as np

ctypedef fused T:
    np.float64_t
    np.complex128_t

def cross2D(T[:, ::1] c, T[:, :, ::1] a, T[:, :, ::1] b):
    cdef:
        int i, j
        T a0, a1, b0, b1
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            a0 = a[0, i, j]
            a1 = a[1, i, j]
            b0 = b[0, i, j]
            b1 = b[1, i, j]
            c[i, j] = a0*b1 - a1*b0

def cross3D(T[:, :, :, ::1] c, T[:, :, :, ::1] a, T[:, :, :, ::1] b):
    cdef:
        int i, j, k
        T a0, a1, a2, b0, b1, b2
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0, i, j, k]
                a1 = a[1, i, j, k]
                a2 = a[2, i, j, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = a1*b2 - a2*b1
                c[1, i, j, k] = a2*b0 - a0*b2
                c[2, i, j, k] = a0*b1 - a1*b0
