#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

import numpy as np
cimport cython
cimport numpy as np

ctypedef fused T:
    np.float64_t
    np.complex128_t

def outer2D(T[:, :, ::1] a, T[:, :, ::1] b, T[:, :, ::1] c, int symmetric):
    cdef int i, j
    if symmetric == 1:
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                c[0, i, j] = a[0, i, j]**2           # (0, 0)
                c[1, i, j] = a[0, i, j]*a[1, i, j]   # (0, 1)
                c[2, i, j] = c[1, i, j]              # (1, 0)
                c[3, i, j] = a[1, i, j]**2           # (1, 1)
    else:
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                c[0, i, j] = a[0, i, j]*b[0, i, j]   # (0, 0)
                c[1, i, j] = a[0, i, j]*b[1, i, j]   # (0, 1)
                c[2, i, j] = a[1, i, j]*b[0, i, j]   # (1, 0)
                c[3, i, j] = a[1, i, j]*b[1, i, j]   # (1, 1)


def outer3D(T[:, :, :, ::1] a, T[:, :, :, ::1] b, T[:, :, :, ::1] c, int symmetric):
    cdef int i, j, k
    if symmetric == 1:
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                for k in range(a.shape[3]):
                    c[0, i, j, k] = a[0, i, j, k]**2              # (0, 0)
                    c[1, i, j, k] = a[0, i, j, k]*a[1, i, j, k]   # (0, 1)
                    c[2, i, j, k] = a[0, i, j, k]*a[2, i, j, k]   # (0, 2)
                    c[3, i, j, k] = c[1, i, j, k]                 # (1, 0)
                    c[4, i, j, k] = a[1, i, j, k]**2              # (1, 1)
                    c[5, i, j, k] = a[1, i, j, k]*a[2, i, j, k]   # (1, 2)
                    c[6, i, j, k] = c[2, i, j, k]                 # (2, 0)
                    c[7, i, j, k] = c[5, i, j, k]                 # (2, 1)
                    c[8, i, j, k] = a[2, i, j, k]**2              # (2, 2)
    else:
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                for k in range(a.shape[3]):
                    c[0, i, j, k] = a[0, i, j, k]*b[0, i, j, k]   # (0, 0)
                    c[1, i, j, k] = a[0, i, j, k]*b[1, i, j, k]   # (0, 1)
                    c[2, i, j, k] = a[0, i, j, k]*b[2, i, j, k]   # (0, 2)
                    c[3, i, j, k] = a[1, i, j, k]*b[0, i, j, k]   # (1, 0)
                    c[4, i, j, k] = a[1, i, j, k]*b[1, i, j, k]   # (1, 1)
                    c[5, i, j, k] = a[1, i, j, k]*b[2, i, j, k]   # (1, 2)
                    c[6, i, j, k] = a[2, i, j, k]*b[0, i, j, k]   # (2, 0)
                    c[7, i, j, k] = a[2, i, j, k]*b[1, i, j, k]   # (2, 1)
                    c[8, i, j, k] = a[2, i, j, k]*b[2, i, j, k]   # (2, 2)
