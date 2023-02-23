#cython: boundscheck=False
#cython: language_level=3
from libc.stdlib cimport malloc, calloc, free
import numpy as np
cimport numpy as np
np.import_array()

ctypedef fused T:
    double
    complex

cpdef np.ndarray chebval(np.ndarray x, np.ndarray c):
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    x = x.astype(float)
    y = np.zeros_like(x, dtype=c.dtype)
    if c.dtype.char in 'fdg':
        _chebval[double](<double *>np.PyArray_DATA(x),
                 <double *>np.PyArray_DATA(c),
                 <double *>np.PyArray_DATA(y),
                 c.shape[0],
                 x.shape[0])
    else:
        _chebval[complex](<double *>np.PyArray_DATA(x),
                 <complex *>np.PyArray_DATA(c),
                 <complex *>np.PyArray_DATA(y),
                 c.shape[0],
                 x.shape[0])

    return y

cdef void _chebval(double* x, T* c, T* y, int N, int M):
    cdef:
        int i, j
        T* c0 = <T*>calloc(M, sizeof(T))
        T* c1 = <T*>calloc(M, sizeof(T))
        T* tmp = <T*>calloc(M, sizeof(T))
        double* x2 = <double*>calloc(M, sizeof(double))

    if N == 1:
        for j in range(M):
            c0[j] = c[0]
            c1[j] = 0
    elif N == 2:
        for j in range(M):
            c0[j] = c[0]
            c1[j] = c[1]
    else:
        for j in range(M):
            x2[j] = 2*x[j]
            c0[j] = c[N-2]
            c1[j] = c[N-1]
        for i in range(3, N + 1):
            for j in range(M):
                tmp[j] = c0[j]
                c0[j] = c[N-i] - c1[j]
                c1[j] = tmp[j] + c1[j]*x2[j]
    for j in range(M):
        y[j] = c0[j] + x[j]*c1[j]

    free(c0)
    free(c1)
    free(tmp)
    free(x2)