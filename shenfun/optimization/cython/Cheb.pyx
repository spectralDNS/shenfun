import numpy as np
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real

ctypedef fused T:
    real_t
    complex_t

def chebval(x, c):
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    x = x.astype(float)
    y = np.zeros_like(x, dtype=c.dtype)
    c0 = np.zeros_like(x, dtype=c.dtype)
    c1 = np.zeros_like(x, dtype=c.dtype)
    tmp = np.zeros_like(x, dtype=c.dtype)
    x2 = np.zeros_like(x)
    _chebval(x, c, y, c0, c1, tmp, x2)
    return y

def _chebval(real_t[:] x, T[:] c, T[:] y, T[:] c0, T[:] c1,
             T[:] tmp, real_t[:] x2):
    cdef:
        int i, j
        int N = c.shape[0]
        int M = x.shape[0]

    if N == 1:
        c0[:] = c[0]
        c1[:] = 0
    elif N == 2:
        c0[:] = c[0]
        c1[:] = c[1]
    else:
        for j in range(M):
            x2[j] = 2*x[j]
            c0[j] = c[-2]
            c1[j] = c[-1]
        for i in range(3, N + 1):
            for j in range(M):
                tmp[j] = c0[j]
                c0[j] = c[-i] - c1[j]
                c1[j] = tmp[j] + c1[j]*x2[j]
    for j in range(M):
        y[j] = c0[j] + x[j]*c1[j]