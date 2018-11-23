cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

ctypedef np.complex128_t complex_t
ctypedef np.int64_t int_t

def convolve_1D(np.ndarray[complex_t, ndim=1] u, np.ndarray[complex_t, ndim=1] v,
                np.ndarray[complex_t, ndim=1] uv, np.ndarray[int_t, ndim=1] k):
    cdef int m, n, p, i, j, N
    cdef complex_t um, vn

    N = u.shape[0]
    for m in k:
        for n in k:
            p = m + n
            if N % 2 == 0:
                if abs(m) == N//2:
                    um = u[m]*0.5
                else:
                    um = u[m]
                if abs(n) == N//2:
                    vn = v[n]*0.5
                else:
                    vn = v[n]
            else:
                um = u[m]
                vn = v[n]
            uv[p] = uv[p] + um*vn



def convolve_real_1D(np.ndarray[complex_t, ndim=1] u, np.ndarray[complex_t, ndim=1] v,
                     np.ndarray[complex_t, ndim=1] uv, np.ndarray[int_t, ndim=1] k):
    cdef int m, n, p, i, j, N
    cdef complex_t um, vn

    N = uv.shape[0]-1
    for m in k:
        for n in k:
            p = m + n
            if p >= 0:
                if N % 2 == 0:
                    if abs(m) == N//2:
                        um = u[abs(m)]*0.5
                    elif m >= 0:
                        um = u[m]
                    else:
                        um = u[abs(m)].real - 1j*u[abs(m)].imag
                    if abs(n) == N//2:
                        vn = v[abs(n)]*0.5
                    elif n >= 0:
                        vn = v[n]
                    else:
                        vn = v[abs(n)].real - 1j*v[abs(n)].imag
                else:
                    if m >= 0:
                        um = u[m]
                    elif m < 0:
                        um = u[abs(m)].real - 1j*u[abs(m)].imag
                    if n >= 0:
                        vn = v[n]
                    elif n < 0:
                        vn = v[abs(n)].real - 1j*v[abs(n)].imag
                uv[p] = uv[p] + um*vn
