import numpy as np
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t

ctypedef fused T:
    real_t
    complex_t

def evaluate_2D(np.ndarray[T, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                list P, int r2c, int M, int start):
    if P[0].dtype == np.complex and P[1].dtype == np.complex:
        b = _evaluate_2D_cc(b, u, P[0], P[1], r2c, M, start)

    elif P[0].dtype == np.float and P[1].dtype == np.complex:
        b = _evaluate_2D_rc(b, u, P[0], P[1], r2c, M, start)

    elif P[0].dtype == np.complex and P[1].dtype == np.float:
        b = _evaluate_2D_cr(b, u, P[0], P[1], r2c, M, start)

    return b


def _evaluate_2D_cc(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                    np.ndarray[complex_t, ndim=2] P0,
                    np.ndarray[complex_t, ndim=2] P1,
                    int r2c, int M, int start):
    cdef int k, l, i, kk, ll
    cdef complex_t p

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                if r2c < 0:
                    b[i] = b[i] + u[k, l] * P0[i, k] * P1[i, l]
                elif r2c == 0:
                    p = u[k, l] * P0[i, k]
                    b[i] = b[i] + p * P1[i, l]
                    kk = k + start
                    if kk > 0 & kk < M:
                        b[i] = b[i] + (p.real - 1j*p.imag)  * P1[i, l]

                elif r2c == 1:
                    p = u[k, l] * P1[i, l]
                    b[i] = b[i] + p * P0[i, k]
                    ll = l + start
                    if ll > 0 & ll < M:
                        b[i] = b[i] + (p.real - 1j*p.imag)  * P0[i, k]

    return b

def _evaluate_2D_rc(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                    np.ndarray[real_t, ndim=2] P0,
                    np.ndarray[complex_t, ndim=2] P1,
                    int r2c, int M, int start):
    cdef int k, l, i, kk, ll
    cdef complex_t p

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                if r2c < 0:
                    b[i] = b[i] + u[k, l] * P0[i, k] * P1[i, l]
                elif r2c == 0:
                    p = u[k, l] * P0[i, k]
                    b[i] = b[i] + p * P1[i, l]
                    kk = k + start
                    if kk > 0 & kk < M:
                        b[i] = b[i] + (p.real - 1j*p.imag)  * P1[i, l]

                elif r2c == 1:
                    p = u[k, l] * P1[i, l]
                    b[i] = b[i] + p * P0[i, k]
                    ll = l + start
                    if ll > 0 & ll < M:
                        b[i] = b[i] + (p.real - 1j*p.imag)  * P0[i, k]

    return b

def _evaluate_2D_cr(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                    np.ndarray[complex_t, ndim=2] P0,
                    np.ndarray[real_t, ndim=2] P1,
                    int r2c, int M, int start):
    cdef int k, l, i, kk, ll
    cdef complex_t p

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                if r2c < 0:
                    b[i] = b[i] + u[k, l] * P0[i, k] * P1[i, l]
                elif r2c == 0:
                    p = u[k, l] * P0[i, k]
                    b[i] = b[i] + p * P1[i, l]
                    kk = k + start
                    if kk > 0 & kk < M:
                        b[i] = b[i] + (p.real - 1j*p.imag)  * P1[i, l]

                elif r2c == 1:
                    p = u[k, l] * P1[i, l]
                    b[i] = b[i] + p * P0[i, k]
                    ll = l + start
                    if ll > 0 & ll < M:
                        b[i] = b[i] + (p.real - 1j*p.imag)  * P0[i, k]

    return b


def evaluate_3D(np.ndarray[T, ndim=1] b, np.ndarray[T, ndim=3] u,
                list P, int r2c, int M, int start):

    if P[0].dtype == np.complex and P[1].dtype == np.complex and P[2].dtype == np.complex:
        b = _evaluate_3D_ccc(b, u, P[0], P[1], P[2], r2c, M, start)
    elif P[0].dtype == np.float and P[1].dtype == np.complex and P[2].dtype == np.complex:
        b = _evaluate_3D_rcc(b, u, P[0], P[1], P[2], r2c, M, start)
    elif P[0].dtype == np.complex and P[1].dtype == np.float and P[2].dtype == np.complex:
        b = _evaluate_3D_crc(b, u, P[0], P[1], P[2], r2c, M, start)
    elif P[0].dtype == np.complex and P[1].dtype == np.complex and P[2].dtype == np.float:
        b = _evaluate_3D_ccr(b, u, P[0], P[1], P[2], r2c, M, start)

    return b

def _evaluate_3D_ccc(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=3] u,
                    np.ndarray[complex_t, ndim=2] P0,
                    np.ndarray[complex_t, ndim=2] P1,
                    np.ndarray[complex_t, ndim=2] P2,
                    int r2c, int M, int start):
    cdef int k, l, m, i, kk, ll, mm
    cdef complex_t p

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    if r2c < 0:
                        b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
                    elif r2c == 0:
                        p = u[k, l, m] * P0[i, k]
                        b[i] = b[i] + p * P1[i, l] * P2[i, m]
                        kk = k + start
                        if kk > 0 & kk < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P1[i, l] * P2[i, m]
                    elif r2c == 1:
                        p = u[k, l, m] * P1[i, l]
                        b[i] = b[i] + p * P0[i, k] * P2[i, m]
                        ll = l + start
                        if ll > 0 & ll < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P0[i, k] * P2[i, m]
                    elif r2c == 2:
                        p = u[k, l, m] * P2[i, m]
                        b[i] = b[i] + p * P0[i, k] * P1[i, l]
                        mm = m + start
                        if mm > 0 & mm < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P0[i, k] * P1[i, l]

    return b

def _evaluate_3D_rcc(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=3] u,
                    np.ndarray[real_t, ndim=2] P0,
                    np.ndarray[complex_t, ndim=2] P1,
                    np.ndarray[complex_t, ndim=2] P2,
                    int r2c, int M, int start):
    cdef int k, l, m, i, ll, mm
    cdef complex_t p

    assert r2c == 1 or r2c == 2 or r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    if r2c < 0:
                        b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
                    elif r2c == 1:
                        p = u[k, l, m] * P1[i, l]
                        b[i] = b[i] + p * P0[i, k] * P2[i, m]
                        ll = l + start
                        if ll > 0 & ll < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P0[i, k] * P2[i, m]
                    elif r2c == 2:
                        p = u[k, l, m] * P2[i, m]
                        b[i] = b[i] + p * P0[i, k] * P1[i, l]
                        mm = m + start
                        if mm > 0 & mm < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P0[i, k] * P1[i, l]
    return b

def _evaluate_3D_crc(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=3] u,
                    np.ndarray[complex_t, ndim=2] P0,
                    np.ndarray[real_t, ndim=2] P1,
                    np.ndarray[complex_t, ndim=2] P2,
                    int r2c, int M, int start):
    cdef int k, l, m, i, kk, ll, mm
    cdef complex_t p

    assert r2c == 0 or r2c == 2 or r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    if r2c < 0:
                        b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
                    elif r2c == 0:
                        p = u[k, l, m] * P0[i, k]
                        b[i] = b[i] + p * P1[i, l] * P2[i, m]
                        kk = k + start
                        if kk > 0 & kk < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P1[i, l] * P2[i, m]
                    elif r2c == 2:
                        p = u[k, l, m] * P2[i, m]
                        b[i] = b[i] + p * P0[i, k] * P1[i, l]
                        mm = m + start
                        if mm > 0 & mm < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P0[i, k] * P1[i, l]
    return b

def _evaluate_3D_ccr(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=3] u,
                    np.ndarray[complex_t, ndim=2] P0,
                    np.ndarray[complex_t, ndim=2] P1,
                    np.ndarray[real_t, ndim=2] P2,
                    int r2c, int M, int start):
    cdef int k, l, m, i, kk, ll
    cdef complex_t p

    assert r2c == 0 or r2c == 1 or r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    if r2c < 0:
                        b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
                    elif r2c == 0:
                        p = u[k, l, m] * P0[i, k]
                        b[i] = b[i] + p * P1[i, l] * P2[i, m]
                        kk = k + start
                        if kk > 0 & kk < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P1[i, l] * P2[i, m]
                    elif r2c == 1:
                        p = u[k, l, m] * P1[i, l]
                        b[i] = b[i] + p * P0[i, k] * P2[i, m]
                        ll = l + start
                        if ll > 0 & ll < M:
                            b[i] = b[i] + (p.real - 1j*p.imag) * P0[i, k] * P2[i, m]
    return b
