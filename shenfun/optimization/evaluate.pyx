#!python
#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
import cython

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t

ctypedef fused T:
    real_t
    complex_t

cdef extern from "complex.h" nogil:
    double complex exp(double complex)

cdef extern from "math.h" nogil:
    double sin(double)
    double cos(double)

def evaluate_2D(np.ndarray[T, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                list P, int r2c, int M, int start):
    if P[0].dtype == np.complex and P[1].dtype == np.complex:
        # 2 Fourier spaces
        if r2c < 0: # No R2C
            b = _evaluate_2D_cc0(b, u, P[0], P[1], r2c, M, start)
        else:
            b = _evaluate_2D_cc1(b, u, P[0], P[1], r2c, M, start)

    elif P[0].dtype == np.float and P[1].dtype == np.complex:
        # One non-fourier in axis=0
        if r2c < 0: # No R2C
            b = _evaluate_2D_rc0(b, u, P[0], P[1], r2c, M, start)
        else:
            b = _evaluate_2D_rc1(b, u, P[0], P[1], r2c, M, start)

    elif P[0].dtype == np.complex and P[1].dtype == np.float:
        # One non-fourier in axis=1
        if r2c < 0: # No R2C
            b = _evaluate_2D_cr0(b, u, P[0], P[1], r2c, M, start)
        else:
            b = _evaluate_2D_cr1(b, u, P[0], P[1], r2c, M, start)

    return b

def _evaluate_2D_cc0(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                     np.ndarray[complex_t, ndim=2] P0,
                     np.ndarray[complex_t, ndim=2] P1,
                     int r2c, int M, int start):
    cdef int k, l, i, kk, ll
    cdef complex_t p

    assert r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                b[i] = b[i] + u[k, l] * P0[i, k] * P1[i, l]
    return b

def _evaluate_2D_cc1(np.ndarray[real_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                     np.ndarray[complex_t, ndim=2] P0,
                     np.ndarray[complex_t, ndim=2] P1,
                     int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p
    assert r2c == 0 or r2c == 1
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                p = (u[k, l] * P0[i, k] * P1[i, l]).real
                b[i] += p
                if r2c == 0:
                    ii = k + start
                else:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p

    return b

def _evaluate_2D_rc0(np.ndarray[complex_t, ndim=1] b,
                     np.ndarray[complex_t, ndim=2] u,
                     np.ndarray[real_t, ndim=2] P0,
                     np.ndarray[complex_t, ndim=2] P1,
                     int r2c, int M, int start):
    cdef int k, l, i

    assert r2c < 0
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                b[i] = b[i] + u[k, l] * P0[i, k] * P1[i, l]
    return b


def _evaluate_2D_rc1(np.ndarray[real_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                     np.ndarray[real_t, ndim=2] P0,
                     np.ndarray[complex_t, ndim=2] P1,
                     int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p
    assert r2c == 0 or r2c == 1

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                p = (u[k, l] * P0[i, k] * P1[i, l]).real
                b[i] += p
                if r2c == 0:
                    ii = k + start
                else:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p
    return b

def _evaluate_2D_cr0(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                     np.ndarray[complex_t, ndim=2] P0,
                     np.ndarray[real_t, ndim=2] P1,
                     int r2c, int M, int start):
    cdef int k, l, i

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                b[i] += u[k, l] * P0[i, k] * P1[i, l]

    return b

def _evaluate_2D_cr1(np.ndarray[real_t, ndim=1] b, np.ndarray[complex_t, ndim=2] u,
                     np.ndarray[complex_t, ndim=2] P0,
                     np.ndarray[real_t, ndim=2] P1,
                     int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p
    assert r2c == 0 or r2c == 1

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                p = (u[k, l] * P0[i, k] * P1[i, l]).real
                b[i] += p
                if r2c == 0:
                    ii = k + start
                else:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p

    return b

def evaluate_3D(np.ndarray[T, ndim=1] b, np.ndarray[complex_t, ndim=3] u,
                list P, int r2c, int M, int start):

    if P[0].dtype == np.complex and P[1].dtype == np.complex and P[2].dtype == np.complex:
        # Only Fourier bases
        if r2c < 0: # No R2C
            b = _evaluate_3D_ccc0(b, u, P[0], P[1], P[2], r2c, M, start)
        else:
            b = _evaluate_3D_ccc1(b, u, P[0], P[1], P[2], r2c, M, start)

    elif P[0].dtype == np.float and P[1].dtype == np.complex and P[2].dtype == np.complex:
        # One non-fourier in axis=0
        if r2c < 0: # No R2C
            b = _evaluate_3D_rcc0(b, u, P[0], P[1], P[2], r2c, M, start)
        else:
            b = _evaluate_3D_rcc1(b, u, P[0], P[1], P[2], r2c, M, start)

    elif P[0].dtype == np.complex and P[1].dtype == np.float and P[2].dtype == np.complex:
        # One non-fourier in axis=1
        if r2c < 0: # No R2C
            b = _evaluate_3D_crc0(b, u, P[0], P[1], P[2], r2c, M, start)
        else:
            b = _evaluate_3D_crc1(b, u, P[0], P[1], P[2], r2c, M, start)

    elif P[0].dtype == np.complex and P[1].dtype == np.complex and P[2].dtype == np.float:
        # One non-fourier in axis=2
        if r2c < 0: # No R2C
            b = _evaluate_3D_ccr0(b, u, P[0], P[1], P[2], r2c, M, start)
        else:
            b = _evaluate_3D_ccr1(b, u, P[0], P[1], P[2], r2c, M, start)


    return b

def _evaluate_3D_ccc0(np.ndarray[complex_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
    return b

def _evaluate_3D_ccc1(np.ndarray[real_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    elif r2c == 1:
                        ii = l + start
                    elif r2c == 2:
                        ii = m + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def _evaluate_3D_rcc0(np.ndarray[complex_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[real_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i

    assert r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    b[i] += u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
    return b

def _evaluate_3D_rcc1(np.ndarray[real_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[real_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p

    assert r2c == 1 or r2c == 2

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    b[i] += p
                    if r2c == 1:
                        ii = l + start
                    else:
                        ii = m + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def _evaluate_3D_crc0(np.ndarray[complex_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[real_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i

    assert r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
    return b

def _evaluate_3D_crc1(np.ndarray[real_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[real_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p

    assert r2c == 0 or r2c == 2

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    else:
                        ii = m + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def _evaluate_3D_ccr0(np.ndarray[complex_t, ndim=1] b, np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[real_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i

    assert r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
    return b

def _evaluate_3D_ccr1(np.ndarray[real_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[real_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p

    assert r2c == 0 or r2c == 1

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    else:
                        ii = l + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def evaluate_lm_2D(list bases, np.ndarray[T, ndim=1] b, np.ndarray[complex_t, ndim=2] u, np.ndarray[real_t, ndim=1] x0, np.ndarray[real_t, ndim=1] x1, np.ndarray[real_t, ndim=1] w0, np.ndarray[real_t, ndim=1] w1, int r2c, int M, int start):

    if np.all([base.family() == 'fourier' for base in bases]):
        # 2 Fourier spaces
        if r2c < 0: # No R2C
            b = _evaluate_lm_2D_cc0(b, u, x0, x1, w0, w1, r2c, M, start)
        else:
            b = _evaluate_lm_2D_cc1(b, u, x0, x1, w0, w1, r2c, M, start)

    elif bases[1].family() == 'fourier':
        # One non-fourier in axis=0
        if r2c < 0: # No R2C
            b = _evaluate_lm_2D_rc0(b, u, x0, x1, w0, w1, bases, r2c, M, start)
        else:
            b = _evaluate_lm_2D_rc1(b, u, x0, x1, w0, w1, bases, r2c, M, start)

    elif bases[0].family() == 'fourier':
        # One non-fourier in axis=1
        if r2c < 0: # No R2C
            b = _evaluate_lm_2D_cr0(b, u, x0, x1, w0, w1, bases, r2c, M, start)
        else:
            b = _evaluate_lm_2D_cr1(b, u, x0, x1, w0, w1, bases, r2c, M, start)

    return b

def _evaluate_lm_2D_cc0(np.ndarray[complex_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        int r2c, int M, int start):
    cdef int k, l, i
    assert r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                b[i] += u[k, l] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))
    return b

def _evaluate_lm_2D_cc1(np.ndarray[real_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                p = (u[k, l] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))).real
                b[i] += p
                if r2c == 0:
                    ii = k + start
                elif r2c == 1:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p
    return b

def _evaluate_lm_2D_rc0(np.ndarray[complex_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        list bases,
                        int r2c, int M, int start):
    cdef int k, l, i
    cdef np.ndarray P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c < 0
    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                b[i] += u[k, l] * P0[i] * exp(1j*w1[l]*x1[i])
    return b


def _evaluate_lm_2D_rc1(np.ndarray[real_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        list bases,
                        int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                p = (u[k, l] * P0[i] * exp(1j*w1[l]*x1[i])).real
                b[i] += p
                if r2c == 0:
                    ii = k + start
                elif r2c == 1:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p
    return b

def _evaluate_lm_2D_cr0(np.ndarray[complex_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        list bases,
                        int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            P0 = bases[1].evaluate_basis(x1, l, P0)
            for i in range(b.shape[0]):
                b[i] += u[k, l] * P0[i] * exp(1j*w0[k]*x0[i])

    return b

def _evaluate_lm_2D_cr1(np.ndarray[real_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        list bases,
                        int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            P0 = bases[1].evaluate_basis(x1, l, P0)
            for i in range(b.shape[0]):
                p = (u[k, l] * P0[i] * exp(1j*w0[k]*x0[i])).real
                b[i] += p
                if r2c == 0:
                    ii = k + start
                elif r2c == 1:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p
    return b

def evaluate_lm_3D(list bases, np.ndarray[T, ndim=1] b, np.ndarray[complex_t, ndim=3] u, np.ndarray[real_t, ndim=1] x0, np.ndarray[real_t, ndim=1] x1, np.ndarray[real_t, ndim=1] x2, np.ndarray[real_t, ndim=1] w0, np.ndarray[real_t, ndim=1] w1, np.ndarray[real_t, ndim=1] w2, int r2c, int M, int start):

    if np.all([base.family() == 'fourier' for base in bases]):
        # Only Fourier bases
        if r2c < 0: # No R2C
            b = _evaluate_lm_3D_ccc0(b, u, x0, x1, x2, w0, w1, w2, r2c, M, start)
        else:
            b = _evaluate_lm_3D_ccc1(b, u, x0, x1, x2, w0, w1, w2, r2c, M, start)

    elif not bases[0].family() == 'fourier':
        # One non-fourier in axis=0
        if r2c < 0: # No R2C
            b = _evaluate_lm_3D_rcc0(b, u, x0, x1, x2, w0, w1, w2, bases, r2c, M, start)
        else:
            b = _evaluate_lm_3D_rcc1(b, u, x0, x1, x2, w0, w1, w2, bases, r2c, M, start)

    elif not bases[1].family() == 'fourier':
        # One non-fourier in axis=1
        if r2c < 0: # No R2C
            b = _evaluate_lm_3D_crc0(b, u, x0, x1, x2, w0, w1, w2, bases, r2c, M, start)
        else:
            b = _evaluate_lm_3D_crc1(b, u, x0, x1, x2, w0, w1, w2, bases, r2c, M, start)

    elif not bases[2].family() == 'fourier':
        # One non-fourier in axis=2
        if r2c < 0: # No R2C
            b = _evaluate_lm_3D_ccr0(b, u, x0, x1, x2, w0, w1, w2, bases, r2c, M, start)
        else:
            b = _evaluate_lm_3D_ccr1(b, u, x0, x1, x2, w0, w1, w2, bases, r2c, M, start)

    return b

def _evaluate_lm_3D_ccc0(np.ndarray[complex_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         int r2c, int M, int start):
    cdef int k, l, m, i
    cdef double xx
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    xx = w0[k]*x0[i] + w1[l]*x1[i] + w2[m]*x2[i]
                    b[i] += u[k, l, m] * (cos(xx) + 1j*sin(xx))

    return b

def _evaluate_lm_3D_ccc1(np.ndarray[real_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i] + w2[m]*x2[i]))).real
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    elif r2c == 1:
                        ii = l + start
                    elif r2c == 2:
                        ii = m + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def _evaluate_lm_3D_rcc0(np.ndarray[complex_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         list bases,
                         int r2c, int M, int start):
    cdef int k, l, m, i, ll, mm
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 1 or r2c == 2 or r2c < 0

    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    b[i] += u[k, l, m] * P0[i] * exp(1j*(w1[l]*x1[i] + w2[m]*x2[i]))
    return b

def _evaluate_lm_3D_rcc1(np.ndarray[real_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         list bases,
                         int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 1 or r2c == 2

    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i] * exp(1j*(w1[l]*x1[i] + w2[m]*x2[i]))).real
                    b[i] += p
                    if r2c == 1:
                        ii = l + start
                    elif r2c == 2:
                        ii = m + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def _evaluate_lm_3D_crc0(np.ndarray[complex_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         list bases,
                         int r2c, int M, int start):
    cdef int k, l, m, i
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c < 0

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            P0 = bases[1].evaluate_basis(x1, l, P0)
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    b[i] += u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w2[m]*x2[i]))
    return b

def _evaluate_lm_3D_crc1(np.ndarray[real_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         list bases,
                         int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 0 or r2c == 2

    for l in range(u.shape[1]):
        P0 = bases[1].evaluate_basis(x1, l, P0)
        for k in range(u.shape[0]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w1[m]*x2[i]))).real
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    elif r2c == 2:
                        ii = m + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

def _evaluate_lm_3D_ccr0(np.ndarray[complex_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         list bases,
                         int r2c, int M, int start):
    cdef int k, l, m, i
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c < 0

    for m in range(u.shape[2]):
        P0 = bases[2].evaluate_basis(x2, m, P0)
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                for i in range(b.shape[0]):
                    b[i] += u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))
    return b

def _evaluate_lm_3D_ccr1(np.ndarray[real_t, ndim=1] b,
                         np.ndarray[complex_t, ndim=3] u,
                         np.ndarray[real_t, ndim=1] x0,
                         np.ndarray[real_t, ndim=1] x1,
                         np.ndarray[real_t, ndim=1] x2,
                         np.ndarray[real_t, ndim=1] w0,
                         np.ndarray[real_t, ndim=1] w1,
                         np.ndarray[real_t, ndim=1] w2,
                         list bases,
                         int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 0 or r2c == 1

    for m in range(u.shape[2]):
        P0 = bases[2].evaluate_basis(x2, m, P0)
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                for i in range(b.shape[0]):
                    p = (u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))).real
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    elif r2c == 1:
                        ii = l + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b

