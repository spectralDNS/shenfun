#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

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
    cdef np.ndarray[complex_t, ndim=3] c = np.zeros((b.shape[0], u.shape[0], u.shape[1]), dtype=np.complex)
    cdef np.ndarray[complex_t, ndim=2] c2 = np.zeros((b.shape[0], u.shape[0]), dtype=np.complex)

    #for k in range(u.shape[0]):
    #    for l in range(u.shape[1]):
    #        for m in range(u.shape[2]):
    #            for i in range(b.shape[0]):
    #                b[i] = b[i] + u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]
    #return b
    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                for m in range(u.shape[2]):
                    c[i, k, l] += u[k, l, m] * P2[i, m]

    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                c2[i, k] += c[i, k, l] * P1[i, l]

    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            b[i] += c2[i, k] * P0[i, k]
    return b

def _evaluate_3D_ccc1(np.ndarray[real_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[complex_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef complex_t p1, p2
    cdef real_t p, ur, ui

    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            if r2c == 0:
                ii = k + start
            for l in range(u.shape[1]):
                p1 = P0[i, k] * P1[i, l]
                if r2c == 1:
                    ii = l + start
                for m in range(u.shape[2]):
                    #p = (u[k, l, m] * p1 * P2[i, m]).real
                    p2 = p1 * P2[i, m]
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    p = ur*p2.real - ui*p2.imag
                    b[i] += p
                    if r2c == 2:
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
    cdef np.ndarray[complex_t, ndim=3] c = np.zeros((b.shape[0], u.shape[0], u.shape[1]), dtype=np.complex)
    cdef np.ndarray[complex_t, ndim=2] c2 = np.zeros((b.shape[0], u.shape[0]), dtype=np.complex)


    assert r2c < 0

    #for k in range(u.shape[0]):
    #    for l in range(u.shape[1]):
    #        for m in range(u.shape[2]):
    #            for i in range(b.shape[0]):
    #                b[i] += P0[i, k] * (u[k, l, m] * P1[i, l] * P2[i, m])

    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                for m in range(u.shape[2]):
                    c[i, k, l] += u[k, l, m] * P2[i, m]

    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                c2[i, k] += c[i, k, l] * P1[i, l]

    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            b[i] += c2[i, k] * P0[i, k]

    return b

def _evaluate_3D_rcc1(np.ndarray[real_t, ndim=1] b,
                      np.ndarray[complex_t, ndim=3] u,
                      np.ndarray[real_t, ndim=2] P0,
                      np.ndarray[complex_t, ndim=2] P1,
                      np.ndarray[complex_t, ndim=2] P2,
                      int r2c, int M, int start):
    cdef int k, l, m, i, ii
    cdef real_t p
    cdef complex_t p0

    assert r2c == 1 or r2c == 2

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    p0 = P1[i, l] * P2[i, m]
                    p = P0[i, k]*(u[k, l, m].real*p0.real - u[k, l, m].imag*p0.imag)
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
    cdef complex_t p0

    assert r2c == 0 or r2c == 2

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    p0 = P0[i, k] * P2[i, m]
                    p = P1[i, l]*(u[k, l, m].real*p0.real - u[k, l, m].imag*p0.imag)
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
    cdef complex_t p0

    assert r2c == 0 or r2c == 1

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #p = (u[k, l, m] * P0[i, k] * P1[i, l] * P2[i, m]).real
                    p0 = P0[i, k] * P1[i, l]
                    p = P2[i, m]*(u[k, l, m].real*p0.real - u[k, l, m].imag*p0.imag)
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
                #b[i] += u[k, l] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))
                xx = w0[k]*x0[i] + w1[l]*x1[i]
                yc = cos(xx)
                ys = sin(xx)
                ur = u[k, l].real
                ui = u[k, l].imag
                b[i].real = b[i].real + (ur * yc - ui * ys)
                b[i].imag = b[i].imag + (ur * ys + ui * yc)

    return b

def _evaluate_lm_2D_cc1(np.ndarray[real_t, ndim=1] b,
                        np.ndarray[complex_t, ndim=2] u,
                        np.ndarray[real_t, ndim=1] x0,
                        np.ndarray[real_t, ndim=1] x1,
                        np.ndarray[real_t, ndim=1] w0,
                        np.ndarray[real_t, ndim=1] w1,
                        int r2c, int M, int start):
    cdef int k, l, i, ii
    cdef real_t p, xx, yc, ys, ur, ui

    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                #p = (u[k, l] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))).real
                xx = w0[k]*x0[i] + w1[l]*x1[i]
                yc = cos(xx)
                ys = sin(xx)
                ur = u[k, l].real
                ui = u[k, l].imag
                p = (ur * yc - ui * ys)
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
    cdef real_t xx, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c < 0
    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                #b[i] += u[k, l] * P0[i] * exp(1j*w1[l]*x1[i])
                xx = w1[l]*x1[i]
                yc = cos(xx)
                ys = sin(xx)
                ur = u[k, l].real
                ui = u[k, l].imag
                b[i].real = b[i].real + P0[i]*(ur * yc - ui * ys)
                b[i].imag = b[i].imag + P0[i]*(ur * ys + ui * yc)

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
    cdef real_t p, xx, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for i in range(b.shape[0]):
                #p = (u[k, l] * P0[i] * exp(1j*w1[l]*x1[i])).real
                xx = w1[l]*x1[i]
                yc = cos(xx)
                ys = sin(xx)
                ur = u[k, l].real
                ui = u[k, l].imag
                p = P0[i]*(ur * yc - ui * ys)
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
    cdef real_t xx, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)

    for l in range(u.shape[1]):
        P0 = bases[1].evaluate_basis(x1, l, P0)
        for k in range(u.shape[0]):
            for i in range(b.shape[0]):
                #b[i] += u[k, l] * P0[i] * exp(1j*w0[k]*x0[i])
                xx = w0[k]*x0[i]
                yc = cos(xx)
                ys = sin(xx)
                ur = u[k, l].real
                ui = u[k, l].imag
                b[i].real = b[i].real + P0[i]*(ur * yc - ui * ys)
                b[i].imag = b[i].imag + P0[i]*(ur * ys + ui * yc)

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
    cdef real_t p, y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)

    for l in range(u.shape[1]):
        P0 = bases[1].evaluate_basis(x1, l, P0)
        for k in range(u.shape[0]):
            for i in range(b.shape[0]):
                #p = (u[k, l] * P0[i] * exp(1j*w0[k]*x0[i])).real
                y2 = w0[k]*x0[i]
                yc = cos(y2)
                ys = sin(y2)
                ur = u[k, l].real
                ui = u[k, l].imag
                p = P0[i]*(ur * yc - ui * ys)
                b[i] += p
                if r2c == 0:
                    ii = k + start
                elif r2c == 1:
                    ii = l + start
                if ii > 0 & ii < M:
                    b[i] += p
    return b

def evaluate_lm_3D(list bases, b, u, x0, x1, x2, w0, w1, w2, int r2c, int M, int start):

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

def _evaluate_lm_3D_ccc0(complex_t[::1] b,
                         complex_t[:, :, ::1] u,
                         real_t[::1] x0,
                         real_t[::1] x1,
                         real_t[::1] x2,
                         real_t[::1] w0,
                         real_t[::1] w1,
                         real_t[::1] w2,
                         int r2c, int M, int start):
    cdef int k, l, m, i
    cdef double xx, y0, y1, yc, ys, br, bi, ur, ui
    for i in range(b.shape[0]):
        br = 0.0
        bi = 0.0
        for k in range(u.shape[0]):
            y0 = w0[k]*x0[i]
            for l in range(u.shape[1]):
                y1 = y0 + w1[l]*x1[i]
                for m in range(u.shape[2]):
                    xx = y1 + w2[m]*x2[i]
                    #b[i] += u[k, l, m] * (cos(xx) + 1j*sin(xx))
                    yc = cos(xx)
                    ys = sin(xx)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    br += (ur * yc - ui * ys)
                    bi += (ur * ys + ui * yc)
        b[i] = br + bi*1j

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
    cdef real_t p, y0, y1, y2, yc, ys, ur, ui
    for i in range(b.shape[0]):
        for k in range(u.shape[0]):
            y0 = w0[k]*x0[i]
            for l in range(u.shape[1]):
                y1 = y0 + w1[l]*x1[i]
                for m in range(u.shape[2]):
                    #p = (u[k, l, m] * exp(1j*(y1+w2[m]*x2[i]))).real
                    y2 = y1+w2[m]*x2[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    p = (ur * yc - ui * ys)
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
    cdef real_t y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 1 or r2c == 2 or r2c < 0

    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #b[i] += u[k, l, m] * P0[i] * exp(1j*(w1[l]*x1[i] + w2[m]*x2[i]))
                    y2 = w1[l]*x1[i] + w2[m]*x2[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    b[i].real = b[i].real + P0[i]*(ur * yc - ui * ys)
                    b[i].imag = b[i].imag + P0[i]*(ur * ys + ui * yc)

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
    cdef real_t p, y0, y1, y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 1 or r2c == 2

    for k in range(u.shape[0]):
        P0 = bases[0].evaluate_basis(x0, k, P0)
        for l in range(u.shape[1]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #p = (u[k, l, m] * P0[i] * exp(1j*(w1[l]*x1[i] + w2[m]*x2[i]))).real
                    y2 = w1[l]*x1[i] + w2[m]*x2[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    p = P0[i]*(ur * yc - ui * ys)
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
    cdef real_t y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c < 0

    for l in range(u.shape[1]):
        P0 = bases[1].evaluate_basis(x1, l, P0)
        for k in range(u.shape[0]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #b[i] += u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w2[m]*x2[i]))
                    y2 = w0[k]*x0[i] + w2[m]*x2[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    b[i].real = b[i].real + P0[i]*(ur * yc - ui * ys)
                    b[i].imag = b[i].imag + P0[i]*(ur * ys + ui * yc)
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
    cdef real_t p, y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 0 or r2c == 2

    for l in range(u.shape[1]):
        P0 = bases[1].evaluate_basis(x1, l, P0)
        for k in range(u.shape[0]):
            for m in range(u.shape[2]):
                for i in range(b.shape[0]):
                    #p = (u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w1[m]*x2[i]))).real
                    y2 = w0[k]*x0[i] + w1[m]*x2[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    p = P0[i]*(ur * yc - ui * ys)
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
    cdef real_t y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c < 0

    for m in range(u.shape[2]):
        P0 = bases[2].evaluate_basis(x2, m, P0)
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                for i in range(b.shape[0]):
                    #b[i] += u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))
                    y2 = w0[k]*x0[i] + w1[l]*x1[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    b[i].real = b[i].real + P0[i]*(ur * yc - ui * ys)
                    b[i].imag = b[i].imag + P0[i]*(ur * ys + ui * yc)

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
    cdef real_t p, y2, yc, ys, ur, ui
    cdef np.ndarray[real_t, ndim=1] P0 = np.zeros(x0.shape[0], dtype=np.float)
    assert r2c == 0 or r2c == 1

    for m in range(u.shape[2]):
        P0 = bases[2].evaluate_basis(x2, m, P0)
        for k in range(u.shape[0]):
            for l in range(u.shape[1]):
                for i in range(b.shape[0]):
                    #p = (u[k, l, m] * P0[i] * exp(1j*(w0[k]*x0[i] + w1[l]*x1[i]))).real
                    y2 = w0[k]*x0[i] + w1[l]*x1[i]
                    yc = cos(y2)
                    ys = sin(y2)
                    ur = u[k, l, m].real
                    ui = u[k, l, m].imag
                    p = P0[i]*(ur * yc - ui * ys)
                    b[i] += p
                    if r2c == 0:
                        ii = k + start
                    elif r2c == 1:
                        ii = l + start
                    if ii > 0 & ii < M:
                        b[i] += p
    return b
