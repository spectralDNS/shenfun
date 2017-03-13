#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.algorithm cimport copy

ctypedef fused T:
    np.float64_t
    np.complex128_t

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real


def PDMA_SymLU(np.ndarray[np.float64_t, ndim=1, mode='c'] d,
               np.ndarray[np.float64_t, ndim=1, mode='c'] e,
               np.ndarray[np.float64_t, ndim=1, mode='c'] f):
    cdef:
        unsigned int n = d.shape[0]
        unsigned int m = e.shape[0]
        unsigned int k = n - m
        unsigned int i
        double lam

    for i in xrange(n-2*k):
        lam = e[i]/d[i]
        d[i+k] -= lam*e[i]
        e[i+k] -= lam*f[i]
        e[i] = lam
        lam = f[i]/d[i]
        d[i+2*k] -= lam*f[i]
        f[i] = lam

    lam = e[n-4]/d[n-4]
    d[n-2] -= lam*e[n-4]
    e[n-4] = lam
    lam = e[n-3]/d[n-3]
    d[n-1] -= lam*e[n-3]
    e[n-3] = lam

def PDMA_Symsolve(np.ndarray[np.float64_t, ndim=1, mode='c'] d,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] e,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] f,
                  np.ndarray[T, ndim=1, mode='c'] b):
    cdef:
        unsigned int n = d.shape[0]
        int k

    b[2] -= e[0]*b[0]
    b[3] -= e[1]*b[1]
    for k in range(4, n):
        b[k] -= (e[k-2]*b[k-2] + f[k-4]*b[k-4])

    b[n-1] /= d[n-1]
    b[n-2] /= d[n-2]
    b[n-3] /= d[n-3]
    b[n-3] -= e[n-3]*b[n-1]
    b[n-4] /= d[n-4]
    b[n-4] -= e[n-4]*b[n-2]
    for k in range(n-5,-1,-1):
        b[k] /= d[k]
        b[k] -= (e[k]*b[k+2] + f[k]*b[k+4])

def PDMA_Symsolve3D(np.float64_t [::1] d,
                    np.float64_t [::1] e,
                    np.float64_t [::1] f,
                    T [:, :, ::1] b,
                    np.int64_t axis):
    cdef:
        int i, j, k
        int n = d.shape[0]

    if axis == 0:
        for i in xrange(b.shape[1]):
            for j in xrange(b.shape[2]):
                #PDMA_Symsolve(d, e, f, b[:, i, j])
                b[2, i, j] -= e[0]*b[0, i, j]
                b[3, i, j] -= e[1]*b[1, i, j]

        for k in xrange(4, n):
            for i in xrange(b.shape[1]):
                for j in xrange(b.shape[2]):
                    b[k, i, j] -= (e[k-2]*b[k-2, i, j] + f[k-4]*b[k-4, i, j])

        for i in xrange(b.shape[1]):
            for j in xrange(b.shape[2]):
                b[n-1, i, j] /= d[n-1]
                b[n-2, i, j] /= d[n-2]
                b[n-3, i, j] /= d[n-3]
                b[n-3, i, j] -= e[n-3]*b[n-1, i, j]
                b[n-4, i, j] /= d[n-4]
                b[n-4, i, j] -= e[n-4]*b[n-2, i, j]

        for k in xrange(n-5,-1,-1):
            for i in xrange(b.shape[1]):
                for j in xrange(b.shape[2]):
                    b[k, i, j] /= d[k]
                    b[k, i, j] -= (e[k]*b[k+2, i, j] + f[k]*b[k+4, i, j])

    elif axis == 1:
        for i in xrange(b.shape[0]):
            for k in xrange(b.shape[2]):
                b[i, 2, k] -= e[0]*b[i, 0, k]
                b[i, 3, k] -= e[1]*b[i, 1, k]

        for i in xrange(b.shape[0]):
            for j in xrange(4, n):
                for k in xrange(b.shape[2]):
                    b[i, j, k] -= (e[j-2]*b[i, j-2, k] + f[j-4]*b[i, j-4, k])

        for i in xrange(b.shape[0]):
            for k in xrange(b.shape[2]):
                b[i, n-1, k] /= d[n-1]
                b[i, n-2, k] /= d[n-2]
                b[i, n-3, k] /= d[n-3]
                b[i, n-3, k] -= e[n-3]*b[i, n-1, k]
                b[i, n-4, k] /= d[n-4]
                b[i, n-4, k] -= e[n-4]*b[i, n-2, k]

        for i in xrange(b.shape[0]):
            for j in xrange(n-5,-1,-1):
                for k in xrange(b.shape[2]):
                    b[i, j, k] /= d[j]
                    b[i, j, k] -= (e[j]*b[i, j+2, k] + f[j]*b[i, j+4, k])

    elif axis == 2:
        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                b[i, j, 2] -= e[0]*b[i, j, 0]
                b[i, j, 3] -= e[1]*b[i, j, 1]

        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                for k in xrange(4, n):
                    b[i, j, k] -= (e[k-2]*b[i, j, k-2] + f[k-4]*b[i, j, k-4])

        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                b[i, j, n-1] /= d[n-1]
                b[i, j, n-2] /= d[n-2]
                b[i, j, n-3] /= d[n-3]
                b[i, j, n-3] -= e[n-3]*b[i, j, n-1]
                b[i, j, n-4] /= d[n-4]
                b[i, j, n-4] -= e[n-4]*b[i, j, n-2]

        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                for k in xrange(n-5,-1,-1):
                    b[i, j, k] /= d[k]
                    b[i, j, k] -= (e[k]*b[i, j, k+2] + f[k]*b[i, j, k+4])


def TDMA_SymLU(np.ndarray[real_t, ndim=1, mode='c'] d,
               np.ndarray[real_t, ndim=1, mode='c'] a,
               np.ndarray[real_t, ndim=1, mode='c'] l):
    cdef:
        unsigned int n = d.shape[0]
        int i

    for i in range(2, n):
        l[i-2] = a[i-2]/d[i-2]
        d[i] = d[i] - l[i-2]*a[i-2]

#def TDMA_SymLU(real_t[:] d,
               #real_t[:] a,
               #real_t[:] l):
    #cdef:
        #unsigned int n = d.shape[0]
        #int i

    #for i in range(2, n):
        #l[i-2] = a[i-2]/d[i-2]
        #d[i] = d[i] - l[i-2]*a[i-2]

def TDMA_SymSolve(np.ndarray[real_t, ndim=1, mode='c'] d,
                  np.ndarray[real_t, ndim=1, mode='c'] a,
                  np.ndarray[real_t, ndim=1, mode='c'] l,
                  np.ndarray[T, ndim=1, mode='c'] x):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i

    for i in range(2, n):
        x[i] -= l[i-2]*x[i-2]

    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - a[i]*x[i+2])/d[i]

def TDMA_SymSolve3D(real_t[::1] d,
                    real_t[::1] a,
                    real_t[::1] l,
                    np.ndarray[T, ndim=3] x,
                    np.int64_t axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, k

    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[i-2]*x[i-2, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1]
                x[n-2, j, k] = x[n-2, j, k]/d[n-2]

        for i in range(n - 3, -1, -1):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i]*x[i+2, j, k])/d[i]

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[j-2]*x[i, j-2, k]

        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]
                x[i, n-2, k] = x[i, n-2, k]/d[n-2]

        for i in range(x.shape[0]):
            for j in range(n - 3, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[j]*x[i, j+2, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(2, n):
                    x[i, j, k] -= l[k-2]*x[i, j, k-2]

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j, n-1] = x[i, j, n-1]/d[n-1]
                x[i, j, n-2] = x[i, j, n-2]/d[n-2]

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(n - 3, -1, -1):
                    x[i, j, k] = (x[i, j, k] - a[k]*x[i, j, k+2])/d[k]


#def TDMA_SymSolve3D(np.ndarray[real_t, ndim=1, mode='c'] d,
                    #np.ndarray[real_t, ndim=1, mode='c'] a,
                    #np.ndarray[real_t, ndim=1, mode='c'] l,
                    #np.ndarray[T, ndim=3, mode='c'] x):
    #cdef:
        #unsigned int n = d.shape[0]
        #int i, j, k
        #np.ndarray[T, ndim=3, mode='c'] y = np.zeros_like(x)

    #for j in range(x.shape[1]):
        #for k in range(x.shape[2]):
            #y[0, j, k] = x[0, j, k]
            #y[1, j, k] = x[1, j, k]

    #for i in range(2, n):
        #for j in range(x.shape[1]):
            #for k in range(x.shape[2]):
                #y[i, j, k] = x[i, j, k] - l[i-2]*y[i-2, j, k]

    #for j in range(x.shape[1]):
        #for k in range(x.shape[2]):
            #x[n-1, j, k] = y[n-1, j, k]/d[n-1]
            #x[n-2, j, k] = y[n-2, j, k]/d[n-2]

    #for i in range(n - 3, -1, -1):
        #for j in range(x.shape[1]):
            #for k in range(x.shape[2]):
                #x[i, j, k] = (y[i, j, k] - a[i]*x[i+2, j, k])/d[i]

def TDMA_1D(np.ndarray[real_t, ndim=1, mode='c'] a,
            np.ndarray[real_t, ndim=1, mode='c'] b,
            np.ndarray[real_t, ndim=1, mode='c'] bc,
            np.ndarray[real_t, ndim=1, mode='c'] c,
            np.ndarray[T, ndim=1, mode='c'] d):
    cdef:
        unsigned int n = b.shape[0]
        unsigned int m = a.shape[0]
        unsigned int k = n - m
        int i

    bc[0] = b[0]
    bc[1] = b[1]
    for i in range(m):
        d[i + k] -= d[i] * a[i] / bc[i]
        bc[i + k] = b[i + k] - c[i] * a[i] / bc[i]
    for i in range(m - 1, -1, -1):
        d[i] -= d[i + k] * c[i] / bc[i + k]
    for i in range(n):
        d[i] /= bc[i]


def TDMA_3D(np.ndarray[real_t, ndim=1, mode='c'] a,
            np.ndarray[real_t, ndim=1, mode='c'] b,
            np.ndarray[real_t, ndim=1, mode='c'] bc,
            np.ndarray[real_t, ndim=1, mode='c'] c,
            np.ndarray[T, ndim=3, mode='c'] d):
    cdef:
        int n = b.shape[0]
        int m = a.shape[0]
        int l = n - m
        int i, j, k

    bc[0] = b[0]
    bc[1] = b[1]
    for i in range(m):
        bc[i + l] = b[i + l] - c[i] * a[i] / bc[i]

    for i in range(m):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                d[i + l, j, k] -= d[i, j, k] * a[i] / bc[i]

    for i in range(m - 1, -1, -1):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                d[i, j, k] -= d[i + l, j, k] * c[i] / bc[i + l]

    for i in range(n):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                d[i, j, k] = d[i, j, k] / bc[i]


def LU_Helmholtz_1D(A, B,
                    np.float_t A_scale,
                    np.float_t B_scale,
                    np.ndarray[real_t, ndim=1] d0,
                    np.ndarray[real_t, ndim=1] d1,
                    np.ndarray[real_t, ndim=1] d2,
                    np.ndarray[real_t, ndim=1] L):
    cdef:
        int i, N
        np.ndarray[real_t, ndim=1] A_0 = A[0]
        np.ndarray[real_t, ndim=1] A_2 = A[2]
        np.ndarray[real_t, ndim=1] A_4 = A[4]
        np.ndarray[real_t, ndim=1] B_m2 = B[-2]
        np.ndarray[real_t, ndim=1] B_0 = B[0]
        np.ndarray[real_t, ndim=1] B_2 = B[2]

    N = A_0.shape[0]
    d0[0] =  A_scale*A_0[0] + B_scale*B_0[0]
    d0[1] =  A_scale*A_0[1] + B_scale*B_0[1]
    d1[0] =  A_scale*A_2[0] + B_scale*B_2[0]
    d1[1] =  A_scale*A_2[1] + B_scale*B_2[1]
    d2[0] =  A_scale*A_4[0]
    d2[1] =  A_scale*A_4[1]
    for i in xrange(2, N):
        L[i-2] = B_scale*B_m2[i-2] / d0[i-2]
        d0[i] = A_scale*A_0[i] + B_scale*B_0[i] - L[i-2]*d1[i-2]
        if i < N-2:
            d1[i] = A_scale*A_2[i] + B_scale*B_2[i] - L[i-2]*d2[i-2]
        if i < N-4:
            d2[i] = A_scale*A_4[i] - L[i-2]*d2[i-2]


def Solve_Helmholtz_1D(np.ndarray[T, ndim=1] fk,
                       np.ndarray[T, ndim=1] u_hat,
                       np.ndarray[real_t, ndim=1] d0,
                       np.ndarray[real_t, ndim=1] d1,
                       np.ndarray[real_t, ndim=1] d2,
                       np.ndarray[real_t, ndim=1] L):
    cdef:
        int i, j, N
        vector[T] y
        T sum_even = 0.0
        T sum_odd = 0.0

    N = d0.shape[0]
    y.resize(N)
    y[0] = fk[0]
    y[1] = fk[1]
    for i in xrange(2, N):
        y[i] = fk[i] - L[i-2]*y[i-2]

    u_hat[N-1] = y[N-1] / d0[N-1]
    u_hat[N-2] = y[N-2] / d0[N-2]
    u_hat[N-3] = (y[N-3] - d1[N-3]*u_hat[N-1]) / d0[N-3]
    u_hat[N-4] = (y[N-4] - d1[N-4]*u_hat[N-2]) / d0[N-4]
    for i in xrange(N-5, -1, -1):
        u_hat[i] = y[i] - d1[i]*u_hat[i+2]
        if i % 2 == 0:
            sum_even += u_hat[i+4]
            u_hat[i] -= d2[i]*sum_even
        else:
            sum_odd += u_hat[i+4]
            u_hat[i] -= d2[i]*sum_odd
        u_hat[i]/=d0[i]

def LU_Helmholtz_3D(A, B, np.int64_t axis,
                    np.ndarray[real_t, ndim=3] A_scale,
                    np.ndarray[real_t, ndim=3] B_scale,
                    np.ndarray[real_t, ndim=3] d0,
                    np.ndarray[real_t, ndim=3] d1,
                    np.ndarray[real_t, ndim=3] d2,
                    np.ndarray[real_t, ndim=3] L):
    cdef:
        unsigned int ii, jj

    if axis == 0:
        for ii in range(d0.shape[1]):
            for jj in range(d0.shape[2]):
                LU_Helmholtz_1D(A, B,
                                A_scale[0, ii, jj],
                                B_scale[0, ii, jj],
                                d0[:, ii, jj],
                                d1[:, ii, jj],
                                d2[:, ii, jj],
                                L [:, ii, jj])

    elif axis == 1:
        for ii in range(d0.shape[0]):
            for jj in range(d0.shape[2]):
                LU_Helmholtz_1D(A, B,
                                A_scale[ii, 0, jj],
                                B_scale[ii, 0, jj],
                                d0[ii, :, jj],
                                d1[ii, :, jj],
                                d2[ii, :, jj],
                                L [ii, :, jj])

    elif axis == 2:
        for ii in range(d0.shape[0]):
            for jj in range(d0.shape[1]):
                LU_Helmholtz_1D(A, B,
                                A_scale[ii, jj, 0],
                                B_scale[ii, jj, 0],
                                d0[ii, jj, :],
                                d1[ii, jj, :],
                                d2[ii, jj, :],
                                L [ii, jj, :])

def Solve_Helmholtz_3D(np.int64_t axis,
                       np.ndarray[T, ndim=3] fk,
                       np.ndarray[T, ndim=3] u_hat,
                       np.ndarray[real_t, ndim=3] d0,
                       np.ndarray[real_t, ndim=3] d1,
                       np.ndarray[real_t, ndim=3] d2,
                       np.ndarray[real_t, ndim=3] L):
    cdef:
        unsigned int ii, jj

    if axis == 0:
        for ii in range(d0.shape[1]):
            for jj in range(d0.shape[2]):
                Solve_Helmholtz_1D(fk[:, ii, jj],
                                   u_hat[:, ii, jj],
                                   d0[:, ii, jj],
                                   d1[:, ii, jj],
                                   d2[:, ii, jj],
                                   L [:, ii, jj])

    elif axis == 1:
        for ii in range(d0.shape[0]):
            for jj in range(d0.shape[2]):
                Solve_Helmholtz_1D(fk[ii, :, jj],
                                   u_hat[ii, :, jj],
                                   d0[ii, :, jj],
                                   d1[ii, :, jj],
                                   d2[ii, :, jj],
                                   L [ii, :, jj])

    elif axis == 2:
        for ii in range(d0.shape[0]):
            for jj in range(d0.shape[1]):
                Solve_Helmholtz_1D(fk[ii, jj, :],
                                   u_hat[ii, jj, :],
                                   d0[ii, jj, :],
                                   d1[ii, jj, :],
                                   d2[ii, jj, :],
                                   L [ii, jj, :])

def LU_Helmholtz_2D(A, B, np.int64_t axis,
                    np.ndarray[real_t, ndim=2] A_scale,
                    np.ndarray[real_t, ndim=2] B_scale,
                    np.ndarray[real_t, ndim=2] d0,
                    np.ndarray[real_t, ndim=2] d1,
                    np.ndarray[real_t, ndim=2] d2,
                    np.ndarray[real_t, ndim=2] L):
    cdef:
        unsigned int ii

    if axis == 0:
        for ii in range(d0.shape[1]):
            LU_Helmholtz_1D(A, B,
                            A_scale[0, ii],
                            B_scale[0, ii],
                            d0[:, ii],
                            d1[:, ii],
                            d2[:, ii],
                            L [:, ii])

    elif axis == 1:
        for ii in range(d0.shape[0]):
            LU_Helmholtz_1D(A, B,
                            A_scale[ii, 0],
                            B_scale[ii, 0],
                            d0[ii, :],
                            d1[ii, :],
                            d2[ii, :],
                            L [ii, :])

def Solve_Helmholtz_2D(np.int64_t axis,
                       np.ndarray[T, ndim=2] fk,
                       np.ndarray[T, ndim=2] u_hat,
                       np.ndarray[real_t, ndim=2] d0,
                       np.ndarray[real_t, ndim=2] d1,
                       np.ndarray[real_t, ndim=2] d2,
                       np.ndarray[real_t, ndim=2] L):
    cdef:
        unsigned int ii

    if axis == 0:
        for ii in range(d0.shape[1]):
            Solve_Helmholtz_1D(fk[:, ii],
                                u_hat[:, ii],
                                d0[:, ii],
                                d1[:, ii],
                                d2[:, ii],
                                L [:, ii])

    elif axis == 1:
        for ii in range(d0.shape[0]):
            Solve_Helmholtz_1D(fk[ii, :],
                                u_hat[ii, :],
                                d0[ii, :],
                                d1[ii, :],
                                d2[ii, :],
                                L [ii, :])

