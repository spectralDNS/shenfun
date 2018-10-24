#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from libc.math cimport M_PI
from cython.parallel import prange

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t

ctypedef fused T:
    real_t
    complex_t

def imult(T[:, :, ::1] array, real_t scale):
    cdef int i, j, k

    #for i in prange(array.shape[0], nogil=True, num_threads=1):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                array[i, j, k] *= scale
    return array


def CDNmat_matvec(np.ndarray[real_t, ndim=1] ud,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[T, ndim=3] v,
                  np.ndarray[T, ndim=3] b,
                  np.int64_t axis):
    cdef:
        int i, j, k
        int N = v.shape[0]-2

    if axis == 0:
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[0, j, k] = ud[0]*v[1, j, k]
                b[N-1, j, k] = ld[N-2]*v[N-2, j, k]

        for i in xrange(1, N-1):
            for j in xrange(b.shape[1]):
                for k in xrange(b.shape[2]):
                    b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-1]*v[i-1, j, k]

    elif axis == 1:
        for i in xrange(b.shape[0]):
            for k in xrange(b.shape[2]):
                b[i, 0, k] = ud[0]*v[i, 1, k]
                b[i, N-1, k] = ld[N-2]*v[i, N-2, k]

        for i in xrange(b.shape[0]):
            for j in xrange(1, N-1):
                for k in xrange(b.shape[2]):
                    b[i, j, k] = ud[j]*v[i, j+1, k] + ld[j-1]*v[i, j-1, k]

    elif axis == 2:
        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                b[i, j, 0] = ud[0]*v[i, j, 1]
                b[i, j, N-1] = ld[N-2]*v[i, j, N-2]

        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                for k in xrange(1, N-1):
                    b[i, j, k] = ud[k]*v[i, j, k+1] + ld[k-1]*v[i, j, k-1]


def BDNmat_matvec(real_t ud,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[real_t, ndim=1] dd,
                  np.ndarray[T, ndim=3] v,
                  np.ndarray[T, ndim=3] b,
                  np.int64_t axis):
    cdef:
        int i, j, k
        int N = v.shape[0]-2

    if axis == 0:
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[0, j, k] = ud*v[2, j, k] + dd[0]*v[0, j, k]
                b[1, j, k] = ud*v[3, j, k] + dd[1]*v[1, j, k]
                b[N-2, j, k] = ld[N-4]*v[N-4, j, k] + dd[N-2]*v[N-2, j, k]
                b[N-1, j, k] = ld[N-3]*v[N-3, j, k] + dd[N-1]*v[N-1, j, k]

        for i in xrange(2, N-2):
            for j in xrange(b.shape[1]):
                for k in xrange(b.shape[2]):
                    b[i, j, k] = ud*v[i+2, j, k] + dd[i]*v[i, j, k] + ld[i-2]*v[i-2, j, k]

    elif axis == 1:
        for i in xrange(b.shape[0]):
            for k in xrange(b.shape[2]):
                b[i, 0, k] = ud*v[i, 2, k] + dd[0]*v[i, 0, k]
                b[i, 1, k] = ud*v[i, 3, k] + dd[1]*v[i, 1, k]
                b[i, N-2, k] = ld[N-4]*v[i, N-4, k] + dd[N-2]*v[i, N-2, k]
                b[i, N-1, k] = ld[N-3]*v[i, N-3, k] + dd[N-1]*v[i, N-1, k]

        for i in xrange(b.shape[0]):
            for j in xrange(2, N-2):
                for k in xrange(b.shape[2]):
                    b[i, j, k] = ud*v[i, j+2, k] + dd[j]*v[i, j, k] + ld[j-2]*v[i, j-2, k]

    elif axis == 2:
        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                b[i, j, 0] = ud*v[i, j, 2] + dd[0]*v[i, j, 0]
                b[i, j, 1] = ud*v[i, j, 3] + dd[1]*v[i, j, 1]
                b[i, j, N-2] = ld[N-4]*v[i, j, N-4] + dd[N-2]*v[i, j, N-2]
                b[i, j, N-1] = ld[N-3]*v[i, j, N-3] + dd[N-1]*v[i, j, N-1]

        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                for k in xrange(2, N-2):
                    b[i, j, k] = ud*v[i, j, k+2] + dd[k]*v[i, j, k] + ld[k-2]*v[i, j, k-2]


def CDDmat_matvec(np.ndarray[real_t, ndim=1] ud,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[T, ndim=3] v,
                  np.ndarray[T, ndim=3] b,
                  np.int64_t axis):
    cdef:
        int i, j, k
        int N = v.shape[axis]-2

    if axis == 0:
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[0, j, k] = ud[0]*v[1, j, k]
                b[N-1, j, k] = ld[N-2]*v[N-2, j, k]

        for i in xrange(1, N-1):
            for j in xrange(b.shape[1]):
                for k in xrange(b.shape[2]):
                    b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-1]*v[i-1, j, k]

    elif axis == 1:
        for i in xrange(b.shape[0]):
            for k in xrange(b.shape[2]):
                b[i, 0, k] = ud[0]*v[i, 1, k]
                b[i, N-1, k] = ld[N-2]*v[i, N-2, k]

        for i in xrange(b.shape[0]):
            for j in xrange(1, N-1):
                for k in xrange(b.shape[2]):
                    b[i, j, k] = ud[j]*v[i, j+1, k] + ld[j-1]*v[i, j-1, k]

    elif axis == 2:
        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                b[i, j, 0] = ud[0]*v[i, j, 1]
                b[i, j, N-1] = ld[N-2]*v[i, j, N-2]

        for i in xrange(b.shape[0]):
            for j in xrange(b.shape[1]):
                for k in xrange(1, N-1):
                    b[i, j, k] = ud[k]*v[i, j, k+1] + ld[k-1]*v[i, j, k-1]


def SBBmat_matvec(np.ndarray[T, ndim=1] v,
                  np.ndarray[T, ndim=1] b,
                  np.ndarray[real_t, ndim=1] dd):
    cdef:
        int i, j, k
        int N = v.shape[0]-4
        double p, r
        T d, s1, s2, o1, o2

    j = N-1
    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    b[j] = dd[j]*v[j]
    b[j-1] = dd[j-1]*v[j-1]
    for k in range(N-3, -1, -1):
        j = k+2
        p = k*dd[k]/(k+1)
        r = 24*(k+1)*(k+2)*np.pi
        d = v[j]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k] = dd[k]*v[k] + p*s1 + r*s2
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k] = dd[k]*v[k] + p*o1 + r*o2

def SBBmat_matvec3D(np.ndarray[T, ndim=3] v,
                    np.ndarray[T, ndim=3] b,
                    np.ndarray[real_t, ndim=1] dd,
                    np.int64_t axis):
    cdef:
        int i, j, k, jj, kk
        double p, r, d2
        T d
        np.ndarray[T, ndim=2] s1
        np.ndarray[T, ndim=2] s2
        np.ndarray[T, ndim=2] o1
        np.ndarray[T, ndim=2] o2
        np.ndarray[real_t, ndim=1] pv
        np.ndarray[real_t, ndim=1] rv
        int N = v.shape[axis]-4

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #SBBmat_matvec(v[:, i, j], bb[:, i, j], dd)

    if axis == 0:
        s1 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        s2 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        o1 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        o2 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)

        k = N-1
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[k, i, j] = dd[k]*v[k, i, j]
                b[k-1, i, j] = dd[k-1]*v[k-1, i, j]

        for k in xrange(N-3, -1, -1):
            jj = k+2
            p = k*dd[k]/(k+1.)
            r = 24*(k+1)*(k+2)*M_PI
            d2 = dd[k]
            for i in xrange(v.shape[1]):
                for j in xrange(v.shape[2]):
                    d = v[jj ,i, j]/(jj+3.)
                    if k % 2 == 0:
                        s1[i, j] += d
                        s2[i, j] += (jj+2)*(jj+2)*d
                        b[k, i, j] = d2*v[k, i, j] + p*s1[i, j] + r*s2[i, j]
                    else:
                        o1[i, j] += d
                        o2[i, j] += (jj+2)*(jj+2)*d
                        b[k, i, j] = d2*v[k, i, j] + p*o1[i, j] + r*o2[i, j]

    elif axis == 1:
        s1 = np.zeros((v.shape[0], v.shape[2]), dtype=v.dtype)
        s2 = np.zeros((v.shape[0], v.shape[2]), dtype=v.dtype)
        o1 = np.zeros((v.shape[0], v.shape[2]), dtype=v.dtype)
        o2 = np.zeros((v.shape[0], v.shape[2]), dtype=v.dtype)
        j = N-1
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, j, k] = dd[j]*v[i, j, k]
                b[i, j-1, k] = dd[j-1]*v[i, j-1, k]

        for j in xrange(N-3, -1, -1):
            jj = j+2
            p = j*dd[j]/(j+1.)
            r = 24*(j+1)*(j+2)*M_PI
            d2 = dd[j]
            for i in xrange(v.shape[0]):
                for k in xrange(v.shape[2]):
                    d = v[i, jj, k]/(jj+3.)
                    if j % 2 == 0:
                        s1[i, k] += d
                        s2[i, k] += (jj+2)*(jj+2)*d
                        b[i, j, k] = d2*v[i, j, k] + p*s1[i, k] + r*s2[i, k]
                    else:
                        o1[i, k] += d
                        o2[i, k] += (jj+2)*(jj+2)*d
                        b[i, j, k] = d2*v[i, j, k] + p*o1[i, k] + r*o2[i, k]

    elif axis == 2:
        s1 = np.zeros((v.shape[0], v.shape[1]), dtype=v.dtype)
        s2 = np.zeros((v.shape[0], v.shape[1]), dtype=v.dtype)
        o1 = np.zeros((v.shape[0], v.shape[1]), dtype=v.dtype)
        o2 = np.zeros((v.shape[0], v.shape[1]), dtype=v.dtype)
        pv = np.zeros(v.shape[2])
        rv = np.zeros(v.shape[2])

        k = N-1
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, k] = dd[k]*v[i, j, k]
                b[i, j, k-1] = dd[k-1]*v[i, j, k-1]

        for i in xrange(v.shape[0]):
            for j in xrange(v.shape[1]):
                for k in xrange(N-3, -1, -1):
                    kk = k+2
                    if i+j == 0: # cache for speedup
                        pv[k] = k*dd[k]/(k+1.)
                        rv[k] = 24*(k+1)*(k+2)*M_PI
                    p = pv[k]
                    r = rv[k]
                    d2 = dd[k]
                    d = v[i, j, kk]/(kk+3.)
                    if k % 2 == 0:
                        s1[i, j] += d
                        s2[i, j] += (kk+2)*(kk+2)*d
                        b[i, j, k] = d2*v[i, j, k] + p*s1[i, j] + r*s2[i, j]
                    else:
                        o1[i, j] += d
                        o2[i, j] += (kk+2)*(kk+2)*d
                        b[i, j, k] = d2*v[i, j, k] + p*o1[i, j] + r*o2[i, j]


def ADDmat_matvec(np.ndarray[T, ndim=1] v,
                  np.ndarray[T, ndim=1] b,
                  np.ndarray[real_t, ndim=1] dd):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
        double p
        double pi = np.pi
        T d, s1, s2

    k = N-1
    s1 = 0.0
    s2 = 0.0
    b[k] = dd[k]*v[k]
    b[k-1] = dd[k-1]*v[k-1]
    for k in range(N-3, -1, -1):
        j = k+2
        p = -4*(k+1)*pi
        if j % 2 == 0:
            s1 += v[j]
            b[k] = dd[k]*v[k] + p*s1
        else:
            s2 += v[j]
            b[k] = dd[k]*v[k] + p*s2


def Tridiagonal_matvec3D(T[:, :, ::1] v,
                         T[:, :, ::1] b,
                         real_t[::1] ld,
                         real_t[::1] dd,
                         real_t[::1] ud,
                         np.int64_t axis):
    cdef:
        np.intp_t i, j, k
        np.intp_t N = dd.shape[0]

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #Tridiagonal_matvec(v[:, i, j], b[:, i, j], ld, dd, ud)
    if axis == 0:
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[0, i, j] = dd[0]*v[0, i, j] + ud[0]*v[2, i, j]
                b[1, i, j] = dd[1]*v[1, i, j] + ud[1]*v[3, i, j]

        for k in xrange(2, N-2):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j] + ud[k]*v[k+2, i, j]

        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                k = N-2
                b[k, i, j] = ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j]
                k = N-1
                b[k, i, j] = ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j]
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, 0, k] = dd[0]*v[i, 0, k] + ud[0]*v[i, 2, k]
                b[i, 1, k] = dd[1]*v[i, 1, k] + ud[1]*v[i, 3, k]

        for i in range(v.shape[0]):
            for j in xrange(2, N-2):
                for k in range(v.shape[2]):
                    b[i, j, k] = ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k] + ud[j]*v[i, j+2, k]

        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                j = N-2
                b[i, j, k] = ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k]
                j = N-1
                b[i, j, k] = ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k]

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, 0] = dd[0]*v[i, j, 0] + ud[0]*v[i, j, 2]
                b[i, j, 1] = dd[1]*v[i, j, 1] + ud[1]*v[i, j, 3]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in xrange(2, N-2):
                    b[i, j, k] = ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                k = N-2
                b[i, j, k] = ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k]
                k = N-1
                b[i, j, k] = ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k]



def Tridiagonal_matvec(np.ndarray[T, ndim=1] v,
                       np.ndarray[T, ndim=1] b,
                       real_t[::1] ld,
                       real_t[::1] dd,
                       real_t[::1] ud):
    cdef:
        np.intp_t i
        np.intp_t N = dd.shape[0]

    #for i in xrange(N-2):
        #b[i] = ud[i]*v[i+2]
    #for i in xrange(N):
        #b[i] += dd[i]*v[i]
    #for i in xrange(2, N):
        #b[i] += ld[i-2]*v[i-2]

    b[0] = dd[0]*v[0] + ud[0]*v[2]
    b[1] = dd[1]*v[1] + ud[1]*v[3]
    for i in xrange(2, N-2):
        b[i] = ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-2
    b[i] = ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-1
    b[i] = ld[i-2]* v[i-2] + dd[i]*v[i]


def Tridiagonal_matvec3DT(np.ndarray[T, ndim=3] v,
                          np.ndarray[T, ndim=3] b,
                          np.ndarray[real_t, ndim=1] ld,
                          np.ndarray[real_t, ndim=1] dd,
                          np.ndarray[real_t, ndim=1] ud):
    cdef:
        int i, j, k
        int N = dd.shape[0]

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            b[i, j, 0] = dd[0]*v[i, j, 0] + ud[0]*v[i, j, 2]
            b[i, j, 1] = dd[1]*v[i, j, 1] + ud[1]*v[i, j, 3]
            for k in xrange(2, N-2):
                b[i, j, k] = ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2]
            b[i, j, N-2] = ld[N-4]* v[i, j, N-4] + dd[N-2]*v[i, j, N-2]
            b[i, j, N-1] = ld[N-3]* v[i, j, N-3] + dd[N-1]*v[i, j, N-1]

def Pentadiagonal_matvec3D(np.ndarray[T, ndim=3] v,
                    np.ndarray[T, ndim=3] b,
                    np.ndarray[real_t, ndim=1] ldd,
                    np.ndarray[real_t, ndim=1] ld,
                    np.ndarray[real_t, ndim=1] dd,
                    np.ndarray[real_t, ndim=1] ud,
                    np.ndarray[real_t, ndim=1] udd,
                    np.int64_t axis):
    cdef:
        int i, j, k
        int N = dd.shape[0]

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #Pentadiagonal_matvec(v[:, i, j], b[:, i, j], ld, dd, ud)
    if axis == 0:
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[0, i, j] = dd[0]*v[0, i, j] + ud[0]*v[2, i, j] + udd[0]*v[4, i, j]
                b[1, i, j] = dd[1]*v[1, i, j] + ud[1]*v[3, i, j] + udd[1]*v[5, i, j]
                b[2, i, j] = ld[0]*v[0, i, j] + dd[2]*v[2, i, j] + ud[2]*v[4, i, j] + udd[2]*v[6, i, j]
                b[3, i, j] = ld[1]*v[1, i, j] + dd[3]*v[3, i, j] + ud[3]*v[5, i, j] + udd[3]*v[7, i, j]

        for k in xrange(4, N-4):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = ldd[k-4]*v[k-4, i, j]+ ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j] + ud[k]*v[k+2, i, j] + udd[k]*v[k+4, i, j]

        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                k = N-4
                b[k, i, j] = ldd[k-4]*v[k-4, i, j]+ ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j] + ud[k]*v[k+2, i, j]
                k = N-3
                b[k, i, j] = ldd[k-4]*v[k-4, i, j]+ ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j] + ud[k]*v[k+2, i, j]
                k = N-2
                b[k, i, j] = ldd[k-4]*v[k-4, i, j]+ ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j]
                k = N-1
                b[k, i, j] = ldd[k-4]*v[k-4, i, j]+ ld[k-2]* v[k-2, i, j] + dd[k]*v[k, i, j]

    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, 0, k] = dd[0]*v[i, 0, k] + ud[0]*v[i, 2, k] + udd[0]*v[i, 4, k]
                b[i, 1, k] = dd[1]*v[i, 1, k] + ud[1]*v[i, 3, k] + udd[1]*v[i, 5, k]
                b[i, 2, k] = ld[0]*v[i, 0, k] + dd[2]*v[i, 2, k] + ud[2]*v[i, 4, k] + udd[2]*v[i, 6, k]
                b[i, 3, k] = ld[1]*v[i, 1, k] + dd[3]*v[i, 3, k] + ud[3]*v[i, 5, k] + udd[3]*v[i, 7, k]

        for i in range(v.shape[0]):
            for j in xrange(4, N-4):
                for k in range(v.shape[2]):
                    b[i, j, k] = ldd[j-4]*v[i, j-4, k]+ ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k] + ud[j]*v[i, j+2, k] + udd[j]*v[i, j+4, k]

        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                j = N-4
                b[i, j, k] = ldd[j-4]*v[i, j-4, k]+ ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k] + ud[j]*v[i, j+2, k]
                j = N-3
                b[i, j, k] = ldd[j-4]*v[i, j-4, k]+ ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k] + ud[j]*v[i, j+2, k]
                j = N-2
                b[i, j, k] = ldd[j-4]*v[i, j-4, k]+ ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k]
                j = N-1
                b[i, j, k] = ldd[j-4]*v[i, j-4, k]+ ld[j-2]* v[i, j-2, k] + dd[j]*v[i, j, k]

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, 0] = dd[0]*v[i, j, 0] + ud[0]*v[i, j, 2] + udd[0]*v[i, j, 4]
                b[i, j, 1] = dd[1]*v[i, j, 1] + ud[1]*v[i, j, 3] + udd[1]*v[i, j, 5]
                b[i, j, 2] = ld[0]*v[i, j, 0] + dd[2]*v[i, j, 2] + ud[2]*v[i, j, 4] + udd[2]*v[i, j, 6]
                b[i, j, 3] = ld[1]*v[i, j, 1] + dd[3]*v[i, j, 3] + ud[3]*v[i, j, 5] + udd[3]*v[i, j, 7]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in xrange(4, N-4):
                    b[i, j, k] = ldd[k-4]*v[i, j, k-4]+ ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2] + udd[k]*v[i, j, k+4]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                k = N-4
                b[i, j, k] = ldd[k-4]*v[i, j, k-4]+ ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2]
                k = N-3
                b[i, j, k] = ldd[k-4]*v[i, j, k-4]+ ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2]
                k = N-2
                b[i, j, k] = ldd[k-4]*v[i, j, k-4]+ ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k]
                k = N-1
                b[i, j, k] = ldd[k-4]*v[i, j, k-4]+ ld[k-2]* v[i, j, k-2] + dd[k]*v[i, j, k]



def Pentadiagonal_matvec(T[::1] v,
                         T[::1] b,
                         real_t[::1] ldd,
                         real_t[::1] ld,
                         real_t[::1] dd,
                         real_t[::1] ud,
                         real_t[::1] udd):
    cdef:
        int i
        int N = dd.shape[0]

    b[0] = dd[0]*v[0] + ud[0]*v[2] + udd[0]*v[4]
    b[1] = dd[1]*v[1] + ud[1]*v[3] + udd[1]*v[5]
    b[2] = ld[0]*v[0] + dd[2]*v[2] + ud[2]*v[4] + udd[2]*v[6]
    b[3] = ld[1]*v[1] + dd[3]*v[3] + ud[3]*v[5] + udd[3]*v[7]
    for i in xrange(4, N-4):
        b[i] = ldd[i-4]*v[i-4] + ld[i-2]*v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]
    i = N-4
    b[i] = ldd[i-4]*v[i-4] + ld[i-2]*v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-3
    b[i] = ldd[i-4]*v[i-4] + ld[i-2]*v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-2
    b[i] = ldd[i-4]*v[i-4] + ld[i-2]*v[i-2] + dd[i]*v[i]
    i = N-1
    b[i] = ldd[i-4]*v[i-4] + ld[i-2]*v[i-2] + dd[i]*v[i]


def CBD_matvec3D(np.ndarray[T, ndim=3] v,
                 np.ndarray[T, ndim=3] b,
                 np.ndarray[real_t, ndim=1] ld,
                 np.ndarray[real_t, ndim=1] ud,
                 np.ndarray[real_t, ndim=1] udd,
                 np.int64_t axis):
    cdef:
        int i, j, k
        int N = udd.shape[0]

    if axis == 0:
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[0, i, j] = ud[0]*v[1, i, j] + udd[0]*v[3, i, j]

        for k in xrange(1, N):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = ld[k-1]* v[k-1, i, j] + ud[k]*v[k+1, i, j] + udd[k]*v[k+3, i, j]

        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[N, i, j] = ld[N-1]* v[N-1, i, j] + ud[N]*v[N+1, i, j]

    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, 0, k] = ud[0]*v[i, 1, k] + udd[0]*v[i, 3, k]

        for i in range(v.shape[0]):
            for j in xrange(1, N):
                for k in range(v.shape[2]):
                    b[i, j, k] = ld[j-1]* v[i, j-1, k] + ud[j]*v[i, j+1, k] + udd[j]*v[i, j+3, k]

        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, N, k] = ld[N-1]* v[i, N-1, k] + ud[N]*v[i, N+1, k]

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, 0] = ud[0]*v[i, j, 1] + udd[0]*v[i, j, 3]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in xrange(1, N):
                    b[i, j, k] = ld[k-1]* v[i, j, k-1] + ud[k]*v[i, j, k+1] + udd[k]*v[i, j, k+3]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, N] = ld[N-1]* v[i, j, N-1] + ud[N]*v[i, j, N+1]


def CBD_matvec(np.ndarray[T, ndim=1] v,
               np.ndarray[T, ndim=1] b,
               np.ndarray[real_t, ndim=1] ld,
               np.ndarray[real_t, ndim=1] ud,
               np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i
        int N = udd.shape[0]

    b[0] = ud[0]*v[1] + udd[0]*v[3]
    for i in xrange(1, N):
        b[i] = ld[i-1]* v[i-1] + ud[i]*v[i+1] + udd[i]*v[i+3]
    i = N
    b[i] = ld[i-1]* v[i-1] + ud[i]*v[i+1]

def CDB_matvec3D(T [:, :, ::1] v,
                 T [:, :, ::1] b,
                 real_t [::1] lld,
                 real_t [::1] ld,
                 real_t [::1] ud,
                 np.int64_t axis):
    cdef:
        int i, j, k
        int N = ud.shape[0]

    if axis == 0:
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[0, i, j] = ud[0]*v[1, i, j]

        for k in xrange(1, 3):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = ld[k-1]*v[k-1, i, j] + ud[k]*v[k+1, i, j]

        for k in xrange(3, N):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = lld[k-3]*v[k-3, i, j] + ld[k-1]* v[k-1, i, j] + ud[k]*v[k+1, i, j]

        for k in xrange(N, N+2):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = lld[k-3]*v[k-3, i, j] + ld[k-1]* v[k-1, i, j]

        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[N+2, i, j] = lld[N-1]* v[N-1, i, j]

    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, 0, k] = ud[0]*v[i, 1, k]

        for i in range(v.shape[0]):
            for j in xrange(1, 3):
                for k in range(v.shape[2]):
                    b[i, j, k] = ld[j-1]*v[i, j-1, k] + ud[j]*v[i, j+1, k]

        for i in range(v.shape[0]):
            for j in xrange(3, N):
                for k in range(v.shape[2]):
                    b[i, j, k] = lld[j-3]*v[i, j-3, k] + ld[j-1]* v[i, j-1, k] + ud[j]*v[i, j+1, k]

        for i in range(v.shape[0]):
            for j in xrange(N, N+2):
                for k in range(v.shape[2]):
                    b[i, j, k] = lld[j-3]*v[i, j-3, k] + ld[j-1]* v[i, j-1, k]

        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, N+2, k] = lld[N-1]* v[i, N-1, k]

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, 0] = ud[0]*v[i, j, 1]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in xrange(1, 3):
                    b[i, j, k] = ld[k-1]*v[i, j, k-1] + ud[k]*v[i, j, k+1]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in xrange(3, N):
                    b[i, j, k] = lld[k-3]*v[i, j, k-3] + ld[k-1]* v[i, j, k-1] + ud[k]*v[i, j, k+1]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in xrange(N, N+2):
                    b[i, j, k] = lld[k-3]*v[i, j, k-3] + ld[k-1]* v[i, j, k-1]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, N+2] = lld[N-1]* v[i, j, N-1]


def BBD_matvec3D(np.ndarray[T, ndim=3] v,
                 np.ndarray[T, ndim=3] b,
                 real_t ld,
                 np.ndarray[real_t, ndim=1] dd,
                 np.ndarray[real_t, ndim=1] ud,
                 np.ndarray[real_t, ndim=1] uud,
                 np.int64_t axis):
    cdef:
        int i, j, k

    if axis == 0:
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                b[0, i, j] = dd[0]*v[0, i, j] + ud[0]*v[2, i, j] + uud[0]*v[4, i, j]
                b[1, i, j] = dd[1]*v[1, i, j] + ud[1]*v[3, i, j] + uud[1]*v[5, i, j]

        for k in range(2, uud.shape[0]):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = ld*v[k-2, i, j] + dd[k]*v[k, i, j] + ud[k]*v[k+2, i, j] + uud[k]*v[k+4, i, j]

        for k in range(uud.shape[0], dd.shape[0]):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    b[k, i, j] = ld*v[k-2, i, j] + dd[k]*v[k, i, j] + ud[k]*v[k+2, i, j]

    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, 0, k] = dd[0]*v[i, 0, k] + ud[0]*v[i, 2, k] + uud[0]*v[i, 4, k]
                b[i, 1, k] = dd[1]*v[i, 1, k] + ud[1]*v[i, 3, k] + uud[1]*v[i, 5, k]

        for i in range(v.shape[0]):
            for j in range(2, uud.shape[0]):
                for k in range(v.shape[2]):
                    b[i, j, k] = ld*v[i, j-2, k] + dd[j]*v[i, j, k] + ud[j]*v[i, j+2, k] + uud[j]*v[i, j+4, k]

        for i in range(v.shape[0]):
            for j in range(uud.shape[0], dd.shape[0]):
                for k in range(v.shape[2]):
                    b[i, j, k] = ld*v[i, j-2, k] + dd[j]*v[i, j, k] + ud[j]*v[i, j+2, k]

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, 0] = dd[0]*v[i, j, 0] + ud[0]*v[i, j, 2] + uud[0]*v[i, j, 4]
                b[i, j, 1] = dd[1]*v[i, j, 1] + ud[1]*v[i, j, 3] + uud[1]*v[i, j, 5]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in range(2, uud.shape[0]):
                    b[i, j, k] = ld*v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2] + uud[k]*v[i, j, k+4]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in range(uud.shape[0], dd.shape[0]):
                    b[i, j, k] = ld*v[i, j, k-2] + dd[k]*v[i, j, k] + ud[k]*v[i, j, k+2]

def Helmholtz_matvec(np.ndarray[T, ndim=1] v,
                     np.ndarray[T, ndim=1] b,
                     real_t alfa,
                     real_t beta,
                     np.ndarray[real_t, ndim=1] dd,
                     np.ndarray[real_t, ndim=1] ud,
                     np.ndarray[real_t, ndim=1] bd):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        int N = dd.shape[0]
        T s1 = 0.0
        T s2 = 0.0
        double pi_half = np.pi/2
        double p

    k = N-1
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*v[k-2]
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - pi_half*beta*v[k-3]

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        if k % 2 == 0:
            s2 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*(v[k-2] + v[k+2]) + p*s2
        else:
            s1 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*(v[k-2] + v[k+2]) + p*s1

    k = 1
    s1 += v[k+2]
    s2 += v[k+1]
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*v[k+2] + ud[k]*alfa*s1
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - pi_half*beta*v[k+1] + ud[k-1]*alfa*s2

def Helmholtz_matvec3D(np.ndarray[T, ndim=3] v,
                       np.ndarray[T, ndim=3] b,
                       real_t alfa,
                       np.ndarray[real_t, ndim=3] beta,
                       np.ndarray[real_t, ndim=1] dd,
                       np.ndarray[real_t, ndim=1] ud,
                       np.ndarray[real_t, ndim=1] bd,
                       int axis):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        int N = dd.shape[0]
        np.ndarray[T, ndim=3] s1 = np.zeros((beta.shape[0], beta.shape[1], beta.shape[2]), dtype=v.dtype)
        np.ndarray[T, ndim=3] s2 = np.zeros((beta.shape[0], beta.shape[1], beta.shape[2]), dtype=v.dtype)
        double pi_half = np.pi/2
        double p

    s1[:] = 0
    s2[:] = 0
    if axis == 0:
        i = N-1

        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                b[i, j, k] = (dd[i]*alfa + bd[i]*beta[0, j, k])*v[i, j, k] - pi_half*beta[0, j, k]*v[i-2, j, k]
                b[i-1, j, k] = (dd[i-1]*alfa + bd[i-1]*beta[0, j, k])*v[i-1, j, k] - pi_half*beta[0, j, k]*v[i-3, j, k]

        for i in range(N-3, 1, -1):
            for j in range(v.shape[1]):
                for k in range(v.shape[2]):
                    p = ud[i]*alfa
                    if i % 2 == 0:
                        s2[0, j, k] += v[i+2, j, k]
                        b[i, j, k] = (dd[i]*alfa + bd[i]*beta[0, j, k])*v[i, j, k] - pi_half*beta[0, j, k]*(v[i-2, j, k] + v[i+2, j, k]) + p*s2[0, j, k]
                    else:
                        s1[0, j, k]+= v[i+2, j, k]
                        b[i, j, k] = (dd[i]*alfa + bd[i]*beta[0, j, k])*v[i, j, k] - pi_half*beta[0, j, k]*(v[i-2, j, k] + v[i+2, j, k]) + p*s1[0, j, k]

        i = 1
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                s1[0, j, k]+= v[i+2, j, k]
                s2[0, j, k]+= v[i+1, j, k]
                b[i, j, k] = (dd[i]*alfa + bd[i]*beta[0, j, k])*v[i, j, k] - pi_half*beta[0, j, k]*v[i+2, j, k] + ud[i]*alfa*s1[0, j, k]
                b[i-1, j, k] = (dd[i-1]*alfa + bd[i-1]*beta[0, j, k])*v[i-1, j, k] - pi_half*beta[0, j, k]*v[i+1, j, k] + ud[i-1]*alfa*s2[0, j, k]

    elif axis == 1:
        j = N-1
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                b[i, j, k] = (dd[j]*alfa + bd[j]*beta[i, 0, k])*v[i, j, k] - pi_half*beta[i, 0, k]*v[i, j-2, k]
                b[i, j-1, k] = (dd[j-1]*alfa + bd[j-1]*beta[i, 0, k])*v[i, j-1, k] - pi_half*beta[i, 0, k]*v[i, j-3, k]

        for i in range(v.shape[0]):
            for j in range(N-3, 1, -1):
                for k in range(v.shape[2]):
                    p = ud[j]*alfa
                    if j % 2 == 0:
                        s2[i, 0, k]+= v[i, j+2, k]
                        b[i, j, k] = (dd[j]*alfa + bd[j]*beta[i, 0, k])*v[i, j, k] - pi_half*beta[i, 0, k]*(v[i, j-2, k] + v[i, j+2, k]) + p*s2[i, 0, k]
                    else:
                        s1[i, 0, k]+= v[i, j+2, k]
                        b[i, j, k] = (dd[j]*alfa + bd[j]*beta[i, 0, k])*v[i, j, k] - pi_half*beta[i, 0, k]*(v[i, j-2, k] + v[i, j+2, k]) + p*s1[i, 0, k]

        j = 1
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                s1[i, 0, k]+= v[i, j+2, k]
                s2[i, 0, k]+= v[i, j+1, k]
                b[i, j, k] = (dd[j]*alfa + bd[j]*beta[i, 0, k])*v[i, j, k] - pi_half*beta[i, 0, k]*v[i, j+2, k] + ud[j]*alfa*s1[i, 0, k]
                b[i, j-1, k] = (dd[j-1]*alfa + bd[j-1]*beta[i, 0, k])*v[i, j-1, k] - pi_half*beta[i, 0, k]*v[i, j+1, k] + ud[j-1]*alfa*s2[i, 0, k]

    elif axis == 2:
        k = N-1
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                b[i, j, k] = (dd[k]*alfa + bd[k]*beta[i, j, 0])*v[i, j, k] - pi_half*beta[i, j, 0]*v[i, j, k-2]
                b[i, j, k-1] = (dd[k-1]*alfa + bd[k-1]*beta[i, j, 0])*v[i, j, k-1] - pi_half*beta[i, j, 0]*v[i, j, k-3]

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in range(N-3, 1, -1):
                    p = ud[k]*alfa
                    if k % 2 == 0:
                        s2[i, j, 0]+= v[i, j, k+2]
                        b[i, j, k] = (dd[k]*alfa + bd[k]*beta[i, j, 0])*v[i, j, k] - pi_half*beta[i, j, 0]*(v[i, j, k-2] + v[i, j, k+2]) + p*s2[i, j, 0]
                    else:
                        s1[i, j, 0]+= v[i, j, k+2]
                        b[i, j, k] = (dd[k]*alfa + bd[k]*beta[i, j, 0])*v[i, j, k] - pi_half*beta[i, j, 0]*(v[i, j, k-2] + v[i, j, k+2]) + p*s1[i, j, 0]

        k = 1
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                s1[i, j, 0]+= v[i, j, k+2]
                s2[i, j, 0]+= v[i, j, k+1]
                b[i, j, k] = (dd[k]*alfa + bd[k]*beta[i, j, 0])*v[i, j, k] - pi_half*beta[i, j, 0]*v[i, j, k+2] + ud[k]*alfa*s1[i, j, 0]
                b[i, j, k-1] = (dd[k-1]*alfa + bd[k-1]*beta[i, j, 0])*v[i, j, k-1] - pi_half*beta[i, j, 0]*v[i, j, k+1] + ud[k-1]*alfa*s2[i, j, 0]


def Helmholtz_matvec2D(np.ndarray[T, ndim=2] v,
                     np.ndarray[T, ndim=2] b,
                     real_t alfa,
                     np.ndarray[real_t, ndim=2] beta,
                     np.ndarray[real_t, ndim=1] dd,
                     np.ndarray[real_t, ndim=1] ud,
                     np.ndarray[real_t, ndim=1] bd,
                     int axis):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        int N = dd.shape[0]
        np.ndarray[T, ndim=2] s1 = np.zeros((beta.shape[0], beta.shape[1]), dtype=v.dtype)
        np.ndarray[T, ndim=2] s2 = np.zeros((beta.shape[0], beta.shape[1]), dtype=v.dtype)
        double pi_half = np.pi/2
        double p

    s1[:] = 0
    s2[:] = 0
    if axis == 0:
        i = N-1
        for j in range(v.shape[1]):
            b[i, j] = (dd[i]*alfa + bd[i]*beta[0, j])*v[i, j] - pi_half*beta[0, j]*v[i-2, j]
            b[i-1, j] = (dd[i-1]*alfa + bd[i-1]*beta[0, j])*v[i-1, j] - pi_half*beta[0, j]*v[i-3, j]

        for i in range(N-3, 1, -1):
            for j in range(v.shape[1]):
                p = ud[i]*alfa
                if i % 2 == 0:
                    s2[0, j] += v[i+2, j]
                    b[i, j] = (dd[i]*alfa + bd[i]*beta[0, j])*v[i, j] - pi_half*beta[0, j]*(v[i-2, j] + v[i+2, j]) + p*s2[0, j]
                else:
                    s1[0, j] += v[i+2, j]
                    b[i, j] = (dd[i]*alfa + bd[i]*beta[0, j])*v[i, j] - pi_half*beta[0, j]*(v[i-2, j] + v[i+2, j]) + p*s1[0, j]

        i = 1
        for j in range(v.shape[1]):
            s1[0, j] += v[i+2, j]
            s2[0, j] += v[i+1, j]
            b[i, j] = (dd[i]*alfa + bd[i]*beta[0, j])*v[i, j] - pi_half*beta[0, j]*v[i+2, j] + ud[i]*alfa*s1[0, j]
            b[i-1, j] = (dd[i-1]*alfa + bd[i-1]*beta[0, j])*v[i-1, j] - pi_half*beta[0, j]*v[i+1, j] + ud[i-1]*alfa*s2[0, j]

    elif axis == 1:
        j = N-1
        for i in range(v.shape[0]):
            b[i, j] = (dd[j]*alfa + bd[j]*beta[i, 0])*v[i, j] - pi_half*beta[i, 0]*v[i, j-2]
            b[i, j-1] = (dd[j-1]*alfa + bd[j-1]*beta[i, 0])*v[i, j-1] - pi_half*beta[i, 0]*v[i, j-3]

        for i in range(v.shape[0]):
            for j in range(N-3, 1, -1):
                p = ud[j]*alfa
                if j % 2 == 0:
                    s2[i, 0] += v[i, j+2]
                    b[i, j] = (dd[j]*alfa + bd[j]*beta[i, 0])*v[i, j] - pi_half*beta[i, 0]*(v[i, j-2] + v[i, j+2]) + p*s2[i, 0]
                else:
                    s1[i, 0] += v[i, j+2]
                    b[i, j] = (dd[j]*alfa + bd[j]*beta[i, 0])*v[i, j] - pi_half*beta[i, 0]*(v[i, j-2] + v[i, j+2]) + p*s1[i, 0]

        j = 1
        for i in range(v.shape[0]):
            s1[i, 0] += v[i, j+2]
            s2[i, 0] += v[i, j+1]
            b[i, j] = (dd[j]*alfa + bd[j]*beta[i, 0])*v[i, j] - pi_half*beta[i, 0]*v[i, j+2] + ud[j]*alfa*s1[i, 0]
            b[i, j-1] = (dd[j-1]*alfa + bd[j-1]*beta[i, 0])*v[i, j-1] - pi_half*beta[i, 0]*v[i, j+1] + ud[j-1]*alfa*s2[i, 0]


def Biharmonic_matvec(np.ndarray[T, ndim=1] v,
                      np.ndarray[T, ndim=1] b,
                      np.float_t a0,
                      np.float_t alfa,
                      np.float_t beta,
                      # 3 upper diagonals of SBB
                      np.ndarray[real_t, ndim=1] sii,
                      np.ndarray[real_t, ndim=1] siu,
                      np.ndarray[real_t, ndim=1] siuu,
                      # All 3 diagonals of ABB
                      np.ndarray[real_t, ndim=1] ail,
                      np.ndarray[real_t, ndim=1] aii,
                      np.ndarray[real_t, ndim=1] aiu,
                      # All 5 diagonals of BBB
                      np.ndarray[real_t, ndim=1] bill,
                      np.ndarray[real_t, ndim=1] bil,
                      np.ndarray[real_t, ndim=1] bii,
                      np.ndarray[real_t, ndim=1] biu,
                      np.ndarray[real_t, ndim=1] biuu):

    cdef:
        int i, j, k
        int N = sii.shape[0]
        np.ndarray[double, ndim=1] ldd, ld, dd, ud, udd
        double p, r
        T d, s1, s2, o1, o2

    dd = np.zeros(N)
    ld = np.zeros(N)
    ldd = np.zeros(N)
    ud = np.zeros(N)
    udd = np.zeros(N)

    for i in xrange(N):
        dd[i] = a0*sii[i] + alfa*aii[i] + beta*bii[i]

    for i in xrange(ail.shape[0]):
        ld[i] = alfa*ail[i] + beta*bil[i]

    for i in xrange(bill.shape[0]):
        ldd[i] = beta*bill[i]

    for i in xrange(siu.shape[0]):
        ud[i] = a0*siu[i] + alfa*aiu[i] + beta*biu[i]

    for i in xrange(siuu.shape[0]):
        udd[i] = a0*siuu[i] + beta*biuu[i]

    i = N-1
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-2
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-3
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-4
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-5
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]
    i = N-6
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]

    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    for k in xrange(N-7, -1, -1):
        j = k+6
        p = k*sii[k]/(k+1.)
        r = 24*(k+1)*(k+2)*np.pi
        d = v[j]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k] = (p*s1 + r*s2)*a0
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k] = (p*o1 + r*o2)*a0

        if k > 3:
            b[k] += ldd[k-4]*v[k-4]+ ld[k-2]* v[k-2] + dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]
        elif k > 1:
            b[k] += ld[k-2]* v[k-2] + dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]
        else:
            b[k] += dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]


def Biharmonic_matvec3D(np.ndarray[T, ndim=3] v,
                        np.ndarray[T, ndim=3] b,
                        np.float_t a0,
                        np.ndarray[real_t, ndim=3] alfa,
                        np.ndarray[real_t, ndim=3] beta,
                        # 3 upper diagonals of SBB
                        np.ndarray[real_t, ndim=1] sii,
                        np.ndarray[real_t, ndim=1] siu,
                        np.ndarray[real_t, ndim=1] siuu,
                        # All 3 diagonals of ABB
                        np.ndarray[real_t, ndim=1] ail,
                        np.ndarray[real_t, ndim=1] aii,
                        np.ndarray[real_t, ndim=1] aiu,
                        # All 5 diagonals of BBB
                        np.ndarray[real_t, ndim=1] bill,
                        np.ndarray[real_t, ndim=1] bil,
                        np.ndarray[real_t, ndim=1] bii,
                        np.ndarray[real_t, ndim=1] biu,
                        np.ndarray[real_t, ndim=1] biuu,
                        int axis):
    cdef:
        int i, j, k, ii
        np.ndarray[double, ndim=1] ldd, ld, dd, ud, udd
        double p, r
        T d, s1, s2, o1, o2
        int N = sii.shape[0]

    ldd = np.zeros(N)
    ld = np.zeros(N)
    dd = np.zeros(N)
    ud = np.zeros(N)
    udd = np.zeros(N)

    if axis == 0:

        #for i in range(v.shape[1]):
        #    for j in range(v.shape[2]):
        #        Biharmonic_matvec(v[:, i, j], b[:, i, j], a0, alfa[0, i, j],
        #                          beta[0, i, j], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)


        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                for i in range(N):
                    dd[i] = a0*sii[i] + alfa[0,j,k]*aii[i] + beta[0,j,k]*bii[i]

                for i in xrange(ail.shape[0]):
                    ld[i] = alfa[0,j,k]*ail[i] + beta[0,j,k]*bil[i]

                for i in xrange(bill.shape[0]):
                    ldd[i] = beta[0,j,k]*bill[i]

                for i in xrange(siu.shape[0]):
                    ud[i] = a0*siu[i] + alfa[0,j,k]*aiu[i] + beta[0,j,k]*biu[i]

                for i in xrange(siuu.shape[0]):
                    udd[i] = a0*siuu[i] + beta[0,j,k]*biuu[i]

                i = N-1
                b[i, j, k] = ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k]
                i = N-2
                b[i, j, k] = ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k]
                i = N-3
                b[i, j, k] = ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k]
                i = N-4
                b[i, j, k] = ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k]
                i = N-5
                b[i, j, k] = ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k] + udd[i]*v[i+4, j, k]
                i = N-6
                b[i, j, k] = ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k] + udd[i]*v[i+4, j, k]

                s1 = 0.0
                s2 = 0.0
                o1 = 0.0
                o2 = 0.0
                for i in xrange(N-7, -1, -1):
                    ii = i+6
                    p = i*sii[i]/(i+1.)
                    r = 24*(i+1)*(i+2)*M_PI
                    d = v[ii, j, k]/(ii+3.)
                    if i % 2 == 0:
                        s1 += d
                        s2 += (ii+2)*(ii+2)*d
                        b[i,j,k] = (p*s1 + r*s2)*a0
                    else:
                        o1 += d
                        o2 += (ii+2)*(ii+2)*d
                        b[i,j,k] = (p*o1 + r*o2)*a0

                    if i > 3:
                        b[i,j,k] += ldd[i-4]*v[i-4, j, k]+ ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k] + udd[i]*v[i+4, j, k]
                    elif i > 1:
                        b[i, j, k] += ld[i-2]* v[i-2, j, k] + dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k] + udd[i]*v[i+4, j, k]
                    else:
                        b[i, j, k] += dd[i]*v[i, j, k] + ud[i]*v[i+2, j, k] + udd[i]*v[i+4, j, k]

    elif axis == 1:
        for i in range(v.shape[0]):
            for j in range(v.shape[2]):
                Biharmonic_matvec(v[i, :, j], b[i, :, j], a0, alfa[i, 0, j],
                                  beta[i, 0, j], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                Biharmonic_matvec(v[i, j], b[i, j], a0, alfa[i, j, 0],
                                  beta[i, j, 0], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)

def Biharmonic_matvec2D(np.ndarray[T, ndim=2] v,
                        np.ndarray[T, ndim=2] b,
                        np.float_t a0,
                        np.ndarray[real_t, ndim=2] alfa,
                        np.ndarray[real_t, ndim=2] beta,
                        # 3 upper diagonals of SBB
                        np.ndarray[real_t, ndim=1] sii,
                        np.ndarray[real_t, ndim=1] siu,
                        np.ndarray[real_t, ndim=1] siuu,
                        # All 3 diagonals of ABB
                        np.ndarray[real_t, ndim=1] ail,
                        np.ndarray[real_t, ndim=1] aii,
                        np.ndarray[real_t, ndim=1] aiu,
                        # All 5 diagonals of BBB
                        np.ndarray[real_t, ndim=1] bill,
                        np.ndarray[real_t, ndim=1] bil,
                        np.ndarray[real_t, ndim=1] bii,
                        np.ndarray[real_t, ndim=1] biu,
                        np.ndarray[real_t, ndim=1] biuu,
                        int axis):
    cdef:
        int i, j

    if axis == 0:
        for i in range(v.shape[1]):
            Biharmonic_matvec(v[:, i], b[:, i], a0, alfa[0, i],
                              beta[0, i], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)
    elif axis == 1:
        for i in range(v.shape[0]):
            Biharmonic_matvec(v[i], b[i], a0, alfa[i, 0],
                              beta[i, 0], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)

cdef void Biharmonic_matvec_ptr(T* v,
                                T* b,
                                real_t a0,
                                real_t alfa,
                                real_t beta,
                                # 3 upper diagonals of SBB
                                real_t* sii,
                                real_t* siu,
                                real_t* siuu,
                                # All 3 diagonals of ABB
                                real_t* ail,
                                real_t* aii,
                                real_t* aiu,
                                # All 5 diagonals of BBB
                                real_t* bill,
                                real_t* bil,
                                real_t* bii,
                                real_t* biu,
                                real_t* biuu,
                                int N,
                                int st):

    cdef:
        int i, j, k
        vector[double] ldd, ld, dd, ud, udd
        double p, r
        T d, s1, s2, o1, o2

    dd.resize(N)
    ld.resize(N)
    ldd.resize(N)
    ud.resize(N)
    udd.resize(N)

    for i in xrange(N):
        dd[i] = a0*sii[i] + alfa*aii[i] + beta*bii[i]

    for i in xrange(N-2):
        ld[i] = alfa*ail[i] + beta*bil[i]

    for i in xrange(N-4):
        ldd[i] = beta*bill[i]

    for i in xrange(N-2):
        ud[i] = a0*siu[i] + alfa*aiu[i] + beta*biu[i]

    for i in xrange(N-4):
        udd[i] = a0*siuu[i] + beta*biuu[i]

    i = N-1
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st]
    i = N-2
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st]
    i = N-3
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = N-4
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = N-5
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st] + udd[i]*v[(i+4)*st]
    i = N-6
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st] + udd[i]*v[(i+4)*st]

    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    for k in xrange(N-7, -1, -1):
        j = k+6
        p = k*sii[k]/(k+1.)
        r = 24*(k+1)*(k+2)*M_PI
        d = v[j*st]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k*st] = (p*s1 + r*s2)*a0
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k*st] = (p*o1 + r*o2)*a0

        if k > 3:
            b[k*st] += ldd[k-4]*v[(k-4)*st]+ ld[k-2]* v[(k-2)*st] + dd[k]*v[k*st] + ud[k]*v[(k+2)*st] + udd[k]*v[(k+4)*st]
        elif k > 1:
            b[k*st] += ld[k-2]* v[(k-2)*st] + dd[k]*v[k*st] + ud[k]*v[(k+2)*st] + udd[k]*v[(k+4)*st]
        else:
            b[k*st] += dd[k]*v[k*st] + ud[k]*v[(k+2)*st] + udd[k]*v[(k+4)*st]

def Biharmonic_matvec3D_ptr(T[:, :, ::1] v,
                            T[:, :, ::1] b,
                            real_t a0,
                            real_t[:, :, ::1] alfa,
                            real_t[:, :, ::1] beta,
                            # 3 upper diagonals of SBB
                            real_t[::1] sii,
                            real_t[::1] siu,
                            real_t[::1] siuu,
                            # All 3 diagonals of ABB
                            real_t[::1] ail,
                            real_t[::1] aii,
                            real_t[::1] aiu,
                            # All 5 diagonals of BBB
                            real_t[::1] bill,
                            real_t[::1] bil,
                            real_t[::1] bii,
                            real_t[::1] biu,
                            real_t[::1] biuu,
                            int axis):
    cdef:
        int i, j, k, strides
        int N = sii.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                Biharmonic_matvec_ptr(v_ptr, b_ptr, a0, alfa[0, j, k],
                                      beta[0, j, k], &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0],
                                      &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                Biharmonic_matvec_ptr(v_ptr, b_ptr, a0, alfa[i, 0, k],
                                      beta[i, 0, k], &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0],
                                      &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], N, strides)

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                Biharmonic_matvec_ptr(v_ptr, b_ptr, a0, alfa[i, j, 0],
                                      beta[i, j, 0], &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0],
                                      &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], N, strides)

def Biharmonic_matvec2D_ptr(T[:, ::1] v,
                            T[:, ::1] b,
                            real_t a0,
                            real_t[:, ::1] alfa,
                            real_t[:, ::1] beta,
                            # 3 upper diagonals of SBB
                            real_t[::1] sii,
                            real_t[::1] siu,
                            real_t[::1] siuu,
                            # All 3 diagonals of ABB
                            real_t[::1] ail,
                            real_t[::1] aii,
                            real_t[::1] aiu,
                            # All 5 diagonals of BBB
                            real_t[::1] bill,
                            real_t[::1] bil,
                            real_t[::1] bii,
                            real_t[::1] biu,
                            real_t[::1] biuu,
                            int axis):
    cdef:
        int i, j, k, strides
        int N = sii.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            Biharmonic_matvec_ptr(v_ptr, b_ptr, a0, alfa[0, j],
                                  beta[0, j], &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0],
                                  &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            Biharmonic_matvec_ptr(v_ptr, b_ptr, a0, alfa[i, 0],
                                  beta[i, 0], &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0],
                                  &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], N, strides)
