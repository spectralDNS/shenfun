#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport cython
cimport numpy as np
from libcpp.vector cimport vector

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t

ctypedef fused T:
    real_t
    complex_t

def CDNmat_matvec(np.ndarray[real_t, ndim=1] ud,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[T, ndim=3] v,
                  np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = ud[0]*v[1, j, k]
            b[N-1, j, k] = ld[N-2]*v[N-2, j, k]

    for i in xrange(1, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-1]*v[i-1, j, k]

def BDNmat_matvec(real_t ud,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[real_t, ndim=1] dd,
                  np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2

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

def CDDmat_matvec(np.ndarray[real_t, ndim=1] ud,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[T, ndim=3] v,
                  np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = ud[0]*v[1, j, k]
            b[N-1, j, k] = ld[N-2]*v[N-2, j, k]

    for i in xrange(1, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-1]*v[i-1, j, k]

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
                    np.ndarray[real_t, ndim=1] dd):
    cdef:
        int i, j, k, jj
        double p, r, d2
        T d
        np.ndarray[T, ndim=2] s1 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        np.ndarray[T, ndim=2] s2 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        np.ndarray[T, ndim=2] o1 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        np.ndarray[T, ndim=2] o2 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        int N = v.shape[0]-4

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #SBBmat_matvec(v[:, i, j], bb[:, i, j], dd)

    k = N-1
    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            b[k, i, j] = dd[k]*v[k, i, j]
            b[k-1, i, j] = dd[k-1]*v[k-1, i, j]

    for k in xrange(N-3, -1, -1):
        jj = k+2
        p = k*dd[k]/(k+1.)
        r = 24*(k+1)*(k+2)*np.pi
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
        p = 4*(k+1)*pi
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
                         real_t[::1] ud):
    cdef:
        np.intp_t i, j, k
        np.intp_t N = dd.shape[0]

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #Tridiagonal_matvec(v[:, i, j], b[:, i, j], ld, dd, ud)

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
                    np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i, j, k
        int N = dd.shape[0]

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #Pentadiagonal_matvec(v[:, i, j], b[:, i, j], ld, dd, ud)

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
                 np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i, j, k
        int N = udd.shape[0]

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
                 real_t [::1] ud):
    cdef:
        int i, j, k
        int N = ud.shape[0]

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

def BBD_matvec3D(np.ndarray[T, ndim=3] v,
                 np.ndarray[T, ndim=3] b,
                 real_t ld,
                 np.ndarray[real_t, ndim=1] dd,
                 np.ndarray[real_t, ndim=1] ud,
                 np.ndarray[real_t, ndim=1] uud):
    cdef:
        int i, j, k

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
