#cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from libc.math cimport M_PI, M_PI_2
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

def CDN_matvec1D_ptr(T[::1] v,
                     T[::1] b,
                     real_t[::1] ld,
                     real_t[::1] ud):
    cdef:
        T* v_ptr = &v[0]
        T* b_ptr = &b[0]
        int N = v.shape[0]-2
    CDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, 1)

cdef void CDN_matvec_ptr(T* v,
                         T* b,
                         real_t* ld,
                         real_t* ud,
                         int N,
                         int st):
    cdef:
        int i

    b[0] = ud[0]*v[st]
    b[(N-1)*st] = ld[N-2]*v[(N-2)*st]
    for i in xrange(1, N-1):
        b[i*st] = ud[i]*v[(i+1)*st] + ld[i-1]*v[(i-1)*st]

def CDN_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t[::1] ld,
                     real_t[::1] ud,
                     int axis):
    cdef:
        int i, j, strides
        int N = v.shape[axis]-2
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            CDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            CDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)

def CDN_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t[::1] ld,
                     real_t[::1] ud,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = v.shape[axis]-2
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                CDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                CDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                CDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)

cdef void BDN_matvec_ptr(T* v,
                         T* b,
                         real_t* ld,
                         real_t* dd,
                         real_t ud,
                         int N,
                         int st):
    cdef:
        int i, j, k

    b[0] = ud*v[2*st] + dd[0]*v[0]
    b[st] = ud*v[3*st] + dd[1]*v[st]
    b[(N-2)*st] = ld[N-4]*v[(N-4)*st] + dd[N-2]*v[(N-2)*st]
    b[(N-1)*st] = ld[N-3]*v[(N-3)*st] + dd[N-1]*v[(N-1)*st]
    for i in xrange(2, N-2):
        b[i*st] = ud*v[(i+2)*st] + dd[i]*v[i*st] + ld[i-2]*v[(i-2)*st]

def BDN_matvec1D_ptr(T[::1] v,
                     T[::1] b,
                     real_t[::1] ld,
                     real_t[::1] dd,
                     real_t ud):
    cdef:
        T* v_ptr = &v[0]
        T* b_ptr = &b[0]
        int N = v.shape[0]-2
    BDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], ud, N, 1)

def BDN_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t[::1] ld,
                     real_t[::1] dd,
                     real_t ud,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = v.shape[axis]-2
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                BDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], ud, N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                BDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], ud, N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                BDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], ud, N, strides)

def BDN_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t[::1] ld,
                     real_t[::1] dd,
                     real_t ud,
                     int axis):
    cdef:
        int i, j, strides
        int N = v.shape[axis]-2
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            BDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], ud, N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            BDN_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], ud, N, strides)

cdef void CDDmat_matvec_ptr(T* v,
                            T* b,
                            real_t* ld,
                            real_t* ud,
                            int N,
                            int st):
    cdef:
        int i

    b[0] = ud[0]*v[st]
    b[(N-1)*st] = ld[N-2]*v[(N-2)*st]
    for i in xrange(1, N-1):
        b[i*st] = ud[i]*v[(i+1)*st] + ld[i-1]*v[(i-1)*st]

def CDD_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t[::1] ld,
                     real_t[::1] ud,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = ud.shape[0]+1
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                CDDmat_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)
    elif axis == 1:
       for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                CDDmat_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                CDDmat_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)

def CDD_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t[::1] ld,
                     real_t[::1] ud,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = ud.shape[0]+1
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            CDDmat_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)
    elif axis == 1:
       for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            CDDmat_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, strides)

def CDD_matvec1D_ptr(T[::1] v,
                     T[::1] b,
                     real_t[::1] ld,
                     real_t[::1] ud):
    cdef:
        int N = ud.shape[0]+1
        T* v_ptr = &v[0]
        T* b_ptr = &b[0]
    CDDmat_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], N, 1)

def SBB_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t[::1] dd,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                SBB_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)
    elif axis == 1:
       for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                SBB_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                SBB_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)

def SBB_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t[::1] dd,
                     int axis):
    cdef:
        int i, j, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            SBB_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)
    elif axis == 1:
       for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            SBB_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)

cdef void SBB_matvec_ptr(T* v,
                         T* b,
                         real_t* dd,
                         int N,
                         int st):
    cdef:
        int i, j, k
        double p, r
        T d, s1, s2, o1, o2

    j = N-1
    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    b[j*st] = dd[j]*v[j*st]
    b[(j-1)*st] = dd[j-1]*v[(j-1)*st]
    for k in range(N-3, -1, -1):
        j = k+2
        p = k*dd[k]/(k+1)
        r = 24*(k+1)*(k+2)*M_PI
        d = v[j*st]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k*st] = dd[k]*v[k*st] + p*s1 + r*s2
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k*st] = dd[k]*v[k*st] + p*o1 + r*o2

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

def ADD_matvec(np.ndarray[T, ndim=1] v,
               np.ndarray[T, ndim=1] b,
               np.ndarray[real_t, ndim=1] dd):
    cdef:
        T* v_ptr=&v[0]
        T* b_ptr=&b[0]
    ADD_matvec_ptr(v_ptr, b_ptr, &dd[0], dd.shape[0], 1)

cdef void ADD_matvec_ptr(T* v,
                         T* b,
                         real_t* dd,
                         int N,
                         int st):
    cdef:
        int i, j, k
        double p
        T s1 = 0.0
        T s2 = 0.0
        T d

    k = N-1
    b[k*st] = dd[k]*v[k*st]
    b[(k-1)*st] = dd[k-1]*v[(k-1)*st]
    for k in range(N-3, -1, -1):
        j = k+2
        p = -4*(k+1)*M_PI
        if j % 2 == 0:
            s1 += v[j*st]
            b[k*st] = dd[k]*v[k*st] + p*s1
        else:
            s2 += v[j*st]
            b[k*st] = dd[k]*v[k*st] + p*s2

def ADD_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t[::1] dd,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                ADD_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                ADD_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                ADD_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)

def ADD_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t[::1] dd,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            ADD_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            ADD_matvec_ptr(v_ptr, b_ptr, &dd[0], N, strides)

cdef void Tridiagonal_matvec_ptr(T* v,
                                 T* b,
                                 real_t* ld,
                                 real_t* dd,
                                 real_t* ud,
                                 int N,
                                 int st):
    cdef:
        int i

    b[0] = dd[0]*v[0] + ud[0]*v[2*st]
    b[st] = dd[1]*v[st] + ud[1]*v[3*st]
    for i in xrange(2, N-2):
        b[i*st] = ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = N-2
    b[i*st] = ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st]
    i = N-1
    b[i*st] = ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st]

def Tridiagonal_matvec2D_ptr(T[:, ::1] v,
                             T[:, ::1] b,
                             real_t[::1] ld,
                             real_t[::1] dd,
                             real_t[::1] ud,
                             int axis):
    cdef:
        int i, j, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            Tridiagonal_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], &ud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            Tridiagonal_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], &ud[0], N, strides)

def Tridiagonal_matvec3D_ptr(T[:, :, ::1] v,
                             T[:, :, ::1] b,
                             real_t[::1] ld,
                             real_t[::1] dd,
                             real_t[::1] ud,
                             int axis):
    cdef:
        int i, j, k, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                Tridiagonal_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], &ud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                Tridiagonal_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], &ud[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                Tridiagonal_matvec_ptr(v_ptr, b_ptr, &ld[0], &dd[0], &ud[0], N, strides)

def Tridiagonal_matvec(T[::1] v,
                       T[::1] b,
                       real_t[::1] ld,
                       real_t[::1] dd,
                       real_t[::1] ud):
    cdef:
        np.intp_t i
        np.intp_t N = dd.shape[0]

    b[0] = dd[0]*v[0] + ud[0]*v[2]
    b[1] = dd[1]*v[1] + ud[1]*v[3]
    for i in xrange(2, N-2):
        b[i] = ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-2
    b[i] = ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-1
    b[i] = ld[i-2]* v[i-2] + dd[i]*v[i]

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

cdef void Pentadiagonal_matvec_ptr(T* v,
                                   T* b,
                                   real_t* ldd,
                                   real_t* ld,
                                   real_t* dd,
                                   real_t* ud,
                                   real_t* udd,
                                   int N,
                                   int st):
    cdef:
        int i

    b[0] = dd[0]*v[0] + ud[0]*v[2*st] + udd[0]*v[4*st]
    b[1*st] = dd[1]*v[1*st] + ud[1]*v[3*st] + udd[1]*v[5*st]
    b[2*st] = ld[0]*v[0] + dd[2]*v[2*st] + ud[2]*v[4*st] + udd[2]*v[6*st]
    b[3*st] = ld[1]*v[1*st] + dd[3]*v[3*st] + ud[3]*v[5*st] + udd[3]*v[7*st]
    for i in xrange(4, N-4):
        b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st] + udd[i]*v[(i+4)*st]
    i = N-4
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = N-3
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = N-2
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st]
    i = N-1
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st]

def Pentadiagonal_matvec3D_ptr(T[:, :, ::1] v,
                               T[:, :, ::1] b,
                               real_t[::1] ldd,
                               real_t[::1] ld,
                               real_t[::1] dd,
                               real_t[::1] ud,
                               real_t[::1] udd,
                               np.int64_t axis):
    cdef:
        int i, j, k, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                Pentadiagonal_matvec_ptr(v_ptr, b_ptr, &ldd[0], &ld[0], &dd[0],
                                         &ud[0], &udd[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                Pentadiagonal_matvec_ptr(v_ptr, b_ptr, &ldd[0], &ld[0], &dd[0],
                                         &ud[0], &udd[0], N, strides)

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                Pentadiagonal_matvec_ptr(v_ptr, b_ptr, &ldd[0], &ld[0], &dd[0],
                                         &ud[0], &udd[0], N, strides)

def Pentadiagonal_matvec2D_ptr(T[:, ::1] v,
                               T[:, ::1] b,
                               real_t[::1] ldd,
                               real_t[::1] ld,
                               real_t[::1] dd,
                               real_t[::1] ud,
                               real_t[::1] udd,
                               int axis):
    cdef:
        int i, j, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            Pentadiagonal_matvec_ptr(v_ptr, b_ptr, &ldd[0], &ld[0], &dd[0],
                                     &ud[0], &udd[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            Pentadiagonal_matvec_ptr(v_ptr, b_ptr, &ldd[0], &ld[0], &dd[0],
                                     &ud[0], &udd[0], N, strides)

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

cdef void CBD_matvec_ptr(T* v,
                         T* b,
                         real_t* ld,
                         real_t* ud,
                         real_t* udd,
                         int N,
                         int st):
    cdef:
        int i

    b[0] = ud[0]*v[1*st] + udd[0]*v[3*st]
    for i in xrange(1, N):
        b[i*st] = ld[i-1]*v[(i-1)*st] + ud[i]*v[(i+1)*st] + udd[i]*v[(i+3)*st]
    i = N
    b[i*st] = ld[i-1]*v[(i-1)*st] + ud[i]*v[(i+1)*st]

def CBD_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t [::1] ld,
                     real_t [::1] ud,
                     real_t [::1] udd,
                     int axis):
    cdef:
        int i, j, strides
        int N = udd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            CBD_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], &udd[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            CBD_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], &udd[0], N, strides)

def CBD_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t [::1] ld,
                     real_t [::1] ud,
                     real_t [::1] udd,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = udd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                CBD_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], &udd[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                CBD_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], &udd[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                CBD_matvec_ptr(v_ptr, b_ptr, &ld[0], &ud[0], &udd[0], N, strides)

cdef void CDB_matvec_ptr(T* v,
                         T* b,
                         real_t* lld,
                         real_t* ld,
                         real_t* ud,
                         int N,
                         int st):
    cdef:
        int i, j, k

    b[0] = ud[0]*v[st]
    for k in xrange(1, 3):
        b[k*st] = ld[k-1]*v[(k-1)*st] + ud[k]*v[(k+1)*st]
    for k in xrange(3, N):
        b[k*st] = lld[k-3]*v[(k-3)*st] + ld[k-1]*v[(k-1)*st] + ud[k]*v[(k+1)*st]
    for k in xrange(N, N+2):
        b[k*st] = lld[k-3]*v[(k-3)*st] + ld[k-1]* v[(k-1)*st]
    b[(N+2)*st] = lld[N-1]*v[(N-1)*st]

def CDB_matvec(T[::1] v,
               T[::1] b,
               real_t [::1] lld,
               real_t [::1] ld,
               real_t [::1] ud):
    cdef:
        int N = ud.shape[0]
        T* v_ptr = &v[0]
        T* b_ptr = &b[0]
    CDB_matvec_ptr(v_ptr, b_ptr, &lld[0], &ld[0], &ud[0], N, 1)

def CDB_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t[::1] lld,
                     real_t[::1] ld,
                     real_t[::1] ud,
                     int axis):
    cdef:
        int i, j, strides
        int N = ud.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            CDB_matvec_ptr(v_ptr, b_ptr, &lld[0], &ld[0], &ud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            CDB_matvec_ptr(v_ptr, b_ptr, &lld[0], &ld[0], &ud[0], N, strides)

def CDB_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t[::1] lld,
                     real_t[::1] ld,
                     real_t[::1] ud,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = ud.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                CDB_matvec_ptr(v_ptr, b_ptr, &lld[0], &ld[0], &ud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                CDB_matvec_ptr(v_ptr, b_ptr, &lld[0], &ld[0], &ud[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                CDB_matvec_ptr(v_ptr, b_ptr, &lld[0], &ld[0], &ud[0], N, strides)

cdef void BBD_matvec_ptr(T* v,
                         T* b,
                         real_t ld,
                         real_t* dd,
                         real_t* ud,
                         real_t* uud,
                         int N,
                         int st):
    cdef:
        int i

    b[0] = dd[0]*v[0] + ud[0]*v[2*st] + uud[0]*v[4*st]
    b[st] = dd[1]*v[st] + ud[1]*v[3*st] + uud[1]*v[5*st]

    for k in range(2, N):
        b[k*st] = ld*v[(k-2)*st] + dd[k]*v[k*st] + ud[k]*v[(k+2)*st] + uud[k]*v[(k+4)*st]

    for k in range(N, N+2):
        b[k*st] = ld*v[(k-2)*st] + dd[k]*v[k*st] + ud[k]*v[(k+2)*st]

def BBD_matvec3D_ptr(T[:, :, ::1] v,
                     T[:, :, ::1] b,
                     real_t ld,
                     real_t[::1] dd,
                     real_t[::1] ud,
                     real_t[::1] uud,
                     int axis):
    cdef:
        int i, j, k, strides
        int N = uud.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                BBD_matvec_ptr(v_ptr, b_ptr, ld, &dd[0], &ud[0], &uud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                BBD_matvec_ptr(v_ptr, b_ptr, ld, &dd[0], &ud[0], &uud[0], N, strides)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                BBD_matvec_ptr(v_ptr, b_ptr, ld, &dd[0], &ud[0], &uud[0], N, strides)

def BBD_matvec2D_ptr(T[:, ::1] v,
                     T[:, ::1] b,
                     real_t ld,
                     real_t[::1] dd,
                     real_t[::1] ud,
                     real_t[::1] uud,
                     int axis):
    cdef:
        int i, j, strides
        int N = uud.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            BBD_matvec_ptr(v_ptr, b_ptr, ld, &dd[0], &ud[0], &uud[0], N, strides)
    elif axis == 1:
        for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            BBD_matvec_ptr(v_ptr, b_ptr, ld, &dd[0], &ud[0], &uud[0], N, strides)

def BBD_matvec1D_ptr(T[::1] v,
                     T[::1] b,
                     real_t ld,
                     real_t[::1] dd,
                     real_t[::1] ud,
                     real_t[::1] uud):
    cdef:
        int i, j, strides
        int N = uud.shape[0]
        T* v_ptr = &v[0]
        T* b_ptr = &b[0]
    BBD_matvec_ptr(v_ptr, b_ptr, ld, &dd[0], &ud[0], &uud[0], N, 1)

def Helmholtz_matvec_1D(np.ndarray[T, ndim=1] v,
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
        double p

    k = N-1
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*v[k-2]
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - M_PI_2*beta*v[k-3]

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        if k % 2 == 0:
            s2 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*(v[k-2] + v[k+2]) + p*s2
        else:
            s1 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*(v[k-2] + v[k+2]) + p*s1

    k = 1
    s1 += v[k+2]
    s2 += v[k+1]
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*v[k+2] + ud[k]*alfa*s1
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - M_PI_2*beta*v[k+1] + ud[k-1]*alfa*s2

cdef void Helmholtz_matvec_ptr(T* v,
                               T* b,
                               real_t alfa,
                               real_t beta,
                               real_t* dd,
                               real_t* ud,
                               real_t* bd,
                               int N,
                               int st):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        T s1 = 0.0
        T s2 = 0.0
        double p

    k = N-1
    b[k*st] = (dd[k]*alfa + bd[k]*beta)*v[k*st] - M_PI_2*beta*v[(k-2)*st]
    b[(k-1)*st] = (dd[k-1]*alfa + bd[k-1]*beta)*v[(k-1)*st] - M_PI_2*beta*v[(k-3)*st]

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        if k % 2 == 0:
            s2 += v[(k+2)*st]
            b[k*st] = (dd[k]*alfa + bd[k]*beta)*v[k*st] - M_PI_2*beta*(v[(k-2)*st] + v[(k+2)*st]) + p*s2
        else:
            s1 += v[(k+2)*st]
            b[k*st] = (dd[k]*alfa + bd[k]*beta)*v[k*st] - M_PI_2*beta*(v[(k-2)*st] + v[(k+2)*st]) + p*s1

    k = 1
    s1 += v[(k+2)*st]
    s2 += v[(k+1)*st]
    b[k*st] = (dd[k]*alfa + bd[k]*beta)*v[k*st] - M_PI_2*beta*v[(k+2)*st] + ud[k]*alfa*s1
    b[(k-1)*st] = (dd[k-1]*alfa + bd[k-1]*beta)*v[(k-1)*st] - M_PI_2*beta*v[(k+1)*st] + ud[k-1]*alfa*s2

def Helmholtz_matvec3D_ptr(T[:, :, ::1] v,
                           T[:, :, ::1] b,
                           real_t[:, :, ::1] alfa,
                           real_t[:, :, ::1] beta,
                           # 3 upper diagonals of SBB
                           real_t[::1] dd,
                           real_t[::1] ud,
                           real_t[::1] bd,
                           int axis):
    cdef:
        int i, j, k, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                v_ptr = &v[0, j, k]
                b_ptr = &b[0, j, k]
                Helmholtz_matvec_ptr(v_ptr, b_ptr, alfa[0, j, k],
                                     beta[0, j, k], &dd[0], &ud[0], &bd[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                v_ptr = &v[i, 0, k]
                b_ptr = &b[i, 0, k]
                Helmholtz_matvec_ptr(v_ptr, b_ptr, alfa[i, 0, k],
                                     beta[i, 0, k], &dd[0], &ud[0], &bd[0], N, strides)

    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_ptr = &v[i, j, 0]
                b_ptr = &b[i, j, 0]
                Helmholtz_matvec_ptr(v_ptr, b_ptr, alfa[i, j, 0],
                                     beta[i, j, 0], &dd[0], &ud[0], &bd[0], N, strides)

def Helmholtz_matvec2D_ptr(T[:, ::1] v,
                           T[:, ::1] b,
                           real_t[:, ::1] alfa,
                           real_t[:, ::1] beta,
                           # 3 upper diagonals of SBB
                           real_t[::1] dd,
                           real_t[::1] ud,
                           real_t[::1] bd,
                           int axis):
    cdef:
        int i, j, strides
        int N = dd.shape[0]
        T* v_ptr
        T* b_ptr

    strides = v.strides[axis]/v.itemsize
    if axis == 0:
        for j in range(v.shape[1]):
            v_ptr = &v[0, j]
            b_ptr = &b[0, j]
            Helmholtz_matvec_ptr(v_ptr, b_ptr, alfa[0, j],
                                 beta[0, j], &dd[0], &ud[0], &bd[0], N, strides)

    elif axis == 1:
       for i in range(v.shape[0]):
            v_ptr = &v[i, 0]
            b_ptr = &b[i, 0]
            Helmholtz_matvec_ptr(v_ptr, b_ptr, alfa[i, 0],
                                 beta[i, 0], &dd[0], &ud[0], &bd[0], N, strides)

def Helmholtz_matvec(v, b, alfa, beta, dd, ud, bd, axis):
    if v.ndim == 1:
        Helmholtz_matvec_1D(v, b, alfa, beta, dd, ud, bd)
    elif v.ndim == 2:
        Helmholtz_matvec2D_ptr(v, b, alfa, beta, dd, ud, bd, axis)
    elif v.ndim == 3:
        Helmholtz_matvec3D_ptr(v, b, alfa, beta, dd, ud, bd, axis)

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

def Biharmonic_matvec_1D(T[::1] v,
                         T[::1] b,
                         real_t a0,
                         real_t[::1] alfa,
                         real_t[::1] beta,
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
    v_ptr = &v[0]
    b_ptr = &b[0]
    Biharmonic_matvec_ptr(v_ptr, b_ptr, a0, alfa[0],
                          beta[0], &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0],
                          &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], N, strides)

def Biharmonic_matvec(v, b, a0, alfa, beta, sii, siu, siuu, ail, aii,
                      aiu, bill, bil, bii, biu, biuu, axis=0):
    if v.ndim == 1:
        Biharmonic_matvec_1D(v, b, a0, alfa, beta, sii, siu, siuu, ail, aii,
                             aiu, bill, bil, bii, biu, biuu)
    elif v.ndim == 2:
        Biharmonic_matvec2D_ptr(v, b, a0, alfa, beta, sii, siu, siuu, ail, aii,
                                aiu, bill, bil, bii, biu, biuu, axis)
    elif v.ndim == 3:
        Biharmonic_matvec3D_ptr(v, b, a0, alfa, beta, sii, siu, siuu, ail, aii,
                                aiu, bill, bil, bii, biu, biuu, axis)
