#cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
import cython
cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from libc.math cimport M_PI, M_PI_2
np.import_array()

ctypedef fused T:
    double
    complex

ctypedef void (*funv)(T* const, T*, int, int, void* const)

cdef IterAllButAxis(funv f, np.ndarray[T, ndim=1] input_array, np.ndarray[T, ndim=1] output_array, int st, int N, int axis, long int[::1] shapein, long int[::1] shapeout, void* const data):
    cdef:
        np.flatiter ita = np.PyArray_IterAllButAxis(input_array.reshape(shapein), &axis)
        np.flatiter ito = np.PyArray_IterAllButAxis(output_array.reshape(shapeout), &axis)
    while np.PyArray_ITER_NOTDONE(ita):
        f(<T* const>np.PyArray_ITER_DATA(ita), <T*>np.PyArray_ITER_DATA(ito), st, N, data)
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ito)

def imult(T[:, :, ::1] array, double scale):
    cdef int i, j, k

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                array[i, j, k] *= scale
    return array

ctypedef struct CDN:
    double* ld
    double* ud
    int N

cdef void CDN_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i
        CDN* c0 = <CDN*>data
        double* ld = c0.ld
        double* ud = c0.ud
        int Nd = c0.N

    b[0] = ud[0]*v[st]
    b[(Nd-1)*st] = ld[Nd-2]*v[(Nd-2)*st]
    for i in range(1, Nd-1):
        b[i*st] = ud[i]*v[(i+1)*st] + ld[i-1]*v[(i-1)*st]

cpdef CDN_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] ld, double[::1] ud):
    cdef:
        CDN c0 = CDN(&ld[0], &ud[0], ud.shape[0]+1)
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CDN_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](CDN_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct BDN:
    double* ld
    double* dd
    double* ud
    int N

cdef void BDN_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        BDN* c0 = <BDN*>data
        double* ld = c0.ld
        double* dd = c0.dd
        double ud = c0.ud[0]
        int M = c0.N

    b[0] = ud*v[2*st] + dd[0]*v[0]
    b[st] = ud*v[3*st] + dd[1]*v[st]
    b[(M-2)*st] = ld[M-4]*v[(M-4)*st] + dd[M-2]*v[(M-2)*st]
    b[(M-1)*st] = ld[M-3]*v[(M-3)*st] + dd[M-1]*v[(M-1)*st]
    for i in range(2, M-2):
        b[i*st] = ud*v[(i+2)*st] + dd[i]*v[i*st] + ld[i-2]*v[(i-2)*st]

cpdef BDN_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] ld, double[::1] dd, double ud):
    cdef:
        BDN c0 = BDN(&ld[0], &dd[0], &ud, dd.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](BDN_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](BDN_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct CDD0:
    double* ld
    double* ud
    int Nd

cdef void CDD_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i
        CDD0* c0 = <CDD0*>data
        double* ld = c0.ld
        double* ud = c0.ud
        int Nd = c0.Nd

    b[0] = ud[0]*v[st]
    b[(Nd-1)*st] = ld[Nd-2]*v[(Nd-2)*st]
    for i in range(1, Nd-1):
        b[i*st] = ud[i]*v[(i+1)*st] + ld[i-1]*v[(i-1)*st]

cpdef CDD_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] ld, double[::1] ud):
    cdef:
        CDD0 c0 = CDD0(&ld[0], &ud[0], ud.shape[0]+1)
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CDD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](CDD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct SBB:
    double* dd
    int N

cdef void SBB_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        SBB* c0 = <SBB*>data
        double* dd = c0.dd
        int M = c0.N
        double p, r
        T d, s1, s2, o1, o2

    j = M-1
    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    b[j*st] = dd[j]*v[j*st]
    b[(j-1)*st] = dd[j-1]*v[(j-1)*st]
    for k in range(M-3, -1, -1):
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

cpdef SBB_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] dd):
    cdef:
        SBB c0 = SBB(&dd[0], dd.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](SBB_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](SBB_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b


ctypedef struct ADD:
    double* dd
    int N

cdef void ADD_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        ADD* c0 = <ADD*>data
        double* dd = c0.dd
        int M = c0.N
        double p
        T s1 = 0.0
        T s2 = 0.0
        T d

    k = M-1
    b[k*st] = dd[k]*v[k*st]
    b[(k-1)*st] = dd[k-1]*v[(k-1)*st]
    for k in range(M-3, -1, -1):
        j = k+2
        p = -4*(k+1)*M_PI
        if j % 2 == 0:
            s1 += v[j*st]
            b[k*st] = dd[k]*v[k*st] + p*s1
        else:
            s2 += v[j*st]
            b[k*st] = dd[k]*v[k*st] + p*s2

cpdef ADD_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] dd):
    cdef:
        ADD c0 = ADD(&dd[0], dd.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](ADD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](ADD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

cdef void ATT_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        double p0, p1
        T s1 = 0.0
        T s2 = 0.0
        T s3 = 0.0
        T s4 = 0.0
        T d

    k = N-1
    b[(N-1)*st] = 0
    b[(N-2)*st] = 0
    for k in range(N-3, -1, -1):
        j = k+2
        p0 = M_PI/2
        p1 = M_PI/2*k**2
        if j % 2 == 0:
            s1 += j*v[j*st]
            s3 += j**3*v[j*st]
            b[k*st] = p0*s3 - p1*s1
        else:
            s2 += j*v[j*st]
            s4 += j**3*v[j*st]
            b[k*st] = p0*s4 - p1*s2

cpdef ATT_matvec(np.ndarray v, np.ndarray b, int axis):
    cdef:
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](ATT_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    else:
        IterAllButAxis[complex](ATT_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    return b

cdef void GLL_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        double p0, p1
        T s1 = 0.0
        T s2 = 0.0
        T s3 = 0.0
        T s4 = 0.0
        T d

    k = N-1
    b[(N-1)*st] = 0
    b[(N-2)*st] = 0
    for k in range(N-3, -1, -1):
        j = k+2
        p0 = 2*(k+0.5)/(2*k+1)
        p1 = p0*k*(k+1)
        if j % 2 == 0:
            s1 += j*(j+1)*v[j*st]
            s3 += v[j*st]
            b[k*st] = p0*s1 - p1*s3
        else:
            s2 += j*(j+1)*v[j*st]
            s4 += v[j*st]
            b[k*st] = p0*s2 - p1*s4

cpdef GLL_matvec(np.ndarray v, np.ndarray b, int axis):
    cdef:
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](GLL_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    else:
        IterAllButAxis[complex](GLL_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    return b


cdef void CLL_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        double p
        T s1 = 0.0
        T s2 = 0.0

    b[(N-1)*st] = 0
    for k in range(N-2, -1, -1):
        j = k+1
        if j % 2 == 0:
            s1 += v[j*st]
            b[k*st] = 2*s1
        else:
            s2 += v[j*st]
            b[k*st] = 2*s2

cpdef CLL_matvec(np.ndarray v, np.ndarray b, int axis):
    cdef:
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CLL_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    else:
        IterAllButAxis[complex](CLL_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    return b

cdef void CTSD_matvec_ptr(T* const v,
                          T* b,
                          int st,
                          int N,
                          void* const data):
    cdef:
        int i, ii
        T sum_u0, sum_u1
        double pi = np.pi
        double pi2 = 2*np.pi

    sum_u0 = 0.0
    sum_u1 = 0.0

    b[(N-1)*st] = 0.0
    b[(N-2)*st] = -(N-2+1)*pi*v[(N-3)*st]
    b[(N-3)*st] = -(N-3+1)*pi*v[(N-4)*st]
    for i in xrange(N-4, -1, -1):
        ii = i*st
        if i > 0:
            b[ii] = -(i+1)*pi*v[(i-1)*st]
        else:
            b[ii] = 0
        if i % 2 == 0:
            sum_u0 = sum_u0 + v[(i+1)*st]
            b[ii] -= sum_u0*pi2
        else:
            sum_u1 = sum_u1 + v[(i+1)*st]
            b[ii] -= sum_u1*pi2

cpdef CTSD_matvec(np.ndarray v, np.ndarray b, int axis):
    cdef:
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CTSD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    else:
        IterAllButAxis[complex](CTSD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    return b

cdef void CTT_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        double p
        T s1 = 0.0
        T s2 = 0.0

    b[(N-1)*st] = 0
    for k in range(N-2, -1, -1):
        j = k+1
        if j % 2 == 0:
            s1 += (k+1)*v[j*st]
            b[k*st] = M_PI*s1
        else:
            s2 += (k+1)*v[j*st]
            b[k*st] = M_PI*s2

cpdef CTT_matvec(np.ndarray v, np.ndarray b, int axis):
    cdef:
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CTT_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    else:
        IterAllButAxis[complex](CTT_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, NULL)
    return b

ctypedef struct TD:
    double* ld
    double* dd
    double* ud
    int N

cdef void Tridiagonal_matvec_ptr(T* const v,
                                 T* b,
                                 int st,
                                 int N,
                                 void* const data):
    cdef:
        int i
        TD* c0 = <TD*>data
        double* ld = c0.ld
        double* dd = c0.dd
        double* ud = c0.ud
        int M = c0.N

    b[0] = dd[0]*v[0] + ud[0]*v[2*st]
    b[st] = dd[1]*v[st] + ud[1]*v[3*st]
    for i in range(2, M-2):
        b[i*st] = ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = M-2
    b[i*st] = ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st]
    i = M-1
    b[i*st] = ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st]

cpdef Tridiagonal_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] ld, double[::1] dd, double[::1] ud):
    cdef:
        TD c0 = TD(&ld[0], &dd[0], &ud[0], dd.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](Tridiagonal_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](Tridiagonal_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct PD:
    double* ldd
    double* ld
    double* dd
    double* ud
    double* udd
    int N

cdef void Pentadiagonal_matvec_ptr(T* const v,
                                   T* b,
                                   int st,
                                   int N,
                                   void* const data):
    cdef:
        int i
        PD* c0 = <PD*>data
        double* ldd = c0.ldd
        double* ld = c0.ld
        double* dd = c0.dd
        double* ud = c0.ud
        double* udd = c0.udd
        int M = c0.N

    b[0] = dd[0]*v[0] + ud[0]*v[2*st] + udd[0]*v[4*st]
    b[1*st] = dd[1]*v[1*st] + ud[1]*v[3*st] + udd[1]*v[5*st]
    b[2*st] = ld[0]*v[0] + dd[2]*v[2*st] + ud[2]*v[4*st] + udd[2]*v[6*st]
    b[3*st] = ld[1]*v[1*st] + dd[3]*v[3*st] + ud[3]*v[5*st] + udd[3]*v[7*st]
    for i in range(4, M-4):
        b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st] + udd[i]*v[(i+4)*st]
    i = M-4
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = M-3
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = M-2
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st]
    i = M-1
    b[i*st] = ldd[i-4]*v[(i-4)*st] + ld[i-2]*v[(i-2)*st] + dd[i]*v[i*st]

cpdef Pentadiagonal_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] ldd, double[::1] ld, double[::1] dd, double[::1] ud, double[::1] udd):
    cdef:
        PD c0 = PD(&ldd[0], &ld[0], &dd[0], &ud[0], &udd[0], dd.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](Pentadiagonal_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](Pentadiagonal_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct CBD:
    double* ld
    double* ud
    double* udd
    int N

cdef void CBD_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i
        CBD* c0 = <CBD*>data
        double* ld = c0.ld
        double* ud = c0.ud
        double* udd = c0.udd
        int M = c0.N

    b[0] = ud[0]*v[1*st] + udd[0]*v[3*st]
    for i in range(1, M):
        b[i*st] = ld[i-1]*v[(i-1)*st] + ud[i]*v[(i+1)*st] + udd[i]*v[(i+3)*st]
    i = M
    b[i*st] = ld[i-1]*v[(i-1)*st] + ud[i]*v[(i+1)*st]

cpdef CBD_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] ld, double[::1] ud, double[::1] udd):
    cdef:
        CBD c0 = CBD(&ld[0], &ud[0], &udd[0], udd.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]/v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CBD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](CBD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct CDB:
    double* lld
    double* ld
    double* ud
    int N

cdef void CDB_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int i, j, k
        CDB* c0 = <CDB*>data
        double* lld = c0.lld
        double* ld = c0.ld
        double* ud = c0.ud
        int M = c0.N

    b[0] = ud[0]*v[st]
    for k in range(1, 3):
        b[k*st] = ld[k-1]*v[(k-1)*st] + ud[k]*v[(k+1)*st]
    for k in range(3, M):
        b[k*st] = lld[k-3]*v[(k-3)*st] + ld[k-1]*v[(k-1)*st] + ud[k]*v[(k+1)*st]
    for k in xrange(M, M+2):
        b[k*st] = lld[k-3]*v[(k-3)*st] + ld[k-1]* v[(k-1)*st]
    b[(M+2)*st] = lld[M-1]*v[(M-1)*st]

cpdef CDB_matvec(np.ndarray v, np.ndarray b, int axis, double[::1] lld, double[::1] ld, double[::1] ud):
    cdef:
        CDB c0 = CDB(&lld[0], &ld[0], &ud[0], ud.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](CDB_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](CDB_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

ctypedef struct BBD:
    double* ld
    double* dd
    double* ud
    double* uud
    int N

cdef void BBD_matvec_ptr(T* const v,
                         T* b,
                         int st,
                         int N,
                         void* const data):
    cdef:
        int k
        BBD* c0 = <BBD*>data
        double* ld = c0.ld
        double* dd = c0.dd
        double* ud = c0.ud
        double* uud = c0.uud
        int M = c0.N

    b[0] = dd[0]*v[0] + ud[0]*v[2*st] + uud[0]*v[4*st]
    b[st] = dd[1]*v[st] + ud[1]*v[3*st] + uud[1]*v[5*st]
    for k in range(2, M):
        b[k*st] = ld[0]*v[(k-2)*st] + dd[k]*v[k*st] + ud[k]*v[(k+2)*st] + uud[k]*v[(k+4)*st]
    for k in range(M, M+2):
        b[k*st] = ld[0]*v[(k-2)*st] + dd[k]*v[k*st] + ud[k]*v[(k+2)*st]

cpdef BBD_matvec(np.ndarray v, np.ndarray b, int axis, double ld, double[::1] dd, double[::1] ud, double[::1] uud):
    cdef:
        BBD c0 = BBD(&ld, &dd[0], &ud[0], &uud[0], uud.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
    if v.dtype.char in 'fdg':
        IterAllButAxis[double](BBD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    else:
        IterAllButAxis[complex](BBD_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), st, N, axis, shape, shape, &c0)
    return b

# Helmholtz solver has nonconstant coefficients alfa and beta, so need special iterator

ctypedef void (*funH)(T* const, T*, double* const, double* const, int, int, void* const)

cdef ABIterAllButAxis(funH f, np.ndarray[T, ndim=1] input_array, np.ndarray[T, ndim=1] output_array,
                      np.ndarray[double, ndim=1] alfa, np.ndarray[double, ndim=1] beta, int st, int N,
                      int axis, long int[::1] shape, long int[::1] ashape, void* const data):
    cdef:
        np.flatiter ita = np.PyArray_IterAllButAxis(input_array.reshape(shape), &axis)
        np.flatiter ito = np.PyArray_IterAllButAxis(output_array.reshape(shape), &axis)
        np.flatiter alfai = np.PyArray_IterAllButAxis(alfa.reshape(ashape), &axis)
        np.flatiter betai = np.PyArray_IterAllButAxis(beta.reshape(ashape), &axis)
    while np.PyArray_ITER_NOTDONE(ita):
        f(<T* const>np.PyArray_ITER_DATA(ita), <T*>np.PyArray_ITER_DATA(ito), <double*>np.PyArray_ITER_DATA(alfai), <double*>np.PyArray_ITER_DATA(betai), st, N, data)
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ito)
        np.PyArray_ITER_NEXT(alfai)
        np.PyArray_ITER_NEXT(betai)

ctypedef struct HH:
    double* dd
    double* ud
    double* bd
    int N

cdef void Helmholtz_matvec_ptr(T* const v,
                               T* b,
                               double* const alfa,
                               double* const beta,
                               int st,
                               int N,
                               void* const data):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        HH* c0 = <HH*>data
        double* dd = c0.dd
        double* ud = c0.ud
        double* bd = c0.bd
        int M = c0.N
        T s1 = 0.0
        T s2 = 0.0
        double p
        double alf = alfa[0]
        double bet = beta[0]

    k = M-1
    b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st] - M_PI_2*bet*v[(k-2)*st]
    b[(k-1)*st] = (dd[k-1]*alf + bd[k-1]*bet)*v[(k-1)*st] - M_PI_2*bet*v[(k-3)*st]

    for k in range(M-3, 1, -1):
        p = ud[k]*alf
        if k % 2 == 0:
            s2 += v[(k+2)*st]
            b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st] - M_PI_2*bet*(v[(k-2)*st] + v[(k+2)*st]) + p*s2
        else:
            s1 += v[(k+2)*st]
            b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st] - M_PI_2*bet*(v[(k-2)*st] + v[(k+2)*st]) + p*s1

    k = 1
    s1 += v[(k+2)*st]
    s2 += v[(k+1)*st]
    b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st] - M_PI_2*bet*v[(k+2)*st] + ud[k]*alf*s1
    b[(k-1)*st] = (dd[k-1]*alf + bd[k-1]*bet)*v[(k-1)*st] - M_PI_2*bet*v[(k+1)*st] + ud[k-1]*alf*s2

cpdef Helmholtz_matvec(np.ndarray v, np.ndarray b, np.ndarray alfa, np.ndarray beta, A, B, int axis):
    cdef:
        double[::1] dd = A[0]
        double[::1] ud = A[2]
        double[::1] bd = B[0]
        HH c0 = HH(&dd[0], &ud[0], &bd[0], A[0].shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        np.ndarray[long int, ndim=1] ashape = np.array(np.shape(alfa), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]/v.itemsize
    if v.dtype.char in 'fdg':
        ABIterAllButAxis[double](Helmholtz_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), np.PyArray_Ravel(alfa, np.NPY_CORDER), np.PyArray_Ravel(beta, np.NPY_CORDER), st, N, axis, shape, ashape, &c0)
    else:
        ABIterAllButAxis[complex](Helmholtz_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), np.PyArray_Ravel(alfa, np.NPY_CORDER), np.PyArray_Ravel(beta, np.NPY_CORDER), st, N, axis, shape, ashape, &c0)
    return b


ctypedef struct HN:
    double* dd
    double* ud
    double* bl
    double* bd
    double* bu
    int N

cdef void Helmholtz_Neumann_matvec_ptr(T* const v,
                                       T* b,
                                       double* const alfa,
                                       double* const beta,
                                       int st,
                                       int N,
                                       void* const data):
    # b = (alfa*A + beta*B)*v
    # A matrix has diagonal dd and upper second diagonal at ud
    # B matrix has diagonal bd and second upper and lower diagonals bu and bl
    cdef:
        int i, j, k, j2
        HN* c0 = <HN*>data
        double* dd = c0.dd
        double* ud = c0.ud
        double* bl = c0.bl
        double* bd = c0.bd
        double* bu = c0.bu
        int M = c0.N
        T s1 = 0.0
        T s2 = 0.0
        double p
        double alf = alfa[0]
        double bet = beta[0]

    for k in (M-1, M-2):
        j2 = k*k
        b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st]*j2 + bl[k-2]*bet*v[(k-2)*st]*j2

    for k in range(M-3, 1, -1):
        p = ud[k]*alf
        if k % 2 == 0:
            s2 += v[(k+2)*st]*(k+2)**2
            b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st]*k**2 + bet*(bl[k-2]*v[(k-2)*st]*(k-2)**2 + bu[k]*v[(k+2)*st]*(k+2)**2) + p*s2
        else:
            s1 += v[(k+2)*st]*(k+2)**2
            b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st]*k**2 + bet*(bl[k-2]*v[(k-2)*st]*(k-2)**2 + bu[k]*v[(k+2)*st]*(k+2)**2) + p*s1

    k = 1
    s1 += v[(k+2)*st]*(k+2)**2
    b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st]*k**2 + bet*(bu[k]*v[(k+2)*st]*(k+2)**2) + ud[k]*alf*s1
    k = 0
    s2 += v[(k+2)*st]*(k+2)**2
    b[k*st] = (dd[k]*alf + bd[k]*bet)*v[k*st]*k**2 + bet*(bu[k]*v[(k+2)*st]*(k+2)**2) + ud[k]*alf*s2
    b[0] += bd[0]*v[0]*bet
    b[2*st] += bl[0]*v[0]*bet

cpdef Helmholtz_Neumann_matvec(np.ndarray v, np.ndarray b, np.ndarray alfa, np.ndarray beta, A, B, int axis):
    cdef:
        HN c0
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        np.ndarray[long int, ndim=1] ashape = np.array(np.shape(alfa), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]//v.itemsize
        int M = A[0].shape[0]
        np.ndarray[long int, ndim=1] k = np.arange(M)
        np.ndarray[long int, ndim=1] j2 = k**2
        double[::1] dd = np.zeros_like(A[0])
        double[::1] ud = np.zeros_like(A[2])
        double[::1] bl = np.zeros_like(B[-2])
        double[::1] bd = np.zeros_like(B[0])
        double[::1] bu = np.zeros_like(B[2])

    j2[0] = 1
    j2[:] = 1/j2
    j2[0] = 0
    dd[:] = A[0]*j2
    ud[:] = A[2]*j2[2:]
    j2[0] = 1
    bd[:] = B[0]*j2
    bu[:] = B[2]*j2[2:]
    bl[:] = B[-2]*j2[:-2]
    c0 = HN(&dd[0], &ud[0], &bl[0], &bd[0], &bu[0], A[0].shape[0])

    if v.dtype.char in 'fdg':
        ABIterAllButAxis[double](Helmholtz_Neumann_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), np.PyArray_Ravel(alfa, np.NPY_CORDER), np.PyArray_Ravel(beta, np.NPY_CORDER), st, N, axis, shape, ashape, &c0)
    else:
        ABIterAllButAxis[complex](Helmholtz_Neumann_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), np.PyArray_Ravel(alfa, np.NPY_CORDER), np.PyArray_Ravel(beta, np.NPY_CORDER), st, N, axis, shape, ashape, &c0)
    return b

ctypedef struct Bi:
    double* a0
    # 3 upper diagonals of SBB
    double* sii
    double* siu
    double* siuu
    # All 3 diagonals of ABB
    double* ail
    double* aii
    double* aiu
    # All 5 diagonals of BBB
    double* bill
    double* bil
    double* bii
    double* biu
    double* biuu
    int N

cdef void Biharmonic_matvec_ptr(T* const v,
                                T* b,
                                double* const alfa,
                                double* const beta,
                                int st,
                                int N,
                                void* const data):
    cdef:
        int i, j, k
        vector[double] ldd, ld, dd, ud, udd
        double p, r
        T d, s1, s2, o1, o2
        Bi* c0 = <Bi*>data
        double* sii = c0.sii
        double* siu = c0.siu
        double* siuu = c0.siuu
        double* ail = c0.ail
        double* aii = c0.aii
        double* aiu = c0.aiu
        double* bill = c0.bill
        double* bil = c0.bil
        double* bii = c0.bii
        double* biu = c0.biu
        double* biuu = c0.biuu
        double a0 = c0.a0[0]
        int M = c0.N
        double alf = alfa[0]
        double bet = beta[0]

    dd.resize(M)
    ld.resize(M)
    ldd.resize(M)
    ud.resize(M)
    udd.resize(M)

    for i in range(M):
        dd[i] = a0*sii[i] + alf*aii[i] + bet*bii[i]

    for i in range(M-2):
        ld[i] = alf*ail[i] + bet*bil[i]

    for i in range(M-4):
        ldd[i] = bet*bill[i]

    for i in range(M-2):
        ud[i] = a0*siu[i] + alf*aiu[i] + bet*biu[i]

    for i in range(M-4):
        udd[i] = a0*siuu[i] + bet*biuu[i]

    i = M-1
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st]
    i = M-2
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st]
    i = M-3
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = M-4
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st]
    i = M-5
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st] + udd[i]*v[(i+4)*st]
    i = M-6
    b[i*st] = ldd[i-4]*v[(i-4)*st]+ ld[i-2]* v[(i-2)*st] + dd[i]*v[i*st] + ud[i]*v[(i+2)*st] + udd[i]*v[(i+4)*st]

    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    for k in range(M-7, -1, -1):
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

cpdef Biharmonic_matvec(np.ndarray v, np.ndarray b, double a0, np.ndarray alfa, np.ndarray beta,
                        double[::1] sii, double[::1] siu, double[::1] siuu,
                        double[::1] ail, double[::1] aii, double[::1] aiu,
                        double[::1] bill, double[::1] bil, double[::1] bii, double[::1] biu, double[::1] biuu, int axis=0):
    cdef:
        Bi c0 = Bi(&a0, &sii[0], &siu[0], &siuu[0], &ail[0], &aii[0], &aiu[0], &bill[0], &bil[0], &bii[0], &biu[0], &biuu[0], sii.shape[0])
        np.ndarray[long int, ndim=1] shape = np.array(np.shape(v), dtype=int)
        np.ndarray[long int, ndim=1] ashape = np.array(np.shape(alfa), dtype=int)
        int N = v.shape[axis]
        int st = v.strides[axis]/v.itemsize
    if v.dtype.char in 'fdg':
        ABIterAllButAxis[double](Biharmonic_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), np.PyArray_Ravel(alfa, np.NPY_CORDER), np.PyArray_Ravel(beta, np.NPY_CORDER), st, N, axis, shape, ashape, &c0)
    else:
        ABIterAllButAxis[complex](Biharmonic_matvec_ptr, np.PyArray_Ravel(v, np.NPY_CORDER), np.PyArray_Ravel(b, np.NPY_CORDER), np.PyArray_Ravel(alfa, np.NPY_CORDER), np.PyArray_Ravel(beta, np.NPY_CORDER), st, N, axis, shape, ashape, &c0)
    return b
