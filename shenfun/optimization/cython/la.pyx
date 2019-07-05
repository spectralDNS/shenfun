#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

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


#def PDMA_SymLU(np.ndarray[np.float64_t, ndim=1, mode='c'] d,
               #np.ndarray[np.float64_t, ndim=1, mode='c'] e,
               #np.ndarray[np.float64_t, ndim=1, mode='c'] f):
def PDMA_SymLU(real_t[::1] d,
               real_t[::1] e,
               real_t[::1] f):

    cdef:
        unsigned int n = d.shape[0]
        unsigned int m = e.shape[0]
        unsigned int k = n - m
        unsigned int i
        real_t lam

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

cdef void PDMA_SymLU_ptr(real_t* d,
                         real_t* e,
                         real_t* f,
                         int n,
                         int st):

    cdef:
        int m, k, i
        real_t lam

    m = n-2
    k = n-m

    for i in xrange(n-2*k):
        lam = e[i*st]/d[i*st]
        d[(i+k)*st] -= lam*e[i*st]
        e[(i+k)*st] -= lam*f[i*st]
        e[i*st] = lam
        lam = f[i*st]/d[i*st]
        d[(i+2*k)*st] -= lam*f[i*st]
        f[i*st] = lam

    lam = e[(n-4)*st]/d[(n-4)*st]
    d[(n-2)*st] -= lam*e[(n-4)*st]
    e[(n-4)*st] = lam
    lam = e[(n-3)*st]/d[(n-3)*st]
    d[(n-1)*st] -= lam*e[(n-3)*st]
    e[(n-3)*st] = lam

def PDMA_SymLU_3D(real_t[:, :, ::1] d0,
                  real_t[:, :, ::1] d1,
                  real_t[:, :, ::1] d2,
                  int axis):
    cdef:
        unsigned int i, j, k, strides

    strides = d0.strides[axis]/d0.itemsize
    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                PDMA_SymLU_ptr(&d0[0,j,k],
                               &d1[0,j,k],
                               &d2[0,j,k],
                               d0.shape[axis]-4,
                               strides)

    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                PDMA_SymLU_ptr(&d0[i, 0, k],
                               &d1[i, 0, k],
                               &d2[i, 0, k],
                               d0.shape[axis]-4,
                               strides)

    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                PDMA_SymLU_ptr(&d0[i,j,0],
                               &d1[i,j,0],
                               &d2[i,j,0],
                               d0.shape[axis]-4,
                               strides)

def PDMA_SymLU_2D(real_t[:, ::1] d0,
                  real_t[:, ::1] d1,
                  real_t[:, ::1] d2,
                  int axis):
    cdef:
        unsigned int i, j, strides

    strides = d0.strides[axis]/d0.itemsize
    if axis == 0:
        for j in range(d0.shape[1]):
            PDMA_SymLU_ptr(&d0[0, j],
                           &d1[0, j],
                           &d2[0, j],
                           d0.shape[axis]-4,
                           strides)

    elif axis == 1:
        for i in range(d0.shape[0]):
            PDMA_SymLU_ptr(&d0[i, 0],
                           &d1[i, 0],
                           &d2[i, 0],
                           d0.shape[axis]-4,
                           strides)

def PDMA_SymLU_VC(d, a, l, axis=0):
    n = d.ndim
    if n == 1:
        PDMA_SymLU(d, a, l)
    elif n == 2:
        PDMA_SymLU_2D(d, a, l, axis)
    elif n == 3:
        PDMA_SymLU_3D(d, a, l, axis)

def PDMA_SymSolve(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        PDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        PDMA_Symsolve2D_ptr(d, a, l, x, axis)
    elif n == 3:
        PDMA_Symsolve3D_ptr(d, a, l, x, axis)

def PDMA_SymSolve_VC(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        PDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        PDMA_SymSolve2D_VC(d, a, l, x, axis)
    elif n == 3:
        PDMA_SymSolve3D_VC(d, a, l, x, axis)

def PDMA_SymSolve3D_VC(real_t[:, :, ::1] d,
                       real_t[:, :, ::1] a,
                       real_t[:, :, ::1] l,
                       T[:, :, ::1] x,
                       np.int64_t axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, k, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                PDMA_SymSolve_ptr3(&d[0, j, k], &a[0, j, k], &l[0, j, k],
                                   &x[0, j, k], x.shape[axis]-4, strides)

    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                PDMA_SymSolve_ptr3(&d[i, 0, k], &a[i, 0, k], &l[i, 0, k],
                                   &x[i, 0, k], x.shape[axis]-4, strides)

    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                PDMA_SymSolve_ptr3(&d[i, j, 0], &a[i, j, 0], &l[i, j, 0],
                                   &x[i, j, 0], x.shape[axis]-4, strides)


def PDMA_SymSolve2D_VC(real_t[:, ::1] d,
                       real_t[:, ::1] a,
                       real_t[:, ::1] l,
                       T[:, ::1] x,
                       np.int64_t axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(d.shape[1]):
            PDMA_SymSolve_ptr3(&d[0, j], &a[0, j], &l[0, j],
                                &x[0, j], d.shape[axis]-4, strides)

    elif axis == 1:
        for i in range(d.shape[0]):
            PDMA_SymSolve_ptr3(&d[i, 0], &a[i, 0], &l[i, 0],
                                &x[i, 0], d.shape[axis]-4, strides)

def PDMA_SymSolve1D(real_t[::1] d,
                    real_t[::1] e,
                    real_t[::1] f,
                    T[::1] b):
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

cdef void PDMA_SymSolve_ptr3(real_t* d,
                             real_t* e,
                             real_t* f,
                             T* b,
                             int n,
                             int st) nogil:
    cdef:
        int k

    b[2*st] -= e[0]*b[0]
    b[3*st] -= e[st]*b[st]
    for k in range(4, n):
        b[k*st] -= (e[(k-2)*st]*b[(k-2)*st] + f[(k-4)*st]*b[(k-4)*st])

    b[(n-1)*st] /= d[(n-1)*st]
    b[(n-2)*st] /= d[(n-2)*st]
    b[(n-3)*st] /= d[(n-3)*st]
    b[(n-3)*st] -= e[(n-3)*st]*b[(n-1)*st]
    b[(n-4)*st] /= d[(n-4)*st]
    b[(n-4)*st] -= e[(n-4)*st]*b[(n-2)*st]
    for k in range(n-5,-1,-1):
        b[k*st] /= d[k*st]
        b[k*st] -= (e[k*st]*b[(k+2)*st] + f[k*st]*b[(k+4)*st])

cdef void PDMA_SymSolve_ptr(real_t* d,
                            real_t* e,
                            real_t* f,
                            T* b,
                            int n,
                            int st) nogil:
    cdef:
        int k

    b[2*st] -= e[0]*b[0]
    b[3*st] -= e[1]*b[st]
    for k in range(4, n):
        b[k*st] -= (e[k-2]*b[(k-2)*st] + f[k-4]*b[(k-4)*st])

    b[(n-1)*st] /= d[n-1]
    b[(n-2)*st] /= d[n-2]
    b[(n-3)*st] /= d[n-3]
    b[(n-3)*st] -= e[n-3]*b[(n-1)*st]
    b[(n-4)*st] /= d[n-4]
    b[(n-4)*st] -= e[n-4]*b[(n-2)*st]
    for k in range(n-5,-1,-1):
        b[k*st] /= d[k]
        b[k*st] -= (e[k]*b[(k+2)*st] + f[k]*b[(k+4)*st])

def PDMA_Symsolve3D_ptr(real_t[::1] d,
                        real_t[::1] e,
                        real_t[::1] f,
                        T[:, :, ::1] x,
                        np.int64_t axis):
    cdef:
        int n = d.shape[0]
        int i, j, k, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                PDMA_SymSolve_ptr(&d[0], &e[0], &f[0], &x[0,j,k], n, strides)

    elif axis == 1:
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                PDMA_SymSolve_ptr(&d[0], &e[0], &f[0], &x[i,0,k], n, strides)

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                PDMA_SymSolve_ptr(&d[0], &e[0], &f[0], &x[i,j,0], n, strides)

def PDMA_Symsolve2D_ptr(real_t[::1] d,
                        real_t[::1] e,
                        real_t[::1] f,
                        T[:, ::1] x,
                        np.int64_t axis):
    cdef:
        int n = d.shape[0]
        int i, j, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(x.shape[1]):
            PDMA_SymSolve_ptr(&d[0], &e[0], &f[0], &x[0,j], n, strides)

    elif axis == 1:
        for i in range(x.shape[0]):
            PDMA_SymSolve_ptr(&d[0], &e[0], &f[0], &x[i,0], n, strides)

def TDMA_O_SymLU(real_t[::1] d,
                 real_t[::1] ud,
                 real_t[::1] ld):
    cdef:
        unsigned int n = d.shape[0]
        int i

    for i in range(1, n):
        ld[i-1] = ud[i-1]/d[i-1]
        d[i] = d[i] - ld[i-1]*ud[i-1]

def TDMA_SymLU(real_t[::1] d,
               real_t[::1] ud,
               real_t[::1] ld):
    cdef:
        unsigned int n = d.shape[0]
        int i

    for i in range(2, n):
        ld[i-2] = ud[i-2]/d[i-2]
        d[i] = d[i] - ld[i-2]*ud[i-2]

cdef TDMA_SymLU_ptr(real_t* d,
                    real_t* ud,
                    real_t* ld,
                    int n,
                    int st):
    cdef:
        int i

    for i in range(2, n):
        ld[(i-2)*st] = ud[(i-2)*st]/d[(i-2)*st]
        d[i*st] = d[i*st] - ld[(i-2)*st]*ud[(i-2)*st]

def TDMA_SymLU_3D(real_t[:, :, ::1] d0,
                  real_t[:, :, ::1] d1,
                  real_t[:, :, ::1] L,
                  int axis):
    cdef:
        unsigned int i, j, k, strides

    strides = d0.strides[axis]/d0.itemsize
    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                TDMA_SymLU_ptr(&d0[0, j, k],
                               &d1[0, j, k],
                               &L[0, j, k],
                               d0.shape[axis]-2,
                               strides)

    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                TDMA_SymLU_ptr(&d0[i, 0, k],
                               &d1[i, 0, k],
                               &L[i, 0, k],
                               d0.shape[axis]-2,
                               strides)

    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                TDMA_SymLU_ptr(&d0[i, j, 0],
                               &d1[i, j, 0],
                               &L[i, j, 0],
                               d0.shape[axis]-2,
                               strides)

def TDMA_SymLU_2D(real_t[:, ::1] d0,
                  real_t[:, ::1] d1,
                  real_t[:, ::1] L,
                  int axis):
    cdef:
        unsigned int i, j, k, strides

    strides = d0.strides[axis]/d0.itemsize
    if axis == 0:
        for j in range(d0.shape[1]):
            TDMA_SymLU_ptr(&d0[0, j],
                            &d1[0, j],
                            &L[0, j],
                            d0.shape[axis]-2,
                            strides)

    elif axis == 1:
        for i in range(d0.shape[0]):
            TDMA_SymLU_ptr(&d0[i, 0],
                            &d1[i, 0],
                            &L[i, 0],
                            d0.shape[axis]-2,
                            strides)

def TDMA_SymLU_VC(d, a, l, axis=0):
    n = d.ndim
    if n == 1:
        TDMA_SymLU(d, a, l)
    elif n == 2:
        TDMA_SymLU_2D(d, a, l, axis)
    elif n == 3:
        TDMA_SymLU_3D(d, a, l, axis)

cdef void TDMA_SymSolve_ptr(real_t* d,
                            real_t* a,
                            real_t* l,
                            T* x,
                            int n,
                            int st):
    cdef:
        int i

    for i in range(2, n):
        x[i*st] -= l[i-2]*x[(i-2)*st]

    x[(n-1)*st] = x[(n-1)*st]/d[n-1]
    x[(n-2)*st] = x[(n-2)*st]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i*st] = (x[i*st] - a[i]*x[(i+2)*st])/d[i]

def TDMA_SymSolve2D_ptr(real_t[::1] d,
                        real_t[::1] a,
                        real_t[::1] l,
                        T[:, ::1] x,
                        np.int64_t axis):
    """Experimental 2D TDMA solver

    Note it seems to be quite a bit slower than TDMA_SymSolve3D along axis=0
    """
    cdef:
        int n = d.shape[0]
        int i, j, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(x.shape[1]):
            TDMA_SymSolve_ptr(&d[0], &a[0], &l[0], &x[0,j], n, strides)

    elif axis == 1:
        for i in range(x.shape[0]):
            TDMA_SymSolve_ptr(&d[0], &a[0], &l[0], &x[i,0], n, strides)

def TDMA_SymSolve3D_ptr(real_t[::1] d,
                        real_t[::1] a,
                        real_t[::1] l,
                        T[:, :, ::1] x,
                        np.int64_t axis):
    """Experimental 3D TDMA solver

    Note it seems to be quite a bit slower than TDMA_SymSolve3D along axis=0
    """
    cdef:
        int n = d.shape[0]
        int i, j, k, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                TDMA_SymSolve_ptr(&d[0], &a[0], &l[0], &x[0,j,k], n, strides)

    elif axis == 1:
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                TDMA_SymSolve_ptr(&d[0], &a[0], &l[0], &x[i,0,k], n, strides)

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                TDMA_SymSolve_ptr(&d[0], &a[0], &l[0], &x[i,j,0], n, strides)

def TDMA_SymSolve(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        TDMA_SymSolve2D(d, a, l, x, axis)
    elif n == 3:
        TDMA_SymSolve3D(d, a, l, x, axis)

def TDMA_O_SymSolve(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_O_SymSolve1D(d, a, l, x)
    elif n == 2:
        TDMA_O_SymSolve2D(d, a, l, x, axis)
    elif n == 3:
        TDMA_O_SymSolve3D(d, a, l, x, axis)

def TDMA_SymSolve_VC(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        TDMA_SymSolve2D_VC(d, a, l, x, axis)
    elif n == 3:
        TDMA_SymSolve3D_VC(d, a, l, x, axis)

def TDMA_SymSolve1D(real_t[::1] d,
                    real_t[::1] a,
                    real_t[::1] l,
                    T[::1] x):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i

    with nogil:
        for i in range(2, n):
            x[i] -= l[i-2]*x[i-2]

        x[n-1] = x[n-1]/d[n-1]
        x[n-2] = x[n-2]/d[n-2]
        for i in range(n - 3, -1, -1):
            x[i] = (x[i] - a[i]*x[i+2])/d[i]

def TDMA_O_SymSolve1D(real_t[::1] d,
                      real_t[::1] a,
                      real_t[::1] l,
                      T[::1] x):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i

    with nogil:
        for i in range(1, n):
            x[i] -= l[i-1]*x[i-1]

        x[n-1] = x[n-1]/d[n-1]
        for i in range(n - 2, -1, -1):
            x[i] = (x[i] - a[i]*x[i+1])/d[i]

def TDMA_SymSolveC(real_t[::1] d,
                   real_t[::1] a,
                   real_t[::1] l,
                   complex_t[::1] x):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i

    for i in range(2, n):
        x[i].real = x[i].real - l[i-2]*x[i-2].real
        x[i].imag = x[i].imag - l[i-2]*x[i-2].imag
    x[n-1].real = x[n-1].real/d[n-1]
    x[n-1].imag = x[n-1].imag/d[n-1]
    x[n-2].real = x[n-2].real/d[n-2]
    x[n-2].imag = x[n-2].imag/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i].real = (x[i].real - a[i]*x[i+2].real)/d[i]
        x[i].imag = (x[i].imag - a[i]*x[i+2].imag)/d[i]

def TDMA_O_SymSolve3D(real_t[::1] d,
                      real_t[::1] a,
                      real_t[::1] l,
                      T[:,:,::1] x,
                      int axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, k
        real_t d1

    if axis == 0:
        for i in range(1, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[i-1]*x[i-1, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1]

        for i in range(n - 2, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i]*x[i+1, j, k])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(1, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[j-1]*x[i, j-1, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]

            for j in range(n - 2, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[j]*x[i, j+1, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(1, n):
                    x[i, j, k] -= l[k-1]*x[i, j, k-1]

                x[i, j, n-1] = x[i, j, n-1]/d[n-1]
                for k in range(n - 2, -1, -1):
                    x[i, j, k] = (x[i, j, k] - a[k]*x[i, j, k+1])/d[k]

def TDMA_SymSolve3D(real_t[::1] d,
                    real_t[::1] a,
                    real_t[::1] l,
                    T[:,:,::1] x,
                    int axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, k
        real_t d1

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
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i]*x[i+2, j, k])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[j-2]*x[i, j-2, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]
                x[i, n-2, k] = x[i, n-2, k]/d[n-2]

            for j in range(n - 3, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[j]*x[i, j+2, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(2, n):
                    x[i, j, k] -= l[k-2]*x[i, j, k-2]

                x[i, j, n-1] = x[i, j, n-1]/d[n-1]
                x[i, j, n-2] = x[i, j, n-2]/d[n-2]
                for k in range(n - 3, -1, -1):
                    x[i, j, k] = (x[i, j, k] - a[k]*x[i, j, k+2])/d[k]

def TDMA_O_SymSolve2D(real_t[::1] d,
                      real_t[::1] a,
                      real_t[::1] l,
                      T[:, ::1] x,
                      int axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, k
        real_t d1

    if axis == 0:
        for i in range(1, n):
            for j in range(x.shape[1]):
                x[i, j] -= l[i-1]*x[i-1, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1]

        for i in range(n - 2, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - a[i]*x[i+1, j])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(1, n):
                x[i, j] -= l[j-1]*x[i, j-1]
            x[i, n-1] = x[i, n-1]/d[n-1]
            for j in range(n - 2, -1, -1):
                x[i, j] = (x[i, j] - a[j]*x[i, j+1])/d[j]


def TDMA_SymSolve2D(real_t[::1] d,
                    real_t[::1] a,
                    real_t[::1] l,
                    T[:, ::1] x,
                    int axis):
    cdef:
        unsigned int n = d.shape[0]
        np.intp_t i, j, k
        real_t d1

    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                x[i, j] -= l[i-2]*x[i-2, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1]
            x[n-2, j] = x[n-2, j]/d[n-2]

        for i in range(n - 3, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - a[i]*x[i+2, j])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                x[i, j] -= l[j-2]*x[i, j-2]

            x[i, n-1] = x[i, n-1]/d[n-1]
            x[i, n-2] = x[i, n-2]/d[n-2]

            for j in range(n - 3, -1, -1):
                x[i, j] = (x[i, j] - a[j]*x[i, j+2])/d[j]

def TDMA_SymSolve3D_VC(real_t[:, :, ::1] d,
                       real_t[:, :, ::1] a,
                       real_t[:, :, ::1] l,
                       T[:, :, ::1] x,
                       np.int64_t axis):
    cdef:
        np.intp_t i, j, k, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                TDMA_SymSolve_ptr3(&d[0, i, j], &a[0, i, j], &l[0, i, j],
                                   &x[0, i, j], d.shape[axis]-2, strides)

    elif axis == 1:
        for i in range(d.shape[0]):
            for j in range(d.shape[2]):
                TDMA_SymSolve_ptr3(&d[i, 0, j], &a[i, 0, j], &l[i, 0, j],
                                   &x[i, 0, j], d.shape[axis]-2, strides)

    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                TDMA_SymSolve_ptr3(&d[i, j, 0], &a[i, j, 0], &l[i, j, 0],
                                   &x[i, j, 0], d.shape[axis]-2, strides)


def TDMA_SymSolve2D_VC(real_t[:, ::1] d,
                       real_t[:, ::1] a,
                       real_t[:, ::1] l,
                       T[:, ::1] x,
                       np.int64_t axis):
    cdef:
        np.intp_t i, j, strides

    strides = x.strides[axis]/x.itemsize
    if axis == 0:
        for j in range(d.shape[1]):
            TDMA_SymSolve_ptr3(&d[0, j], &a[0, j], &l[0, j],
                               &x[0, j], d.shape[axis]-2, strides)

    elif axis == 1:
        for i in range(d.shape[0]):
            TDMA_SymSolve_ptr3(&d[i, 0], &a[i, 0], &l[i, 0],
                               &x[i, 0], d.shape[axis]-2, strides)


cdef void TDMA_SymSolve_ptr3(real_t* d,
                             real_t* a,
                             real_t* l,
                             T* x,
                             int n,
                             int st) nogil:
    cdef:
        np.intp_t i

    for i in range(2, n):
        x[i*st] -= l[(i-2)*st]*x[(i-2)*st]

    x[(n-1)*st] = x[(n-1)*st]/d[(n-1)*st]
    x[(n-2)*st] = x[(n-2)*st]/d[(n-2)*st]
    for i in range(n - 3, -1, -1):
        x[i*st] = (x[i*st] - a[i*st]*x[(i+2)*st])/d[i*st]

def TDMA_SymSolve2D_VC2(real_t[:, ::1] d,
                        real_t[:, ::1] a,
                        real_t[:, ::1] l,
                        T[:, ::1] x,
                        int axis):
    cdef:
        unsigned int n = d.shape[axis]
        np.intp_t i, j, k
        real_t d1

    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                x[i, j] -= l[i-2, j]*x[i-2, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1, j]
            x[n-2, j] = x[n-2, j]/d[n-2, j]

        for i in range(n - 3, -1, -1):
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - a[i, j]*x[i+2, j])/d[i, j]

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                x[i, j] -= l[i, j-2]*x[i, j-2]

            x[i, n-1] = x[i, n-1]/d[i, n-1]
            x[i, n-2] = x[i, n-2]/d[i, n-2]

            for j in range(n - 3, -1, -1):
                x[i, j] = (x[i, j] - a[i, j]*x[i, j+2])/d[i, j]

def TDMA_SymSolve3D_VC2(real_t[:, :, ::1] d,
                        real_t[:, :, ::1] a,
                        real_t[:, :, ::1] l,
                        T[:, :, ::1] x,
                        np.int64_t axis):
    cdef:
        unsigned int n = d.shape[axis]
        np.intp_t i, j, k
        real_t d1

    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[i-2, j, k]*x[i-2, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1, j, k]
                x[n-2, j, k] = x[n-2, j, k]/d[n-2, j, k]

        for i in range(n - 3, -1, -1):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i, j, k]*x[i+2, j, k])/d[i, j, k]

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[i, j-2, k]*x[i, j-2, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[i, n-1, k]
                x[i, n-2, k] = x[i, n-2, k]/d[i, n-2, k]

            for j in range(n - 3, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i, j, k]*x[i, j+2, k])/d[i, j, k]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(2, n):
                    x[i, j, k] -= l[i, j, k-2]*x[i, j, k-2]

                x[i, j, n-1] = x[i, j, n-1]/d[i, j, n-1]
                x[i, j, n-2] = x[i, j, n-2]/d[i, j, n-2]
                for k in range(n - 3, -1, -1):
                    x[i, j, k] = (x[i, j, k] - a[i, j, k]*x[i, j, k+2])/d[i, j, k]

def LU_Helmholtz(A, B, A_s, B_s, neumann, d0, d1, d2, L, axis):
    n = d0.ndim
    if n == 1:
        LU_Helmholtz_1D(A, B, A_s, B_s, neumann, d0, d1, d2, L)
    elif n == 2:
        LU_Helmholtz_2D(A, B, axis, A_s, B_s, neumann, d0, d1, d2, L)
    elif n == 3:
        LU_Helmholtz_3D(A, B, axis, A_s, B_s, neumann, d0, d1, d2, L)

def LU_Helmholtz_1D(A, B,
                    np.float_t A_scale,
                    np.float_t B_scale,
                    bint neumann,
                    np.ndarray[real_t, ndim=1] d0,
                    np.ndarray[real_t, ndim=1] d1,
                    np.ndarray[real_t, ndim=1] d2,
                    np.ndarray[real_t, ndim=1] L):
    cdef:
        int i, N
        np.ndarray[real_t, ndim=1] A_0 = A[0].copy()
        np.ndarray[real_t, ndim=1] A_2 = A[2].copy()
        np.ndarray[real_t, ndim=1] A_4 = A[4].copy()
        np.ndarray[real_t, ndim=1] B_m2 = B[-2].copy()
        np.ndarray[real_t, ndim=1] B_0 = B[0].copy()
        np.ndarray[real_t, ndim=1] B_2 = B[2].copy()

    N = A_0.shape[0]
    if neumann:
        if abs(B_scale) < 1e-8:
            A_0[0] = 1.0/A_scale
            B_0[0] = 0.0

        for i in xrange(1, N):
            A_0[i] /= pow(i, 2)
            B_0[i] /= pow(i, 2)
        for i in xrange(2, N):
            A_2[i-2] /= pow(i, 2)
            B_2[i-2] /= pow(i, 2)
        for i in xrange(4, N):
            A_4[i-4] /= pow(i, 2)
        for i in xrange(1, N-2):
            B_m2[i] /= pow(i, 2)

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

def LU_Helmholtz_3D(A, B, np.int64_t axis,
                    np.ndarray[real_t, ndim=3] A_scale,
                    np.ndarray[real_t, ndim=3] B_scale,
                    bint neumann,
                    np.ndarray[real_t, ndim=3] d0,
                    np.ndarray[real_t, ndim=3] d1,
                    np.ndarray[real_t, ndim=3] d2,
                    np.ndarray[real_t, ndim=3] L):
    cdef:
        unsigned int i, j, k

    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                LU_Helmholtz_1D(A, B,
                                A_scale[0,j,k],
                                B_scale[0,j,k],
                                neumann,
                                d0[:,j,k],
                                d1[:,j,k],
                                d2[:,j,k],
                                L [:,j,k])

    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                LU_Helmholtz_1D(A, B,
                                A_scale[i, 0, k],
                                B_scale[i, 0, k],
                                neumann,
                                d0[i,:,k],
                                d1[i,:,k],
                                d2[i,:,k],
                                L [i,:,k])

    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                LU_Helmholtz_1D(A, B,
                                A_scale[i, j, 0],
                                B_scale[i, j, 0],
                                neumann,
                                d0[i, j, :],
                                d1[i, j, :],
                                d2[i, j, :],
                                L [i, j, :])

def LU_Helmholtz_2D(A, B, np.int64_t axis,
                    np.ndarray[real_t, ndim=2] A_scale,
                    np.ndarray[real_t, ndim=2] B_scale,
                    bint neumann,
                    np.ndarray[real_t, ndim=2] d0,
                    np.ndarray[real_t, ndim=2] d1,
                    np.ndarray[real_t, ndim=2] d2,
                    np.ndarray[real_t, ndim=2] L):
    cdef:
        unsigned int i

    if axis == 0:
        for i in range(d0.shape[1]):
            LU_Helmholtz_1D(A, B,
                            A_scale[0, i],
                            B_scale[0, i],
                            neumann,
                            d0[:, i],
                            d1[:, i],
                            d2[:, i],
                            L [:, i])

    elif axis == 1:
        for i in range(d0.shape[0]):
            LU_Helmholtz_1D(A, B,
                            A_scale[i, 0],
                            B_scale[i, 0],
                            neumann,
                            d0[i, :],
                            d1[i, :],
                            d2[i, :],
                            L [i, :])

def Solve_Helmholtz(b, u, neumann, d0, d1, d2, L, axis):
    n = d0.ndim
    if n == 1:
        Solve_Helmholtz_1D(b, u, neumann, d0, d1, d2, L)
    elif n == 2:
        Solve_Helmholtz_2D_ptr(axis, b, u, neumann, d0, d1, d2, L)
    elif n == 3:
        Solve_Helmholtz_3D_ptr(axis, b, u, neumann, d0, d1, d2, L)

def Solve_Helmholtz_1D(T[::1] fk,
                       T[::1] u_hat,
                       bint neumann,
                       real_t[::1] d0,
                       real_t[::1] d1,
                       real_t[::1] d2,
                       real_t[::1] L):
    cdef:
        T* fk_ptr = &fk[0]
        T* u_hat_ptr = &u_hat[0]
        vector[T] y
        int N = d0.shape[0]-2
    y.resize(N)
    Solve_Helmholtz_1D_ptr(fk_ptr, u_hat_ptr, neumann, &d0[0], &d1[0], &d2[0], &L[0], &y[0], N, 1)

cdef void Solve_Helmholtz_1D_ptr(T* fk,
                                 T* u_hat,
                                 bint neumann,
                                 real_t* d0,
                                 real_t* d1,
                                 real_t* d2,
                                 real_t* L,
                                 T* y,
                                 int N,
                                 int strides) nogil:
    cdef:
        int i, j, st, ii, jj
        T sum_even = 0.0
        T sum_odd = 0.0

    st = strides
    y[0] = fk[0]
    y[1] = fk[st]
    for i in xrange(2, N):
        y[i] = fk[i*st] - L[(i-2)*st]*y[i-2]

    u_hat[(N-1)*st] = y[N-1] / d0[(N-1)*st]
    u_hat[(N-2)*st] = y[N-2] / d0[(N-2)*st]
    u_hat[(N-3)*st] = (y[N-3] - d1[(N-3)*st]*u_hat[(N-1)*st]) / d0[(N-3)*st]
    u_hat[(N-4)*st] = (y[N-4] - d1[(N-4)*st]*u_hat[(N-2)*st]) / d0[(N-4)*st]
    for i in xrange(N-5, -1, -1):
        ii = i*st
        u_hat[ii] = y[i] - d1[ii]*u_hat[(i+2)*st]
        if i % 2 == 0:
            sum_even += u_hat[(i+4)*st]
            u_hat[ii] -= d2[ii]*sum_even
        else:
            sum_odd += u_hat[(i+4)*st]
            u_hat[ii] -= d2[ii]*sum_odd
        u_hat[ii]/=d0[ii]

    if neumann:
        if (d0[0]-1.0)*(d0[0]-1.0) < 1e-16:
            u_hat[0] = 0.0

        for i in xrange(1, N):
            u_hat[i*st] /= (i*i)

def Solve_Helmholtz_3D_ptr(np.int64_t axis,
                           T[:,:,::1] fk,
                           T[:,:,::1] u_hat,
                           bint neumann,
                           real_t[:,:,::1] d0,
                           real_t[:,:,::1] d1,
                           real_t[:,:,::1] d2,
                           real_t[:,:,::1] L):
    cdef:
        T* fk_ptr
        T* u_hat_ptr
        real_t* d0_ptr
        real_t* d1_ptr
        real_t* d2_ptr
        real_t* L_ptr
        vector[T] y
        int i, j, k, strides, N

    strides = fk.strides[axis]/fk.itemsize
    N = d0.shape[axis] - 2
    y.resize(N)
    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                fk_ptr = &fk[0,j,k]
                u_hat_ptr = &u_hat[0,j,k]
                d0_ptr = &d0[0,j,k]
                d1_ptr = &d1[0,j,k]
                d2_ptr = &d2[0,j,k]
                L_ptr = &L[0,j,k]
                Solve_Helmholtz_1D_ptr(fk_ptr, u_hat_ptr, neumann, d0_ptr,
                                       d1_ptr, d2_ptr, L_ptr, &y[0], N,
                                       strides)
    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                fk_ptr = &fk[i,0,k]
                u_hat_ptr = &u_hat[i,0,k]
                d0_ptr = &d0[i,0,k]
                d1_ptr = &d1[i,0,k]
                d2_ptr = &d2[i,0,k]
                L_ptr = &L[i,0,k]
                Solve_Helmholtz_1D_ptr(fk_ptr, u_hat_ptr, neumann, d0_ptr,
                                       d1_ptr, d2_ptr, L_ptr, &y[0], N,
                                       strides)

    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                fk_ptr = &fk[i,j,0]
                u_hat_ptr = &u_hat[i,j,0]
                d0_ptr = &d0[i,j,0]
                d1_ptr = &d1[i,j,0]
                d2_ptr = &d2[i,j,0]
                L_ptr = &L[i,j,0]
                Solve_Helmholtz_1D_ptr(fk_ptr, u_hat_ptr, neumann, d0_ptr,
                                       d1_ptr, d2_ptr, L_ptr, &y[0], N,
                                       strides)

def Solve_Helmholtz_2D_ptr(np.int64_t axis,
                           T[:,::1] fk,
                           T[:,::1] u_hat,
                           bint neumann,
                           real_t[:,::1] d0,
                           real_t[:,::1] d1,
                           real_t[:,::1] d2,
                           real_t[:,::1] L):
    cdef:
        T* fk_ptr
        T* u_hat_ptr
        real_t* d0_ptr
        real_t* d1_ptr
        real_t* d2_ptr
        real_t* L_ptr
        vector[T] y
        int i, j, strides, N

    strides = fk.strides[axis]/fk.itemsize
    N = d0.shape[axis] - 2
    y.resize(N)
    if axis == 0:
        for j in range(d0.shape[1]):
            fk_ptr = &fk[0,j]
            u_hat_ptr = &u_hat[0,j]
            d0_ptr = &d0[0,j]
            d1_ptr = &d1[0,j]
            d2_ptr = &d2[0,j]
            L_ptr = &L[0,j]
            Solve_Helmholtz_1D_ptr(fk_ptr, u_hat_ptr, neumann, d0_ptr,
                                    d1_ptr, d2_ptr, L_ptr, &y[0], N,
                                    strides)
    elif axis == 1:
        for i in range(d0.shape[0]):
            fk_ptr = &fk[i,0]
            u_hat_ptr = &u_hat[i,0]
            d0_ptr = &d0[i,0]
            d1_ptr = &d1[i,0]
            d2_ptr = &d2[i,0]
            L_ptr = &L[i,0]
            Solve_Helmholtz_1D_ptr(fk_ptr, u_hat_ptr, neumann, d0_ptr,
                                    d1_ptr, d2_ptr, L_ptr, &y[0], N,
                                    strides)

def LU_Biharmonic(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                  bill, bil, bii, biu, biuu, u0, u1,
                  u2, l0, l1, axis):
    if l1.ndim == 2:
        LU_Biharmonic_1D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                         bill, bil, bii, biu, biuu, u0, u1,
                         u2, l0, l1)
    elif l1.ndim == 3:
        LU_Biharmonic_2D_n(axis, a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                           bill, bil, bii, biu, biuu, u0, u1,
                           u2, l0, l1)
    elif l1.ndim == 4:
        LU_Biharmonic_3D_n(axis, a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                           bill, bil, bii, biu, biuu, u0, u1,
                           u2, l0, l1)

def LU_Biharmonic_1D(np.float_t a,
                     np.float_t b,
                     np.float_t c,
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
                     # Three upper and two lower diagonals of LU decomposition
                     np.ndarray[real_t, ndim=2] u0,
                     np.ndarray[real_t, ndim=2] u1,
                     np.ndarray[real_t, ndim=2] u2,
                     np.ndarray[real_t, ndim=2] l0,
                     np.ndarray[real_t, ndim=2] l1):

    LU_oe_Biharmonic_1D(0, a, b, c, sii[::2], siu[::2], siuu[::2], ail[::2], aii[::2], aiu[::2], bill[::2], bil[::2], bii[::2], biu[::2], biuu[::2], u0[0], u1[0], u2[0], l0[0], l1[0])
    LU_oe_Biharmonic_1D(1, a, b, c, sii[1::2], siu[1::2], siuu[1::2], ail[1::2], aii[1::2], aiu[1::2], bill[1::2], bil[1::2], bii[1::2], biu[1::2], biuu[1::2], u0[1], u1[1], u2[1], l0[1], l1[1])

def LU_oe_Biharmonic_1D(bint odd,
                        np.float_t a,
                        np.float_t b,
                        np.float_t c,
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
                        # Two upper and two lower diagonals of LU decomposition
                        np.ndarray[real_t, ndim=1] u0,
                        np.ndarray[real_t, ndim=1] u1,
                        np.ndarray[real_t, ndim=1] u2,
                        np.ndarray[real_t, ndim=1] l0,
                        np.ndarray[real_t, ndim=1] l1):

    cdef:
        int i, j, kk
        long long int m, k
        real pi = np.pi
        vector[real] c0, c1, c2

    M = sii.shape[0]

    c0.resize(M)
    c1.resize(M)
    c2.resize(M)

    c0[0] = a*sii[0] + b*aii[0] + c*bii[0]
    c0[1] = a*siu[0] + b*aiu[0] + c*biu[0]
    c0[2] = a*siuu[0] + c*biuu[0]
    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
    c0[3] = m*a*pi/(6+odd+3.)
    #c0[3] = a*8./(6+odd+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(6+odd+2., 2))
    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
    c0[4] = m*a*pi/(8+odd+3.)
    #c0[4] = a*8./(8+odd+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(8+odd+2., 2))
    c1[0] = b*ail[0] + c*bil[0]
    c1[1] = a*sii[1] + b*aii[1] + c*bii[1]
    c1[2] = a*siu[1] + b*aiu[1] + c*biu[1]
    c1[3] = a*siuu[1] + c*biuu[1]
    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
    c1[4] = m*a*pi/(8+odd+3.)
    #c1[4] = a*8./(8+odd+3.)*pi*(odd+3.)*(odd+4.)*((odd+2.)*(odd+6.)+3.*pow(8+odd+2., 2))
    c2[0] = c*bill[0]
    c2[1] = b*ail[1] + c*bil[1]
    c2[2] = a*sii[2] + b*aii[2] + c*bii[2]
    c2[3] = a*siu[2] + b*aiu[2] + c*biu[2]
    c2[4] = a*siuu[2] + c*biuu[2]
    for i in xrange(5, M):
        j = 2*i+odd
        m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(j+2)*(j+2))
        c0[i] = m*a*pi/(j+3.)
        m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(j+2)*(j+2))
        c1[i] = m*a*pi/(j+3.)
        m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(j+2)*(j+2))
        c2[i] = m*a*pi/(j+3.)
        #c0[i] = a*8./(j+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(j+2., 2))
        #c1[i] = a*8./(j+3.)*pi*(odd+3.)*(odd+4.)*((odd+2)*(odd+6.)+3.*pow(j+2., 2))
        #c2[i] = a*8./(j+3.)*pi*(odd+5.)*(odd+6.)*((odd+4)*(odd+8.)+3.*pow(j+2., 2))

    u0[0] = c0[0]
    u1[0] = c0[1]
    u2[0] = c0[2]
    for kk in xrange(1, M):
        l0[kk-1] = c1[kk-1]/u0[kk-1]
        if kk < M-1:
            l1[kk-1] = c2[kk-1]/u0[kk-1]

        for i in xrange(kk, M):
            c1[i] = c1[i] - l0[kk-1]*c0[i]

        if kk < M-1:
            for i in xrange(kk, M):
                c2[i] = c2[i] - l1[kk-1]*c0[i]

        for i in xrange(kk, M):
            c0[i] = c1[i]
            c1[i] = c2[i]

        if kk < M-2:
            c2[kk] = c*bill[kk]
            c2[kk+1] = b*ail[kk+1] + c*bil[kk+1]
            c2[kk+2] = a*sii[kk+2] + b*aii[kk+2] + c*bii[kk+2]
            if kk < M-3:
                c2[kk+3] = a*siu[kk+2] + b*aiu[kk+2] + c*biu[kk+2]
            if kk < M-4:
                c2[kk+4] = a*siuu[kk+2] + c*biuu[kk+2]
            if kk < M-5:
                k = 2*(kk+2)+odd
                for i in xrange(kk+5, M):
                    j = 2*i+odd
                    m = 8*(k+1)*(k+2)*(k*(k+4)+3*(j+2)*(j+2))
                    c2[i] = m*a*pi/(j+3.)
                    #c2[i] = a*8./(j+3.)*pi*(k+1.)*(k+2.)*(k*(k+4.)+3.*pow(j+2., 2))

        u0[kk] = c0[kk]
        if kk < M-1:
            u1[kk] = c0[kk+1]
        if kk < M-2:
            u2[kk] = c0[kk+2]

cdef ForwardBsolve_L(np.ndarray[T, ndim=1] y,
                     np.ndarray[real_t, ndim=1] l0,
                     np.ndarray[real_t, ndim=1] l1,
                     np.ndarray[T, ndim=1] fk):
    # Solve Forward Ly = f
    cdef np.intp_t i, N
    y[0] = fk[0]
    y[1] = fk[1] - l0[0]*y[0]
    N = l0.shape[0]
    for i in xrange(2, N):
        y[i] = fk[i] - l0[i-1]*y[i-1] - l1[i-2]*y[i-2]

def Biharmonic_factor_pr_3D(np.int64_t axis,
                            np.ndarray[real_t, ndim=4] a,
                            np.ndarray[real_t, ndim=4] b,
                            np.ndarray[real_t, ndim=4] l0,
                            np.ndarray[real_t, ndim=4] l1):

    cdef:
        unsigned int ii, jj

    if axis == 0:
        for ii in range(a.shape[2]):
            for jj in range(a.shape[3]):
                Biharmonic_factor_pr_1D(a[:, :, ii, jj],
                                        b[:, :, ii, jj],
                                        l0[:, :, ii, jj],
                                        l1[:, :, ii, jj])
    elif axis == 1:
        for ii in range(a.shape[1]):
            for jj in range(a.shape[3]):
                Biharmonic_factor_pr_1D(a[:, ii, :, jj],
                                        b[:, ii, :, jj],
                                        l0[:, ii, :, jj],
                                        l1[:, ii, :, jj])

    elif axis == 2:
        for ii in range(a.shape[1]):
            for jj in range(a.shape[2]):
                Biharmonic_factor_pr_1D(a[:, ii, jj, :],
                                        b[:, ii, jj, :],
                                        l0[:, ii, jj, :],
                                        l1[:, ii, jj, :])

def Biharmonic_factor_pr_2D(np.int64_t axis,
                            np.ndarray[real_t, ndim=3] a,
                            np.ndarray[real_t, ndim=3] b,
                            np.ndarray[real_t, ndim=3] l0,
                            np.ndarray[real_t, ndim=3] l1):

    cdef:
        unsigned int ii

    if axis == 0:
        for ii in range(a.shape[2]):
            Biharmonic_factor_pr_1D(a[:, :, ii],
                                    b[:, :, ii],
                                    l0[:, :, ii],
                                    l1[:, :, ii])
    elif axis == 1:
        for ii in range(a.shape[1]):
            Biharmonic_factor_pr_1D(a[:, ii, :],
                                    b[:, ii, :],
                                    l0[:, ii, :],
                                    l1[:, ii, :])

def Biharmonic_factor_pr(a, b, l0, l1, axis):
    if a.ndim == 2:
        Biharmonic_factor_pr_1D(a, b, l0, l1)
    elif a.ndim == 3:
        Biharmonic_factor_pr_2D(axis, a, b, l0, l1)
    elif a.ndim == 4:
        Biharmonic_factor_pr_3D(axis, a, b, l0, l1)

def Biharmonic_factor_pr_1D(np.ndarray[real_t, ndim=2] a,
                            np.ndarray[real_t, ndim=2] b,
                            np.ndarray[real_t, ndim=2] l0,
                            np.ndarray[real_t, ndim=2] l1):

    Biharmonic_factor_oe_pr(0, a[0], b[0], l0[0], l1[0])
    Biharmonic_factor_oe_pr(1, a[1], b[1], l0[1], l1[1])

def Biharmonic_factor_oe_pr(bint odd,
                            np.ndarray[real_t, ndim=1] a,
                            np.ndarray[real_t, ndim=1] b,
                            np.ndarray[real_t, ndim=1] l0,
                            np.ndarray[real_t, ndim=1] l1):
    cdef:
        int i, j, M
        real pi = np.pi
        long long int pp, rr, k, kk

    M = l0.shape[0]+1
    k = odd
    a[0] = 8*k*(k+1)*(k+2)*(k+4)*pi
    b[0] = 24*(k+1)*(k+2)*pi
    k = 2+odd
    a[1] = 8*k*(k+1)*(k+2)*(k+4)*pi - l0[0]*a[0]
    b[1] = 24*(k+1)*(k+2)*pi - l0[0]*b[0]
    for k in xrange(2, M-3):
        kk = 2*k+odd
        pp = 8*kk*(kk+1)*(kk+2)*(kk+4)
        rr = 24*(kk+1)*(kk+2)
        a[k] = pp*pi - l0[k-1]*a[k-1] - l1[k-2]*a[k-2]
        b[k] = rr*pi - l0[k-1]*b[k-1] - l1[k-2]*b[k-2]

def Biharmonic_Solve(b, u, u0, u1, u2, l0, l1, ak, bk, a0, axis=0):
    if b.ndim == 1:
        Solve_Biharmonic_1D(b, u, u0, u1, u2, l0, l1, ak, bk, a0)
    elif b.ndim == 2:
        Solve_Biharmonic_2D_n(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0)
    elif b.ndim == 3:
        Solve_Biharmonic_3D_n(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0)

def Solve_Biharmonic_1D(np.ndarray[T, ndim=1] fk,
                        np.ndarray[T, ndim=1] uk,
                        np.ndarray[real_t, ndim=2] u0,
                        np.ndarray[real_t, ndim=2] u1,
                        np.ndarray[real_t, ndim=2] u2,
                        np.ndarray[real_t, ndim=2] l0,
                        np.ndarray[real_t, ndim=2] l1,
                        np.ndarray[real_t, ndim=2] a,
                        np.ndarray[real_t, ndim=2] b,
                        np.float_t ac):
    Solve_oe_Biharmonic_1D(0, fk[::2], uk[::2], u0[0], u1[0], u2[0], l0[0], l1[0], a[0], b[0], ac)
    Solve_oe_Biharmonic_1D(1, fk[1::2], uk[1::2], u0[1], u1[1], u2[1], l0[1], l1[1], a[1], b[1], ac)

cdef BackBsolve_U(int M,
                  bint odd,
                  np.ndarray[T, ndim=1] f,  # Uc = f
                  np.ndarray[T, ndim=1] uk,
                  np.ndarray[real_t, ndim=1] u0,
                  np.ndarray[real_t, ndim=1] u1,
                  np.ndarray[real_t, ndim=1] u2,
                  np.ndarray[real_t, ndim=1] l0,
                  np.ndarray[real_t, ndim=1] l1,
                  np.ndarray[real_t, ndim=1] a,
                  np.ndarray[real_t, ndim=1] b,
                  np.float_t ac):
    cdef:
        int i, j, k, kk
        T s1 = 0.0
        T s2 = 0.0

    uk[M-1] = f[M-1] / u0[M-1]
    uk[M-2] = (f[M-2] - u1[M-2]*uk[M-1]) / u0[M-2]
    uk[M-3] = (f[M-3] - u1[M-3]*uk[M-2] - u2[M-3]*uk[M-1]) / u0[M-3]

    s1 = 0.0
    s2 = 0.0
    for kk in xrange(M-4, -1, -1):
        k = 2*kk+odd
        j = k+6
        s1 += uk[kk+3]/(j+3.)
        s2 += (uk[kk+3]/(j+3.))*((j+2)*(j+2))
        uk[kk] = (f[kk] - u1[kk]*uk[kk+1] - u2[kk]*uk[kk+2] - a[kk]*ac*s1 - b[kk]*ac*s2) / u0[kk]

def Solve_oe_Biharmonic_1D(bint odd,
                           np.ndarray[T, ndim=1] fk,
                           np.ndarray[T, ndim=1] uk,
                           np.ndarray[real_t, ndim=1] u0,
                           np.ndarray[real_t, ndim=1] u1,
                           np.ndarray[real_t, ndim=1] u2,
                           np.ndarray[real_t, ndim=1] l0,
                           np.ndarray[real_t, ndim=1] l1,
                           np.ndarray[real_t, ndim=1] a,
                           np.ndarray[real_t, ndim=1] b,
                           np.float_t ac):
    """
    Solve (aS+b*A+cB)x = f, where S, A and B are 4th order Laplace, stiffness and mass matrices of Shen with Dirichlet BC
    """
    cdef:
        unsigned int M
        np.ndarray[T, ndim=1] y = np.zeros(u0.shape[0], dtype=fk.dtype)

    M = u0.shape[0]
    ForwardBsolve_L(y, l0, l1, fk)

    # Solve Backward U u = y
    BackBsolve_U(M, odd, y, uk, u0, u1, u2, l0, l1, a, b, ac)

# This one is fastest by far
@cython.cdivision(True)
def Solve_Biharmonic_3D_n(np.int64_t axis,
                          np.ndarray[T, ndim=3, mode='c'] fk,
                          np.ndarray[T, ndim=3, mode='c'] uk,
                          np.ndarray[real_t, ndim=4, mode='c'] u0,
                          np.ndarray[real_t, ndim=4, mode='c'] u1,
                          np.ndarray[real_t, ndim=4, mode='c'] u2,
                          np.ndarray[real_t, ndim=4, mode='c'] l0,
                          np.ndarray[real_t, ndim=4, mode='c'] l1,
                          np.ndarray[real_t, ndim=4, mode='c'] a,
                          np.ndarray[real_t, ndim=4, mode='c'] b,
                          real_t a0):

    cdef:
        int i, j, k, kk, m, M, ke, ko, jj, je, jo
        np.float_t ac
        np.ndarray[T, ndim=2, mode='c'] s1
        np.ndarray[T, ndim=2, mode='c'] s2
        np.ndarray[T, ndim=2, mode='c'] o1
        np.ndarray[T, ndim=2, mode='c'] o2
        np.ndarray[T, ndim=3, mode='c'] y = np.zeros((fk.shape[0], fk.shape[1], fk.shape[2]), dtype=fk.dtype)

    if axis == 0:
        s1 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        s2 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        o1 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        o2 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)

        M = u0.shape[1]
        for j in range(fk.shape[1]):
            for k in range(fk.shape[2]):
                y[0, j, k] = fk[0, j, k]
                y[1, j, k] = fk[1, j, k]
                y[2, j, k] = fk[2, j, k] - l0[0, 0, j, k]*y[0, j, k]
                y[3, j, k] = fk[3, j, k] - l0[1, 0, j, k]*y[1, j, k]

        for i in xrange(2, M):
            ke = 2*i
            ko = ke+1
            for j in range(fk.shape[1]):
                for k in range(fk.shape[2]):
                    y[ko, j, k] = fk[ko, j, k] - l0[1, i-1, j, k]*y[ko-2, j, k] - l1[1, i-2, j, k]*y[ko-4, j, k]
                    y[ke, j, k] = fk[ke, j, k] - l0[0, i-1, j, k]*y[ke-2, j, k] - l1[0, i-2, j, k]*y[ke-4, j, k]

        ke = 2*(M-1)
        ko = ke+1
        for j in range(fk.shape[1]):
            for k in range(fk.shape[2]):
                uk[ke, j, k] = y[ke, j, k] / u0[0, M-1, j, k]
                uk[ko, j, k] = y[ko, j, k] / u0[1, M-1, j, k]

        ke = 2*(M-2)
        ko = ke+1
        for j in range(fk.shape[1]):
            for k in range(fk.shape[2]):
                uk[ke, j, k] = (y[ke, j, k] - u1[0, M-2, j, k]*uk[ke+2, j, k]) / u0[0, M-2, j, k]
                uk[ko, j, k] = (y[ko, j, k] - u1[1, M-2, j, k]*uk[ko+2, j, k]) / u0[1, M-2, j, k]

        ke = 2*(M-3)
        ko = ke+1
        for j in range(fk.shape[1]):
            for k in range(fk.shape[2]):
                uk[ke, j, k] = (y[ke, j, k] - u1[0, M-3, j, k]*uk[ke+2, j, k] - u2[0, M-3, j, k]*uk[ke+4, j, k]) / u0[0, M-3, j, k]
                uk[ko, j, k] = (y[ko, j, k] - u1[1, M-3, j, k]*uk[ko+2, j, k] - u2[1, M-3, j, k]*uk[ko+4, j, k]) / u0[1, M-3, j, k]

        for kk in xrange(M-4, -1, -1):
            ke = 2*kk
            ko = ke+1
            je = ke+6
            jo = ko+6
            for j in range(fk.shape[1]):
                for k in range(fk.shape[2]):
                    ac = a0
                    s1[j, k] += uk[je, j, k]/(je+3.)
                    s2[j, k] += (uk[je, j, k]/(je+3.))*((je+2.)*(je+2.))
                    uk[ke, j, k] = (y[ke, j, k] - u1[0, kk, j, k]*uk[ke+2, j, k] - u2[0, kk, j, k]*uk[ke+4, j, k] - a[0, kk, j, k]*ac*s1[j, k] - b[0, kk, j, k]*ac*s2[j, k]) / u0[0, kk, j, k]
                    o1[j, k] += uk[jo, j, k]/(jo+3.)
                    o2[j, k] += (uk[jo, j, k]/(jo+3.))*((jo+2.)*(jo+2.))
                    uk[ko, j, k] = (y[ko, j, k] - u1[1, kk, j, k]*uk[ko+2, j, k] - u2[1, kk, j, k]*uk[ko+4, j, k] - a[1, kk, j, k]*ac*o1[j, k] - b[1, kk, j, k]*ac*o2[j, k]) / u0[1, kk, j, k]

    elif axis == 1:
        s1 = np.zeros((fk.shape[0], fk.shape[2]), dtype=fk.dtype)
        s2 = np.zeros((fk.shape[0], fk.shape[2]), dtype=fk.dtype)
        o1 = np.zeros((fk.shape[0], fk.shape[2]), dtype=fk.dtype)
        o2 = np.zeros((fk.shape[0], fk.shape[2]), dtype=fk.dtype)

        M = u0.shape[2]
        for j in range(fk.shape[0]):
            for k in range(fk.shape[2]):
                y[j, 0, k] = fk[j, 0, k]
                y[j, 1, k] = fk[j, 1, k]
                y[j, 2, k] = fk[j, 2, k] - l0[0, j, 0, k]*y[j, 0, k]
                y[j, 3, k] = fk[j, 3, k] - l0[1, j, 0, k]*y[j, 1, k]

            for i in xrange(2, M):
                ke = 2*i
                ko = ke+1
                for k in range(fk.shape[2]):
                    y[j, ko, k] = fk[j, ko, k] - l0[1, j, i-1, k]*y[j, ko-2, k] - l1[1, j, i-2, k]*y[j, ko-4, k]
                    y[j, ke, k] = fk[j, ke, k] - l0[0, j, i-1, k]*y[j, ke-2, k] - l1[0, j, i-2, k]*y[j, ke-4, k]

            ke = 2*(M-1)
            ko = ke+1
            for k in range(fk.shape[2]):
                uk[j, ke, k] = y[j, ke, k] / u0[0, j, M-1, k]
                uk[j, ko, k] = y[j, ko, k] / u0[1, j, M-1, k]

            ke = 2*(M-2)
            ko = ke+1
            for k in range(fk.shape[2]):
                uk[j, ke, k] = (y[j, ke, k] - u1[0, j, M-2, k]*uk[j, ke+2, k]) / u0[0, j, M-2, k]
                uk[j, ko, k] = (y[j, ko, k] - u1[1, j, M-2, k]*uk[j, ko+2, k]) / u0[1, j, M-2, k]

            ke = 2*(M-3)
            ko = ke+1
            for k in range(fk.shape[2]):
                uk[j, ke, k] = (y[j, ke, k] - u1[0, j, M-3, k]*uk[j, ke+2, k] - u2[0, j, M-3, k]*uk[j, ke+4, k]) / u0[0, j, M-3, k]
                uk[j, ko, k] = (y[j, ko, k] - u1[1, j, M-3, k]*uk[j, ko+2, k] - u2[1, j, M-3, k]*uk[j, ko+4, k]) / u0[1, j, M-3, k]

            for kk in xrange(M-4, -1, -1):
                ke = 2*kk
                ko = ke+1
                je = ke+6
                jo = ko+6
                for k in range(fk.shape[2]):
                    ac = a0
                    s1[j, k] += uk[j, je, k]/(je+3.)
                    s2[j, k] += (uk[j, je, k]/(je+3.))*((je+2.)*(je+2.))
                    uk[j, ke, k] = (y[j, ke, k] - u1[0, j, kk, k]*uk[j, ke+2, k] - u2[0, j, kk, k]*uk[j, ke+4, k] - a[0, j, kk, k]*ac*s1[j, k] - b[0, j, kk, k]*ac*s2[j, k]) / u0[0, j, kk, k]
                    o1[j, k] += uk[j, jo, k]/(jo+3.)
                    o2[j, k] += (uk[j, jo, k]/(jo+3.))*((jo+2.)*(jo+2.))
                    uk[j, ko, k] = (y[j, ko, k] - u1[1, j, kk, k]*uk[j, ko+2, k] - u2[1, j, kk, k]*uk[j, ko+4, k] - a[1, j, kk, k]*ac*o1[j, k] - b[1, j, kk, k]*ac*o2[j, k]) / u0[1, j, kk, k]


    elif axis == 2:
        s1 = np.zeros((fk.shape[0], fk.shape[1]), dtype=fk.dtype)
        s2 = np.zeros((fk.shape[0], fk.shape[1]), dtype=fk.dtype)
        o1 = np.zeros((fk.shape[0], fk.shape[1]), dtype=fk.dtype)
        o2 = np.zeros((fk.shape[0], fk.shape[1]), dtype=fk.dtype)

        M = u0.shape[3]
        for j in range(fk.shape[0]):
            for k in range(fk.shape[1]):
                y[j, k, 0] = fk[j, k, 0]
                y[j, k, 1] = fk[j, k, 1]
                y[j, k, 2] = fk[j, k, 2] - l0[0, j, k, 0]*y[j, k, 0]
                y[j, k, 3] = fk[j, k, 3] - l0[1, j, k, 0]*y[j, k, 1]

                for i in xrange(2, M):
                    ke = 2*i
                    ko = ke+1
                    y[j, k, ko] = fk[j, k, ko] - l0[1, j, k, i-1]*y[j, k, ko-2] - l1[1, j, k, i-2]*y[j, k, ko-4]
                    y[j, k, ke] = fk[j, k, ke] - l0[0, j, k, i-1]*y[j, k, ke-2] - l1[0, j, k, i-2]*y[j, k, ke-4]

                ke = 2*(M-1)
                ko = ke+1
                uk[j, k, ke] = y[j, k, ke] / u0[0, j, k, M-1]
                uk[j, k, ko] = y[j, k, ko] / u0[1, j, k, M-1]

                ke = 2*(M-2)
                ko = ke+1
                uk[j, k, ke] = (y[j, k, ke] - u1[0, j, k, M-2]*uk[j, k, ke+2]) / u0[0, j, k, M-2]
                uk[j, k, ko] = (y[j, k, ko] - u1[1, j, k, M-2]*uk[j, k, ko+2]) / u0[1, j, k, M-2]

                ke = 2*(M-3)
                ko = ke+1
                uk[j, k, ke] = (y[j, k, ke] - u1[0, j, k, M-3]*uk[j, k, ke+2] - u2[0, j, k, M-3]*uk[j, k, ke+4]) / u0[0, j, k, M-3]
                uk[j, k, ko] = (y[j, k, ko] - u1[1, j, k, M-3]*uk[j, k, ko+2] - u2[1, j, k, M-3]*uk[j, k, ko+4]) / u0[1, j, k, M-3]

                for kk in xrange(M-4, -1, -1):
                    ke = 2*kk
                    ko = ke+1
                    je = ke+6
                    jo = ko+6
                    ac = a0
                    s1[j, k] += uk[j, k, je]/(je+3.)
                    s2[j, k] += (uk[j, k, je]/(je+3.))*((je+2.)*(je+2.))
                    uk[j, k, ke] = (y[j, k, ke] - u1[0, j, k, kk]*uk[j, k, ke+2] - u2[0, j, k, kk]*uk[j, k, ke+4] - a[0, j, k, kk]*ac*s1[j, k] - b[0, j, k, kk]*ac*s2[j, k]) / u0[0, j, k, kk]
                    o1[j, k] += uk[j, k, jo]/(jo+3.)
                    o2[j, k] += (uk[j, k, jo]/(jo+3.))*((jo+2.)*(jo+2.))
                    uk[j, k, ko] = (y[j, k, ko] - u1[1, j, k, kk]*uk[j, k, ko+2] - u2[1, j, k, kk]*uk[j, k, ko+4] - a[1, j, k, kk]*ac*o1[j, k] - b[1, j, k, kk]*ac*o2[j, k]) / u0[1, j, k, kk]

@cython.cdivision(True)
def Solve_Biharmonic_2D_n(np.int64_t axis,
                          np.ndarray[T, ndim=2, mode='c'] fk,
                          np.ndarray[T, ndim=2, mode='c'] uk,
                          np.ndarray[real_t, ndim=3, mode='c'] u0,
                          np.ndarray[real_t, ndim=3, mode='c'] u1,
                          np.ndarray[real_t, ndim=3, mode='c'] u2,
                          np.ndarray[real_t, ndim=3, mode='c'] l0,
                          np.ndarray[real_t, ndim=3, mode='c'] l1,
                          np.ndarray[real_t, ndim=3, mode='c'] a,
                          np.ndarray[real_t, ndim=3, mode='c'] b,
                          real_t a0):

    cdef:
        int i, j, k, kk, m, M, ke, ko, jj, je, jo
        np.float_t ac
        np.ndarray[T, ndim=1, mode='c'] s1
        np.ndarray[T, ndim=1, mode='c'] s2
        np.ndarray[T, ndim=1, mode='c'] o1
        np.ndarray[T, ndim=1, mode='c'] o2
        np.ndarray[T, ndim=2, mode='c'] y = np.zeros((fk.shape[0], fk.shape[1]), dtype=fk.dtype)

    if axis == 0:
        s1 = np.zeros(fk.shape[1], dtype=fk.dtype)
        s2 = np.zeros(fk.shape[1], dtype=fk.dtype)
        o1 = np.zeros(fk.shape[1], dtype=fk.dtype)
        o2 = np.zeros(fk.shape[1], dtype=fk.dtype)

        M = u0.shape[1]
        for j in range(fk.shape[1]):
            y[0, j] = fk[0, j]
            y[1, j] = fk[1, j]
            y[2, j] = fk[2, j] - l0[0, 0, j]*y[0, j]
            y[3, j] = fk[3, j] - l0[1, 0, j]*y[1, j]

        for i in xrange(2, M):
            ke = 2*i
            ko = ke+1
            for j in range(fk.shape[1]):
                y[ko, j] = fk[ko, j] - l0[1, i-1, j]*y[ko-2, j] - l1[1, i-2, j]*y[ko-4, j]
                y[ke, j] = fk[ke, j] - l0[0, i-1, j]*y[ke-2, j] - l1[0, i-2, j]*y[ke-4, j]

        ke = 2*(M-1)
        ko = ke+1
        for j in range(fk.shape[1]):
            uk[ke, j] = y[ke, j] / u0[0, M-1, j]
            uk[ko, j] = y[ko, j] / u0[1, M-1, j]

        ke = 2*(M-2)
        ko = ke+1
        for j in range(fk.shape[1]):
            uk[ke, j] = (y[ke, j] - u1[0, M-2, j]*uk[ke+2, j]) / u0[0, M-2, j]
            uk[ko, j] = (y[ko, j] - u1[1, M-2, j]*uk[ko+2, j]) / u0[1, M-2, j]

        ke = 2*(M-3)
        ko = ke+1
        for j in range(fk.shape[1]):
            uk[ke, j] = (y[ke, j] - u1[0, M-3, j]*uk[ke+2, j] - u2[0, M-3, j]*uk[ke+4, j]) / u0[0, M-3, j]
            uk[ko, j] = (y[ko, j] - u1[1, M-3, j]*uk[ko+2, j] - u2[1, M-3, j]*uk[ko+4, j]) / u0[1, M-3, j]

        for kk in xrange(M-4, -1, -1):
            ke = 2*kk
            ko = ke+1
            je = ke+6
            jo = ko+6
            for j in range(fk.shape[1]):
                ac = a0
                s1[j] += uk[je, j]/(je+3.)
                s2[j] += (uk[je, j]/(je+3.))*((je+2.)*(je+2.))
                uk[ke, j] = (y[ke, j] - u1[0, kk, j]*uk[ke+2, j] - u2[0, kk, j]*uk[ke+4, j] - a[0, kk, j]*ac*s1[j] - b[0, kk, j]*ac*s2[j]) / u0[0, kk, j]
                o1[j] += uk[jo, j]/(jo+3.)
                o2[j] += (uk[jo, j]/(jo+3.))*((jo+2.)*(jo+2.))
                uk[ko, j] = (y[ko, j] - u1[1, kk, j]*uk[ko+2, j] - u2[1, kk, j]*uk[ko+4, j] - a[1, kk, j]*ac*o1[j] - b[1, kk, j]*ac*o2[j]) / u0[1, kk, j]

    elif axis == 1:
        s1 = np.zeros(fk.shape[0], dtype=fk.dtype)
        s2 = np.zeros(fk.shape[0], dtype=fk.dtype)
        o1 = np.zeros(fk.shape[0], dtype=fk.dtype)
        o2 = np.zeros(fk.shape[0], dtype=fk.dtype)

        M = u0.shape[2]
        for j in range(fk.shape[0]):
            y[j, 0] = fk[j, 0]
            y[j, 1] = fk[j, 1]
            y[j, 2] = fk[j, 2] - l0[0, j, 0]*y[j, 0]
            y[j, 3] = fk[j, 3] - l0[1, j, 0]*y[j, 1]

            for i in xrange(2, M):
                ke = 2*i
                ko = ke+1
                y[j, ko] = fk[j, ko] - l0[1, j, i-1]*y[j, ko-2] - l1[1, j, i-2]*y[j, ko-4]
                y[j, ke] = fk[j, ke] - l0[0, j, i-1]*y[j, ke-2] - l1[0, j, i-2]*y[j, ke-4]

            ke = 2*(M-1)
            ko = ke+1
            uk[j, ke] = y[j, ke] / u0[0, j, M-1]
            uk[j, ko] = y[j, ko] / u0[1, j, M-1]

            ke = 2*(M-2)
            ko = ke+1
            uk[j, ke] = (y[j, ke] - u1[0, j, M-2]*uk[j, ke+2]) / u0[0, j, M-2]
            uk[j, ko] = (y[j, ko] - u1[1, j, M-2]*uk[j, ko+2]) / u0[1, j, M-2]

            ke = 2*(M-3)
            ko = ke+1
            uk[j, ke] = (y[j, ke] - u1[0, j, M-3]*uk[j, ke+2] - u2[0, j, M-3]*uk[j, ke+4]) / u0[0, j, M-3]
            uk[j, ko] = (y[j, ko] - u1[1, j, M-3]*uk[j, ko+2] - u2[1, j, M-3]*uk[j, ko+4]) / u0[1, j, M-3]

            for kk in xrange(M-4, -1, -1):
                ke = 2*kk
                ko = ke+1
                je = ke+6
                jo = ko+6
                ac = a0
                s1[j] += uk[j, je]/(je+3.)
                s2[j] += (uk[j, je]/(je+3.))*((je+2.)*(je+2.))
                uk[j, ke] = (y[j, ke] - u1[0, j, kk]*uk[j, ke+2] - u2[0, j, kk]*uk[j, ke+4] - a[0, j, kk]*ac*s1[j] - b[0, j, kk]*ac*s2[j]) / u0[0, j, kk]
                o1[j] += uk[j, jo]/(jo+3.)
                o2[j] += (uk[j, jo]/(jo+3.))*((jo+2.)*(jo+2.))
                uk[j, ko] = (y[j, ko] - u1[1, j, kk]*uk[j, ko+2] - u2[1, j, kk]*uk[j, ko+4] - a[1, j, kk]*ac*o1[j] - b[1, j, kk]*ac*o2[j]) / u0[1, j, kk]

@cython.cdivision(True)
#@cython.linetrace(True)
#@cython.binding(True)
def LU_Biharmonic_3D_n(np.int64_t axis,
                       real_t alfa,
                       np.ndarray[real_t, ndim=3] beta,
                       np.ndarray[real_t, ndim=3] ceta,
                       # 3 upper diagonals of SBB
                       np.ndarray[real_t, ndim=1, mode='c'] sii,
                       np.ndarray[real_t, ndim=1, mode='c'] siu,
                       np.ndarray[real_t, ndim=1, mode='c'] siuu,
                       # All 3 diagonals of ABB
                       np.ndarray[real_t, ndim=1, mode='c'] ail,
                       np.ndarray[real_t, ndim=1, mode='c'] aii,
                       np.ndarray[real_t, ndim=1, mode='c'] aiu,
                       # All 5 diagonals of BBB
                       np.ndarray[real_t, ndim=1, mode='c'] bill,
                       np.ndarray[real_t, ndim=1, mode='c'] bil,
                       np.ndarray[real_t, ndim=1, mode='c'] bii,
                       np.ndarray[real_t, ndim=1, mode='c'] biu,
                       np.ndarray[real_t, ndim=1, mode='c'] biuu,
                       np.ndarray[real_t, ndim=4, mode='c'] u0,
                       np.ndarray[real_t, ndim=4, mode='c'] u1,
                       np.ndarray[real_t, ndim=4, mode='c'] u2,
                       np.ndarray[real_t, ndim=4, mode='c'] l0,
                       np.ndarray[real_t, ndim=4, mode='c'] l1):
    cdef:
        unsigned int ii, jj, N1, N2, odd, i, j, k, kk, M, ll
        long long int m, n, p, dd, w0
        double a, b, c, pp
        double pi = np.pi
        vector[double] c0, c1, c2
        #double* pc0, pc1, pc2
        #np.ndarray[real_t, ndim=1] c0 = np.zeros(sii.shape[0]//2)
        #np.ndarray[real_t, ndim=1] c1 = np.zeros(sii.shape[0]//2)
        #np.ndarray[real_t, ndim=1] c2 = np.zeros(sii.shape[0]//2)

    M = sii.shape[0]//2

    c0.resize(M)
    c1.resize(M)
    c2.resize(M)

    if axis == 0:
        N1 = beta.shape[1]
        N2 = beta.shape[2]

        for j in xrange(N1):
            for k in xrange(N2):
                a = alfa
                b = beta[0, j, k]
                c = ceta[0, j, k]
                for odd in xrange(2):
                    c0[0] = a*sii[odd] + b*aii[odd] + c*bii[odd]
                    c0[1] = a*siu[odd] + b*aiu[odd] + c*biu[odd]
                    c0[2] = a*siuu[odd] + c*biuu[odd]
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
                    c0[3] = m*a*pi/(6+odd+3.)
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
                    c0[4] = m*a*pi/(8+odd+3.)

                    c1[0] = b*ail[odd] + c*bil[odd]
                    c1[1] = a*sii[2+odd] + b*aii[2+odd] + c*bii[2+odd]
                    c1[2] = a*siu[2+odd] + b*aiu[2+odd] + c*biu[2+odd]
                    c1[3] = a*siuu[2+odd] + c*biuu[2+odd]
                    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
                    c1[4] = m*a*pi/(8+odd+3.)

                    c2[0] = c*bill[odd]
                    c2[1] = b*ail[2+odd] + c*bil[2+odd]
                    c2[2] = a*sii[4+odd] + b*aii[4+odd] + c*bii[4+odd]
                    c2[3] = a*siu[4+odd] + b*aiu[4+odd] + c*biu[4+odd]
                    c2[4] = a*siuu[4+odd] + c*biuu[4+odd]

                    for i in xrange(5, M):
                        p = 2*i+odd
                        pp = pi/(p+3.)
                        m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(p+2)*(p+2))
                        c0[i] = m*a*pp
                        m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(p+2)*(p+2))
                        c1[i] = m*a*pp
                        m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(p+2)*(p+2))
                        c2[i] = m*a*pp

                    u0[odd, 0, j, k] = c0[0]
                    u1[odd, 0, j, k] = c0[1]
                    u2[odd, 0, j, k] = c0[2]
                    for kk in xrange(1, M):
                        l0[odd, kk-1, j, k] = c1[kk-1]/u0[odd, kk-1, j, k]
                        if kk < M-1:
                            l1[odd, kk-1, j, k] = c2[kk-1]/u0[odd, kk-1, j, k]

                        for i in xrange(kk, M):
                            c1[i] -= l0[odd, kk-1, j, k]*c0[i]

                        if kk < M-1:
                            for i in xrange(kk, M):
                                c2[i] -= l1[odd, kk-1, j, k]*c0[i]

                        #for i in xrange(kk, M):
                        #    c0[i] = c1[i]
                        #    c1[i] = c2[i]
                        copy(c1.begin()+kk, c1.end(), c0.begin()+kk)
                        copy(c2.begin()+kk, c2.end(), c1.begin()+kk)

                        if kk < M-2:
                            ll = 2*kk+odd
                            c2[kk] = c*bill[ll]
                            c2[kk+1] = b*ail[ll+2] + c*bil[ll+2]
                            c2[kk+2] = a*sii[ll+4] + b*aii[ll+4] + c*bii[ll+4]
                            if kk < M-3:
                                c2[kk+3] = a*siu[ll+4] + b*aiu[ll+4] + c*biu[ll+4]
                            if kk < M-4:
                                c2[kk+4] = a*siuu[ll+4] + c*biuu[ll+4]
                            if kk < M-5:
                                n = 2*(kk+2)+odd
                                dd = 8*(n+1)*(n+2)
                                w0 = dd*n*(n+4)
                                for i in xrange(kk+5, M):
                                    p = 2*i+odd
                                    c2[i] = (w0 + dd*3*(p+2)*(p+2))*a*pi/(p+3.)

                        u0[odd, kk, j, k] = c0[kk]
                        if kk < M-1:
                            u1[odd, kk, j, k] = c0[kk+1]
                        if kk < M-2:
                            u2[odd, kk, j, k] = c0[kk+2]

    elif axis == 1:
        N1 = beta.shape[0]
        N2 = beta.shape[2]

        for j in xrange(N1):
            for k in xrange(N2):
                a = alfa
                b = beta[j, 0, k]
                c = ceta[j, 0, k]
                for odd in xrange(2):
                    c0[0] = a*sii[odd] + b*aii[odd] + c*bii[odd]
                    c0[1] = a*siu[odd] + b*aiu[odd] + c*biu[odd]
                    c0[2] = a*siuu[odd] + c*biuu[odd]
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
                    c0[3] = m*a*pi/(6+odd+3.)
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
                    c0[4] = m*a*pi/(8+odd+3.)

                    c1[0] = b*ail[odd] + c*bil[odd]
                    c1[1] = a*sii[2+odd] + b*aii[2+odd] + c*bii[2+odd]
                    c1[2] = a*siu[2+odd] + b*aiu[2+odd] + c*biu[2+odd]
                    c1[3] = a*siuu[2+odd] + c*biuu[2+odd]
                    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
                    c1[4] = m*a*pi/(8+odd+3.)

                    c2[0] = c*bill[odd]
                    c2[1] = b*ail[2+odd] + c*bil[2+odd]
                    c2[2] = a*sii[4+odd] + b*aii[4+odd] + c*bii[4+odd]
                    c2[3] = a*siu[4+odd] + b*aiu[4+odd] + c*biu[4+odd]
                    c2[4] = a*siuu[4+odd] + c*biuu[4+odd]

                    for i in xrange(5, M):
                        p = 2*i+odd
                        pp = pi/(p+3.)
                        m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(p+2)*(p+2))
                        c0[i] = m*a*pp
                        m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(p+2)*(p+2))
                        c1[i] = m*a*pp
                        m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(p+2)*(p+2))
                        c2[i] = m*a*pp

                    u0[odd, j, 0, k] = c0[0]
                    u1[odd, j, 0, k] = c0[1]
                    u2[odd, j, 0, k] = c0[2]
                    for kk in xrange(1, M):
                        l0[odd, j, kk-1, k] = c1[kk-1]/u0[odd, j, kk-1, k]
                        if kk < M-1:
                            l1[odd, j, kk-1, k] = c2[kk-1]/u0[odd, j, kk-1, k]

                        for i in xrange(kk, M):
                            c1[i] -= l0[odd, j, kk-1, k]*c0[i]

                        if kk < M-1:
                            for i in xrange(kk, M):
                                c2[i] -= l1[odd, j, kk-1, k]*c0[i]

                        #for i in xrange(kk, M):
                        #    c0[i] = c1[i]
                        #    c1[i] = c2[i]
                        copy(c1.begin()+kk, c1.end(), c0.begin()+kk)
                        copy(c2.begin()+kk, c2.end(), c1.begin()+kk)

                        if kk < M-2:
                            ll = 2*kk+odd
                            c2[kk] = c*bill[ll]
                            c2[kk+1] = b*ail[ll+2] + c*bil[ll+2]
                            c2[kk+2] = a*sii[ll+4] + b*aii[ll+4] + c*bii[ll+4]
                            if kk < M-3:
                                c2[kk+3] = a*siu[ll+4] + b*aiu[ll+4] + c*biu[ll+4]
                            if kk < M-4:
                                c2[kk+4] = a*siuu[ll+4] + c*biuu[ll+4]
                            if kk < M-5:
                                n = 2*(kk+2)+odd
                                dd = 8*(n+1)*(n+2)
                                w0 = dd*n*(n+4)
                                for i in xrange(kk+5, M):
                                    p = 2*i+odd
                                    c2[i] = (w0 + dd*3*(p+2)*(p+2))*a*pi/(p+3.)

                        u0[odd, j, kk, k] = c0[kk]
                        if kk < M-1:
                            u1[odd, j, kk, k] = c0[kk+1]
                        if kk < M-2:
                            u2[odd, j, kk, k] = c0[kk+2]

    elif axis == 2:
        N1 = beta.shape[0]
        N2 = beta.shape[1]

        for j in xrange(N1):
            for k in xrange(N2):
                a = alfa
                b = beta[j, k, 0]
                c = ceta[j, k, 0]
                for odd in xrange(2):
                    c0[0] = a*sii[odd] + b*aii[odd] + c*bii[odd]
                    c0[1] = a*siu[odd] + b*aiu[odd] + c*biu[odd]
                    c0[2] = a*siuu[odd] + c*biuu[odd]
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
                    c0[3] = m*a*pi/(6+odd+3.)
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
                    c0[4] = m*a*pi/(8+odd+3.)

                    c1[0] = b*ail[odd] + c*bil[odd]
                    c1[1] = a*sii[2+odd] + b*aii[2+odd] + c*bii[2+odd]
                    c1[2] = a*siu[2+odd] + b*aiu[2+odd] + c*biu[2+odd]
                    c1[3] = a*siuu[2+odd] + c*biuu[2+odd]
                    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
                    c1[4] = m*a*pi/(8+odd+3.)

                    c2[0] = c*bill[odd]
                    c2[1] = b*ail[2+odd] + c*bil[2+odd]
                    c2[2] = a*sii[4+odd] + b*aii[4+odd] + c*bii[4+odd]
                    c2[3] = a*siu[4+odd] + b*aiu[4+odd] + c*biu[4+odd]
                    c2[4] = a*siuu[4+odd] + c*biuu[4+odd]

                    for i in xrange(5, M):
                        p = 2*i+odd
                        pp = pi/(p+3.)
                        m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(p+2)*(p+2))
                        c0[i] = m*a*pp
                        m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(p+2)*(p+2))
                        c1[i] = m*a*pp
                        m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(p+2)*(p+2))
                        c2[i] = m*a*pp

                    u0[odd, j, k, 0] = c0[0]
                    u1[odd, j, k, 0] = c0[1]
                    u2[odd, j, k, 0] = c0[2]
                    for kk in xrange(1, M):
                        l0[odd, j, k, kk-1] = c1[kk-1]/u0[odd, j, k, kk-1]
                        if kk < M-1:
                            l1[odd, j, k, kk-1] = c2[kk-1]/u0[odd, j, k, kk-1]

                        for i in xrange(kk, M):
                            c1[i] -= l0[odd, j, k, kk-1]*c0[i]

                        if kk < M-1:
                            for i in xrange(kk, M):
                                c2[i] -= l1[odd, j, k, kk-1]*c0[i]

                        #for i in xrange(kk, M):
                        #    c0[i] = c1[i]
                        #    c1[i] = c2[i]
                        copy(c1.begin()+kk, c1.end(), c0.begin()+kk)
                        copy(c2.begin()+kk, c2.end(), c1.begin()+kk)

                        if kk < M-2:
                            ll = 2*kk+odd
                            c2[kk] = c*bill[ll]
                            c2[kk+1] = b*ail[ll+2] + c*bil[ll+2]
                            c2[kk+2] = a*sii[ll+4] + b*aii[ll+4] + c*bii[ll+4]
                            if kk < M-3:
                                c2[kk+3] = a*siu[ll+4] + b*aiu[ll+4] + c*biu[ll+4]
                            if kk < M-4:
                                c2[kk+4] = a*siuu[ll+4] + c*biuu[ll+4]
                            if kk < M-5:
                                n = 2*(kk+2)+odd
                                dd = 8*(n+1)*(n+2)
                                w0 = dd*n*(n+4)
                                for i in xrange(kk+5, M):
                                    p = 2*i+odd
                                    c2[i] = (w0 + dd*3*(p+2)*(p+2))*a*pi/(p+3.)

                        u0[odd, j, k, kk] = c0[kk]
                        if kk < M-1:
                            u1[odd, j, k, kk] = c0[kk+1]
                        if kk < M-2:
                            u2[odd, j, k, kk] = c0[kk+2]


@cython.cdivision(True)
#@cython.linetrace(True)
#@cython.binding(True)
def LU_Biharmonic_2D_n(np.int64_t axis,
                       real_t alfa,
                       np.ndarray[real_t, ndim=2] beta,
                       np.ndarray[real_t, ndim=2] ceta,
                       # 3 upper diagonals of SBB
                       np.ndarray[real_t, ndim=1, mode='c'] sii,
                       np.ndarray[real_t, ndim=1, mode='c'] siu,
                       np.ndarray[real_t, ndim=1, mode='c'] siuu,
                       # All 3 diagonals of ABB
                       np.ndarray[real_t, ndim=1, mode='c'] ail,
                       np.ndarray[real_t, ndim=1, mode='c'] aii,
                       np.ndarray[real_t, ndim=1, mode='c'] aiu,
                       # All 5 diagonals of BBB
                       np.ndarray[real_t, ndim=1, mode='c'] bill,
                       np.ndarray[real_t, ndim=1, mode='c'] bil,
                       np.ndarray[real_t, ndim=1, mode='c'] bii,
                       np.ndarray[real_t, ndim=1, mode='c'] biu,
                       np.ndarray[real_t, ndim=1, mode='c'] biuu,
                       np.ndarray[real_t, ndim=3, mode='c'] u0,
                       np.ndarray[real_t, ndim=3, mode='c'] u1,
                       np.ndarray[real_t, ndim=3, mode='c'] u2,
                       np.ndarray[real_t, ndim=3, mode='c'] l0,
                       np.ndarray[real_t, ndim=3, mode='c'] l1):
    cdef:
        unsigned int ii, jj, N1, N2, odd, i, j, k, kk, M, ll
        long long int m, n, p, dd, w0
        double a, b, c, pp
        double pi = np.pi
        vector[double] c0, c1, c2
        #double* pc0, pc1, pc2
        #np.ndarray[real_t, ndim=1] c0 = np.zeros(sii.shape[0]//2)
        #np.ndarray[real_t, ndim=1] c1 = np.zeros(sii.shape[0]//2)
        #np.ndarray[real_t, ndim=1] c2 = np.zeros(sii.shape[0]//2)

    M = sii.shape[0]//2

    c0.resize(M)
    c1.resize(M)
    c2.resize(M)

    if axis == 0:
        N1 = beta.shape[1]

        for j in xrange(N1):
            a = alfa
            b = beta[0, j]
            c = ceta[0, j]
            for odd in xrange(2):
                c0[0] = a*sii[odd] + b*aii[odd] + c*bii[odd]
                c0[1] = a*siu[odd] + b*aiu[odd] + c*biu[odd]
                c0[2] = a*siuu[odd] + c*biuu[odd]
                m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
                c0[3] = m*a*pi/(6+odd+3.)
                m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
                c0[4] = m*a*pi/(8+odd+3.)

                c1[0] = b*ail[odd] + c*bil[odd]
                c1[1] = a*sii[2+odd] + b*aii[2+odd] + c*bii[2+odd]
                c1[2] = a*siu[2+odd] + b*aiu[2+odd] + c*biu[2+odd]
                c1[3] = a*siuu[2+odd] + c*biuu[2+odd]
                m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
                c1[4] = m*a*pi/(8+odd+3.)

                c2[0] = c*bill[odd]
                c2[1] = b*ail[2+odd] + c*bil[2+odd]
                c2[2] = a*sii[4+odd] + b*aii[4+odd] + c*bii[4+odd]
                c2[3] = a*siu[4+odd] + b*aiu[4+odd] + c*biu[4+odd]
                c2[4] = a*siuu[4+odd] + c*biuu[4+odd]

                for i in xrange(5, M):
                    p = 2*i+odd
                    pp = pi/(p+3.)
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(p+2)*(p+2))
                    c0[i] = m*a*pp
                    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(p+2)*(p+2))
                    c1[i] = m*a*pp
                    m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(p+2)*(p+2))
                    c2[i] = m*a*pp

                u0[odd, 0, j] = c0[0]
                u1[odd, 0, j] = c0[1]
                u2[odd, 0, j] = c0[2]
                for kk in xrange(1, M):
                    l0[odd, kk-1, j] = c1[kk-1]/u0[odd, kk-1, j]
                    if kk < M-1:
                        l1[odd, kk-1, j] = c2[kk-1]/u0[odd, kk-1, j]

                    for i in xrange(kk, M):
                        c1[i] -= l0[odd, kk-1, j]*c0[i]

                    if kk < M-1:
                        for i in xrange(kk, M):
                            c2[i] -= l1[odd, kk-1, j]*c0[i]

                    #for i in xrange(kk, M):
                        #c0[i] = c1[i]
                        #c1[i] = c2[i]
                    copy(c1.begin()+kk, c1.end(), c0.begin()+kk)
                    copy(c2.begin()+kk, c2.end(), c1.begin()+kk)

                    if kk < M-2:
                        ll = 2*kk+odd
                        c2[kk] = c*bill[ll]
                        c2[kk+1] = b*ail[ll+2] + c*bil[ll+2]
                        c2[kk+2] = a*sii[ll+4] + b*aii[ll+4] + c*bii[ll+4]
                        if kk < M-3:
                            c2[kk+3] = a*siu[ll+4] + b*aiu[ll+4] + c*biu[ll+4]
                        if kk < M-4:
                            c2[kk+4] = a*siuu[ll+4] + c*biuu[ll+4]
                        if kk < M-5:
                            n = 2*(kk+2)+odd
                            dd = 8*(n+1)*(n+2)
                            w0 = dd*n*(n+4)
                            for i in xrange(kk+5, M):
                                p = 2*i+odd
                                c2[i] = (w0 + dd*3*(p+2)*(p+2))*a*pi/(p+3.)

                    u0[odd, kk, j] = c0[kk]
                    if kk < M-1:
                        u1[odd, kk, j] = c0[kk+1]
                    if kk < M-2:
                        u2[odd, kk, j] = c0[kk+2]

    elif axis == 1:
        N1 = beta.shape[0]

        for j in xrange(N1):
            a = alfa
            b = beta[j, 0]
            c = ceta[j, 0]
            for odd in xrange(2):
                c0[0] = a*sii[odd] + b*aii[odd] + c*bii[odd]
                c0[1] = a*siu[odd] + b*aiu[odd] + c*biu[odd]
                c0[2] = a*siuu[odd] + c*biuu[odd]
                m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
                c0[3] = m*a*pi/(6+odd+3.)
                m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
                c0[4] = m*a*pi/(8+odd+3.)

                c1[0] = b*ail[odd] + c*bil[odd]
                c1[1] = a*sii[2+odd] + b*aii[2+odd] + c*bii[2+odd]
                c1[2] = a*siu[2+odd] + b*aiu[2+odd] + c*biu[2+odd]
                c1[3] = a*siuu[2+odd] + c*biuu[2+odd]
                m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
                c1[4] = m*a*pi/(8+odd+3.)

                c2[0] = c*bill[odd]
                c2[1] = b*ail[2+odd] + c*bil[2+odd]
                c2[2] = a*sii[4+odd] + b*aii[4+odd] + c*bii[4+odd]
                c2[3] = a*siu[4+odd] + b*aiu[4+odd] + c*biu[4+odd]
                c2[4] = a*siuu[4+odd] + c*biuu[4+odd]

                for i in xrange(5, M):
                    p = 2*i+odd
                    pp = pi/(p+3.)
                    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(p+2)*(p+2))
                    c0[i] = m*a*pp
                    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(p+2)*(p+2))
                    c1[i] = m*a*pp
                    m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(p+2)*(p+2))
                    c2[i] = m*a*pp

                u0[odd, j, 0] = c0[0]
                u1[odd, j, 0] = c0[1]
                u2[odd, j, 0] = c0[2]
                for kk in xrange(1, M):
                    l0[odd, j, kk-1] = c1[kk-1]/u0[odd, j, kk-1]
                    if kk < M-1:
                        l1[odd, j, kk-1] = c2[kk-1]/u0[odd, j, kk-1]

                    for i in xrange(kk, M):
                        c1[i] -= l0[odd, j, kk-1]*c0[i]

                    if kk < M-1:
                        for i in xrange(kk, M):
                            c2[i] -= l1[odd, j, kk-1]*c0[i]

                    #for i in xrange(kk, M):
                        #c0[i] = c1[i]
                        #c1[i] = c2[i]
                    copy(c1.begin()+kk, c1.end(), c0.begin()+kk)
                    copy(c2.begin()+kk, c2.end(), c1.begin()+kk)

                    if kk < M-2:
                        ll = 2*kk+odd
                        c2[kk] = c*bill[ll]
                        c2[kk+1] = b*ail[ll+2] + c*bil[ll+2]
                        c2[kk+2] = a*sii[ll+4] + b*aii[ll+4] + c*bii[ll+4]
                        if kk < M-3:
                            c2[kk+3] = a*siu[ll+4] + b*aiu[ll+4] + c*biu[ll+4]
                        if kk < M-4:
                            c2[kk+4] = a*siuu[ll+4] + c*biuu[ll+4]
                        if kk < M-5:
                            n = 2*(kk+2)+odd
                            dd = 8*(n+1)*(n+2)
                            w0 = dd*n*(n+4)
                            for i in xrange(kk+5, M):
                                p = 2*i+odd
                                c2[i] = (w0 + dd*3*(p+2)*(p+2))*a*pi/(p+3.)

                    u0[odd, j, kk] = c0[kk]
                    if kk < M-1:
                        u1[odd, j, kk] = c0[kk+1]
                    if kk < M-2:
                        u2[odd, j, kk] = c0[kk+2]

def Solve_Helmholtz_Biharmonic_1D(np.ndarray[T, ndim=1] fk,
                                  np.ndarray[T, ndim=1] u_hat,
                                  np.ndarray[real_t, ndim=1] l2,
                                  np.ndarray[real_t, ndim=1] l1,
                                  np.ndarray[real_t, ndim=1] d,
                                  np.ndarray[real_t, ndim=1] u1,
                                  np.ndarray[real_t, ndim=1] u2):
    cdef:
        int i, j, n, k
        vector[T] y
        np.ndarray[T, ndim=1] bc = fk.copy()

    n = fk.shape[0]-4
    bc[:] = fk

    bc[2] -= l1[0]*bc[0]
    bc[3] -= l1[1]*bc[1]
    for k in range(4, n):
        bc[k] -= (l1[k-2]*bc[k-2] + l2[k-4]*bc[k-4])

    bc[n-1] /= d[n-1]
    bc[n-2] /= d[n-2]
    bc[n-3] /= d[n-3]
    bc[n-3] -= u1[n-3]*bc[n-1]/d[n-3]
    bc[n-4] /= d[n-4]
    bc[n-4] -= u1[n-4]*bc[n-2]/d[n-4]
    for k in range(n-5,-1,-1):
        bc[k] /= d[k]
        bc[k] -= (u1[k]*bc[k+2]/d[k] + u2[k]*bc[k+4]/d[k])
    u_hat[:] = bc

cdef void Solve_Helmholtz_Biharmonic_1D_ptr(T* fk,
                                            T* u_hat,
                                            real_t* l2,
                                            real_t* l1,
                                            real_t* d,
                                            real_t* u1,
                                            real_t* u2,
                                            int n,
                                            int strides) nogil:
    cdef:
        int st, k

    st = strides
    for k in range(n):
        u_hat[k*st] = fk[k*st]

    u_hat[2*st] -= l1[0]*u_hat[0]
    u_hat[3*st] -= l1[st]*u_hat[st]
    for k in range(4, n):
        u_hat[k*st] -= (l1[(k-2)*st]*u_hat[(k-2)*st] + l2[(k-4)*st]*u_hat[(k-4)*st])

    u_hat[(n-1)*st] /= d[(n-1)*st]
    u_hat[(n-2)*st] /= d[(n-2)*st]
    u_hat[(n-3)*st] /= d[(n-3)*st]
    u_hat[(n-3)*st] -= u1[(n-3)*st]*u_hat[(n-1)*st]/d[(n-3)*st]
    u_hat[(n-4)*st] /= d[(n-4)*st]
    u_hat[(n-4)*st] -= u1[(n-4)*st]*u_hat[(n-2)*st]/d[(n-4)*st]
    for k in range(n-5,-1,-1):
        u_hat[k*st] /= d[k*st]
        u_hat[k*st] -= (u1[k*st]*u_hat[(k+2)*st]/d[k*st] + u2[k*st]*u_hat[(k+4)*st]/d[k*st])

def Solve_Helmholtz_Biharmonic_1D_p(T[::1] fk,
                                    T[::1] u_hat,
                                    real_t[::1] l2,
                                    real_t[::1] l1,
                                    real_t[::1] d,
                                    real_t[::1] u1,
                                    real_t[::1] u2):
    cdef:
        T* fk_ptr
        T* u_hat_ptr
        real_t* d_ptr
        real_t* u1_ptr
        real_t* u2_ptr
        real_t* l1_ptr
        real_t* l2_ptr
        int ii, jj, strides

    strides = fk.strides[0]/fk.itemsize
    fk_ptr = &fk[0]
    u_hat_ptr = &u_hat[0]
    d_ptr = &d[0]
    u1_ptr = &u1[0]
    u2_ptr = &u2[0]
    l1_ptr = &l1[0]
    l2_ptr = &l2[0]
    Solve_Helmholtz_Biharmonic_1D_ptr(fk_ptr, u_hat_ptr, l2_ptr, l1_ptr,
                                      d_ptr, u1_ptr, u2_ptr, fk.shape[0]-4,
                                      strides)

def Solve_Helmholtz_Biharmonic_3D_ptr(np.int64_t axis,
                                      T[:,:,::1] fk,
                                      T[:,:,::1] u_hat,
                                      real_t[:,:,::1] l2,
                                      real_t[:,:,::1] l1,
                                      real_t[:,:,::1] d,
                                      real_t[:,:,::1] u1,
                                      real_t[:,:,::1] u2):
    cdef:
        T* fk_ptr
        T* u_hat_ptr
        real_t* d_ptr
        real_t* u1_ptr
        real_t* u2_ptr
        real_t* l1_ptr
        real_t* l2_ptr
        int ii, jj, strides

    strides = fk.strides[axis]/fk.itemsize
    if axis == 0:
        for ii in range(d.shape[1]):
            for jj in range(d.shape[2]):
                fk_ptr = &fk[0,ii,jj]
                u_hat_ptr = &u_hat[0,ii,jj]
                d_ptr = &d[0,ii,jj]
                u1_ptr = &u1[0,ii,jj]
                u2_ptr = &u2[0,ii,jj]
                l1_ptr = &l1[0,ii,jj]
                l2_ptr = &l2[0,ii,jj]
                Solve_Helmholtz_Biharmonic_1D_ptr(fk_ptr, u_hat_ptr, l2_ptr, l1_ptr,
                                                  d_ptr, u1_ptr, u2_ptr, d.shape[axis]-4,
                                                  strides)

    elif axis == 1:
        for ii in range(d.shape[0]):
            for jj in range(d.shape[2]):
                fk_ptr = &fk[ii,0,jj]
                u_hat_ptr = &u_hat[ii,0,jj]
                d_ptr = &d[ii,0,jj]
                u1_ptr = &u1[ii,0,jj]
                u2_ptr = &u2[ii,0,jj]
                l1_ptr = &l1[ii,0,jj]
                l2_ptr = &l2[ii,0,jj]
                Solve_Helmholtz_Biharmonic_1D_ptr(fk_ptr, u_hat_ptr, l2_ptr, l1_ptr,
                                                  d_ptr, u1_ptr, u2_ptr, d.shape[axis]-4,
                                                  strides)

    elif axis == 2:
        for ii in range(d.shape[0]):
            for jj in range(d.shape[1]):
                fk_ptr = &fk[ii,jj,0]
                u_hat_ptr = &u_hat[ii,jj,0]
                d_ptr = &d[ii,jj,0]
                u1_ptr = &u1[ii,jj,0]
                u2_ptr = &u2[ii,jj,0]
                l1_ptr = &l1[ii,jj,0]
                l2_ptr = &l2[ii,jj,0]
                Solve_Helmholtz_Biharmonic_1D_ptr(fk_ptr, u_hat_ptr, l2_ptr, l1_ptr,
                                                  d_ptr, u1_ptr, u2_ptr, d.shape[axis]-4,
                                                  strides)

def Solve_Helmholtz_Biharmonic_2D_ptr(np.int64_t axis,
                                      T[:,::1] fk,
                                      T[:,::1] u_hat,
                                      real_t[:,::1] l2,
                                      real_t[:,::1] l1,
                                      real_t[:,::1] d,
                                      real_t[:,::1] u1,
                                      real_t[:,::1] u2):
    cdef:
        T* fk_ptr
        T* u_hat_ptr
        real_t* d_ptr
        real_t* u1_ptr
        real_t* u2_ptr
        real_t* l1_ptr
        real_t* l2_ptr
        int ii, jj, strides

    strides = fk.strides[axis]/fk.itemsize
    if axis == 0:
        for ii in range(d.shape[1]):
            fk_ptr = &fk[0,ii]
            u_hat_ptr = &u_hat[0,ii]
            d_ptr = &d[0,ii]
            u1_ptr = &u1[0,ii]
            u2_ptr = &u2[0,ii]
            l1_ptr = &l1[0,ii]
            l2_ptr = &l2[0,ii]
            Solve_Helmholtz_Biharmonic_1D_ptr(fk_ptr, u_hat_ptr, l2_ptr, l1_ptr,
                                              d_ptr, u1_ptr, u2_ptr, d.shape[axis]-4,
                                              strides)

    elif axis == 1:
        for ii in range(d.shape[0]):
            fk_ptr = &fk[ii,0]
            u_hat_ptr = &u_hat[ii,0]
            d_ptr = &d[ii,0]
            u1_ptr = &u1[ii,0]
            u2_ptr = &u2[ii,0]
            l1_ptr = &l1[ii,0]
            l2_ptr = &l2[ii,0]
            Solve_Helmholtz_Biharmonic_1D_ptr(fk_ptr, u_hat_ptr, l2_ptr, l1_ptr,
                                              d_ptr, u1_ptr, u2_ptr, d.shape[axis]-4,
                                              strides)

def LU_Helmholtz_Biharmonic_3D(A, B, np.int64_t axis,
                               np.ndarray[real_t, ndim=3] A_scale,
                               np.ndarray[real_t, ndim=3] B_scale,
                               np.ndarray[real_t, ndim=3] l2,
                               np.ndarray[real_t, ndim=3] l1,
                               np.ndarray[real_t, ndim=3] d,
                               np.ndarray[real_t, ndim=3] u1,
                               np.ndarray[real_t, ndim=3] u2):
    cdef:
        unsigned int i, j, k

    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                LU_Helmholtz_Biharmonic_1D(A, B,
                                A_scale[0, j, k],
                                B_scale[0, j, k],
                                l2[:, j, k],
                                l1[:, j, k],
                                d[:, j, k],
                                u1[:, j, k],
                                u2[:, j, k])

    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                LU_Helmholtz_Biharmonic_1D(A, B,
                                A_scale[i, 0, k],
                                B_scale[i, 0, k],
                                l2[i,:,k],
                                l1[i,:,k],
                                d[i,:,k],
                                u1[i,:,k],
                                u2[i,:,k])

    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                LU_Helmholtz_Biharmonic_1D(A, B,
                                A_scale[i, j, 0],
                                B_scale[i, j, 0],
                                l2[i, j, :],
                                l1[i, j, :],
                                d[i, j, :],
                                u1[i, j, :],
                                u2[i, j, :])

def LU_Helmholtz_Biharmonic_1D(A, B,
                               np.float_t A_scale,
                               np.float_t B_scale,
                               np.ndarray[real_t, ndim=1] l2,
                               np.ndarray[real_t, ndim=1] l1,
                               np.ndarray[real_t, ndim=1] d,
                               np.ndarray[real_t, ndim=1] u1,
                               np.ndarray[real_t, ndim=1] u2
                               ):
    cdef:
        int i, n, k
        double lam
        np.ndarray[real_t, ndim=1] A_0 = A[0].copy()
        np.ndarray[real_t, ndim=1] A_2 = A[2].copy()
        np.ndarray[real_t, ndim=1] A_m2 = A[-2].copy()
        np.ndarray[real_t, ndim=1] B_m4 = B[-4].copy()
        np.ndarray[real_t, ndim=1] B_m2 = B[-2].copy()
        np.ndarray[real_t, ndim=1] B_0 = B[0].copy()
        np.ndarray[real_t, ndim=1] B_2 = B[2].copy()
        np.ndarray[real_t, ndim=1] B_4 = B[4].copy()

    n = A_0.shape[0]
    k = 2

    # Set up matrix diagonals
    l2[:] = B_scale*B_m4
    l1[:] = A_scale*A_m2 + B_scale*B_m2
    d[:] =  A_scale*A_0 + B_scale*B_0
    u1[:] = A_scale*A_2 + B_scale*B_2
    u2[:] = B_scale*B_4

    for i in range(n-2*k):
        lam = l1[i]/d[i]
        d[i+k] -= lam*u1[i]
        u1[i+k] -= lam*u2[i]
        l1[i] = lam
        lam = l2[i]/d[i]
        l1[i+k] -= lam*u1[i]
        d[i+2*k] -= lam*u2[i]
        l2[i] = lam

    i = n-4
    lam = l1[i]/d[i]
    d[i+k] -= lam*u1[i]
    l1[i] = lam
    i = n-3
    lam = l1[i]/d[i]
    d[i+k] -= lam*u1[i]
    l1[i] = lam
