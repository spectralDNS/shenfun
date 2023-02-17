#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

import numpy as np
cimport numpy as np
import cython
cimport cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport copy
from libc.stdlib cimport malloc, free
from cpython cimport array
import array
np.import_array()

ctypedef fused T:
    double
    complex

#ctypedef complex complex_t
#ctypedef double double
#ctypedef np.int64_t int

ctypedef void (*innerfunc)(complex*, int, double*, int, int)
ctypedef void (*funcT)(T*, int, double*, int, int)

# XXX_Solve - Solve multidimensional array u along axis

def ThreeDMA_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            ThreeDMA_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, ThreeDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, ThreeDMA_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            ThreeDMA_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, ThreeDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, ThreeDMA_inner_solve_ptr, axis)

def TwoDMA_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            TwoDMA_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, TwoDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, TwoDMA_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            TwoDMA_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, TwoDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, TwoDMA_inner_solve_ptr, axis)

def PDMA_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            PDMA_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, PDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, PDMA_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            PDMA_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, PDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, PDMA_inner_solve_ptr, axis)

def TDMA_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            TDMA_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, TDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, TDMA_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            TDMA_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, TDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, TDMA_inner_solve_ptr, axis)

def TDMA_O_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            TDMA_O_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, TDMA_O_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, TDMA_O_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            TDMA_O_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, TDMA_O_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, TDMA_O_inner_solve_ptr, axis)

cpdef DiagMA_Solve(u, double[:, ::1] data, int axis):
    cdef:
        int n = u.ndim

    if u.dtype.char in 'FDG':
        if n == 1:
            DiagMA_inner_solve[complex](u, data)
        elif n == 2:
            DiagMA_Solve_2D[complex](u, data, axis)
        elif n == 3:
            DiagMA_Solve_3D[complex](u, data, axis)
    else:
        if n == 1:
            DiagMA_inner_solve[double](u, data)
        elif n == 2:
            DiagMA_Solve_2D[double](u, data, axis)
        elif n == 3:
            DiagMA_Solve_3D[double](u, data, axis)

cpdef DiagMA_Solve_2D(np.ndarray[T, ndim=2] u, double[:, ::1] data, int axis):
    cdef:
        int st = u.strides[axis]/u.itemsize
        np.flatiter ita = np.PyArray_IterAllButAxis(u, &axis)
        int N = u.shape[axis]
        double* dp = &data[0, 0]
    while np.PyArray_ITER_NOTDONE(ita):
        DiagMA_inner_solve_ptr[T](<T*>np.PyArray_ITER_DATA(ita), st, dp, 0, N)
        np.PyArray_ITER_NEXT(ita)

cpdef DiagMA_Solve_3D(np.ndarray[T, ndim=3] u, double[:, ::1] data, int axis):
    cdef:
        int st = u.strides[axis]/u.itemsize
        np.flatiter ita = np.PyArray_IterAllButAxis(u, &axis)
        int N = u.shape[axis]
        double* dp = &data[0, 0]
    while np.PyArray_ITER_NOTDONE(ita):
        DiagMA_inner_solve_ptr[T](<T*>np.PyArray_ITER_DATA(ita), st, dp, 0, N)
        np.PyArray_ITER_NEXT(ita)

#def DiagMA_Solve(u, data, int axis):
#    if u.dtype.char in 'FDG':
#        if u.ndim == 1:
#            DiagMA_inner_solve[complex](u, data)
#        elif u.ndim == 2:
#            Solve_axis_2D[complex](u, data, DiagMA_inner_solve_ptr, axis)
#        elif u.ndim == 3:
#            Solve_axis_3D[complex](u, data, DiagMA_inner_solve_ptr, axis)
#    else:
#        if u.ndim == 1:
#            DiagMA_inner_solve[double](u, data)
#        elif u.ndim == 2:
#            Solve_axis_2D[double](u, data, DiagMA_inner_solve_ptr, axis)
#        elif u.ndim == 3:
#            Solve_axis_3D[double](u, data, DiagMA_inner_solve_ptr, axis)


def FDMA_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            FDMA_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, FDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, FDMA_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            FDMA_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, FDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, FDMA_inner_solve_ptr, axis)

def HeptaDMA_Solve(u, data, axis):
    if u.dtype.char in 'FDG':
        if u.ndim == 1:
            HeptaDMA_inner_solve[complex](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[complex](u, data, HeptaDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[complex](u, data, HeptaDMA_inner_solve_ptr, axis)
    else:
        if u.ndim == 1:
            HeptaDMA_inner_solve[double](u, data)
        elif u.ndim == 2:
            Solve_axis_2D[double](u, data, HeptaDMA_inner_solve_ptr, axis)
        elif u.ndim == 3:
            Solve_axis_3D[double](u, data, HeptaDMA_inner_solve_ptr, axis)

# LU - decomposition

def FDMA_LU(double[:, ::1] data):
    cdef:
        int n, i
        double[::1] ld = data[0, :-2]
        double[::1] d = data[1, :]
        double[::1] u1 = data[2, 2:]
        double[::1] u2 = data[3, 4:]
    n = d.shape[0]
    for i in range(2, n):
        ld[i-2] = ld[i-2]/d[i-2]
        d[i] = d[i] - ld[i-2]*u1[i-2]
        if i < n-2:
            u1[i] = u1[i] - ld[i-2]*u2[i-2]

def PDMA_LU(double[:, ::1] data):
    cdef:
        int i, n, m, k
        double[::1] a = data[0, :-4]
        double[::1] b = data[1, :-2]
        double[::1] d = data[2, :]
        double[::1] e = data[3, 2:]
        double[::1] f = data[4, 4:]
        double lam
    n = d.shape[0]
    m = e.shape[0]
    k = n - m
    for i in range(n-2*k):
        lam = b[i]/d[i]
        d[i+k] -= lam*e[i]
        e[i+k] -= lam*f[i]
        b[i] = lam
        lam = a[i]/d[i]
        b[i+k] -= lam*e[i]
        d[i+2*k] -= lam*f[i]
        a[i] = lam
    i = n-4
    lam = b[i]/d[i]
    d[i+k] -= lam*e[i]
    b[i] = lam
    i = n-3
    lam = b[i]/d[i]
    d[i+k] -= lam*e[i]
    b[i] = lam

def TDMA_LU(double[:, ::1] data):
    cdef:
        int i
        int n = data.shape[1]
        double[::1] ld = data[0, :-2]
        double[::1] d = data[1, :]
        double[::1] ud = data[2, 2:]
    for i in range(2, n):
        ld[i-2] = ld[i-2]/d[i-2]
        d[i] = d[i] - ld[i-2]*ud[i-2]

def TDMA_O_LU(double[:, ::1] data):
    cdef:
        int i
        int n = data.shape[1]
        double[::1] ld = data[0, :-1]
        double[::1] d = data[1, :]
        double[::1] ud = data[2, 1:]
    for i in range(1, n):
        ld[i-1] = ld[i-1]/d[i-1]
        d[i] = d[i] - ld[i-1]*ud[i-1]

def HeptaDMA_LU(double[:, ::1] data):
    cdef:
        int i, n, m, k
        double[::1] a = data[0, :-4]
        double[::1] b = data[1, :-2]
        double[::1] d = data[2, :]
        double[::1] e = data[3, 2:]
        double[::1] f = data[4, 4:]
        double[::1] g = data[5, 6:]
        double[::1] h = data[6, 8:]
        double lam
    n = d.shape[0]
    m = e.shape[0]
    k = n - m
    for i in range(n-2*k):
        lam = b[i]/d[i]
        d[i+k] -= lam*e[i]
        e[i+k] -= lam*f[i]
        if i < n-6:
            f[i+k] -= lam*g[i]
        if i < n-8:
            g[i+k] -= lam*h[i]
        b[i] = lam
        lam = a[i]/d[i]
        b[i+k] -= lam*e[i]
        d[i+2*k] -= lam*f[i]
        if i < n-6:
            e[i+2*k] -= lam*g[i]
        if i < n-8:
            f[i+2*k] -= lam*h[i]
        a[i] = lam
    i = n-4
    lam = b[i]/d[i]
    d[i+k] -= lam*e[i]
    b[i] = lam
    i = n-3
    lam = b[i]/d[i]
    d[i+k] -= lam*e[i]
    b[i] = lam

# Map Python functions to pure C functions

cdef innerfunc func_from_name(fun_name) except NULL:
    if fun_name == "PDMA_inner_solve":
        return PDMA_inner_solve_ptr
    elif fun_name == "TDMA_inner_solve":
        return TDMA_inner_solve_ptr
    elif fun_name == "TDMA_O_inner_solve":
        return TDMA_O_inner_solve_ptr
    elif fun_name == "FDMA_inner_solve":
        return FDMA_inner_solve_ptr
    elif fun_name == "ThreeDMA_inner_solve":
        return ThreeDMA_inner_solve_ptr
    elif fun_name == "TwoDMA_inner_solve":
        return TwoDMA_inner_solve_ptr
    elif fun_name == "DiagMA_inner_solve":
        return DiagMA_inner_solve_ptr
    elif fun_name == "HeptaDMA_inner_solve":
        return HeptaDMA_inner_solve_ptr
    else:
        return NULL

#def SolverGeneric1ND_solve_data2D(complex[:, ::1] u, double[:, :, ::1] data, sol, int naxes, bint is_zero_index):
#    cdef:
#        int i
#    if naxes == 0:
#        for i in range(u.shape[1]):
#            if i == 0 and is_zero_index:
#                continue
#            sol(u[:, i], data[i])
#
#    elif naxes == 1:
#        for i in range(u.shape[0]):
#            if i == 0 and is_zero_index:
#                continue
#            sol(u[i], data[i])
#
#def SolverGeneric1ND_solve_data3D(complex[:, :, ::1] u, double[:, :, :, ::1] data, sol, int naxes, bint is_zero_index):
#    cdef:
#        int i, j
#    if naxes == 0:
#        for i in range(u.shape[1]):
#            for j in range(u.shape[2]):
#                if i == 0 and j == 0 and is_zero_index:
#                    continue
#                sol(u[:, i, j], data[i, j])
#
#    elif naxes == 1:
#        for i in range(u.shape[0]):
#            for j in range(u.shape[2]):
#                if i == 0 and j == 0 and is_zero_index:
#                    continue
#                sol(u[i, :, j], data[i, j])
#
#    elif naxes == 2:
#        for i in range(u.shape[0]):
#            for j in range(u.shape[1]):
#                if i == 0 and j == 0 and is_zero_index:
#                    continue
#                sol(u[i, j], data[i, j])

# I have not found a way to get SolverGeneric1ND_solve_data2D and
# SolverGeneric1ND_solve_data3D to run fast. Seems like they insist
# on calling Python sol. Workaround in SolverGeneric1ND_solve_MA2D/3D
# for now.

def SolverGeneric1ND_solve_data(u, data, sol, naxes, is_zero_index):
    cdef:
        innerfunc f = func_from_name(sol.__name__)

    if u.ndim == 2:
        SolverGeneric1ND_solve_2D(u, data, f, naxes, is_zero_index)
    elif u.ndim == 3:
        SolverGeneric1ND_solve_3D(u, data, f, naxes, is_zero_index)
    #if u.ndim == 2:
    #    SolverGeneric1ND_solve_data2D(u, data, sol, naxes, is_zero_index)
    #elif u.ndim == 3:
    #    SolverGeneric1ND_solve_data3D(u, data, sol, naxes, is_zero_index)
    return u

cdef void SolverGeneric1ND_solve_3D(complex[:, :, ::1] u, double[:, :, :, ::1] data, innerfunc sol, int naxes, bint is_zero_index):
    cdef:
        int i, j, st

    st = u.strides[naxes]/u.itemsize
    if naxes == 0:
        for i in range(u.shape[1]):
            for j in range(u.shape[2]):
                if i == 0 and j == 0 and is_zero_index:
                    continue
                sol(&u[0, i, j], st, &data[i, j, 0, 0], data.shape[2], data.shape[3])

    elif naxes == 1:
        for i in range(u.shape[0]):
            for j in range(u.shape[2]):
                if i == 0 and j == 0 and is_zero_index:
                    continue
                sol(&u[i, 0, j], st, &data[i, j, 0, 0], data.shape[2], data.shape[3])

    elif naxes == 2:
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                if i == 0 and j == 0 and is_zero_index:
                    continue
                sol(&u[i, j, 0], st, &data[i, j, 0, 0], data.shape[2], data.shape[3])

cdef void SolverGeneric1ND_solve_2D(complex[:, ::1] u, double[:, :, ::1] data, innerfunc sol, int naxes, bint is_zero_index):
    cdef:
        int i, j, st

    st = u.strides[naxes]/u.itemsize
    if naxes == 0:
        for i in range(u.shape[1]):
            if i == 0 and is_zero_index:
                continue
            sol(&u[0, i], st, &data[i, 0, 0], data.shape[1], data.shape[2])

    elif naxes == 1:
        for i in range(u.shape[0]):
            if i == 0 and is_zero_index:
                continue
            sol(&u[i, 0], st, &data[i, 0, 0], data.shape[1], data.shape[2])

@cython.cdivision(True)
cdef void Solve_axis_3D(T[:, :, ::1] u, double[:, ::1] data, funcT sol, int naxes):
    cdef:
        int i, j, st

    st = u.strides[naxes]/u.itemsize
    if naxes == 0:
        for i in range(u.shape[1]):
            for j in range(u.shape[2]):
                sol(&u[0, i, j], st, &data[0, 0], data.shape[0], data.shape[1])

    elif naxes == 1:
        for i in range(u.shape[0]):
            for j in range(u.shape[2]):
                sol(&u[i, 0, j], st, &data[0, 0], data.shape[0], data.shape[1])

    elif naxes == 2:
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                sol(&u[i, j, 0], st, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void Solve_axis_2D(T[:, ::1] u, double[:, ::1] data, funcT sol, int naxes):
    cdef:
        int i, j, st

    st = u.strides[naxes]/u.itemsize
    if naxes == 0:
        for i in range(u.shape[1]):
            sol(&u[0, i], st, &data[0, 0], data.shape[0], data.shape[1])
    elif naxes == 1:
        for i in range(u.shape[0]):
            sol(&u[i, 0], st, &data[0, 0], data.shape[0], data.shape[1])

cpdef HeptaDMA_inner_solve(T[:] u, double[:, ::1] data):
    HeptaDMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void HeptaDMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int n = m1
        int k
        double* a = &data[0]
        double* b = &data[m1]
        double* d = &data[2*m1]
        double* e = &data[3*m1+2]
        double* f = &data[4*m1+4]
        double* g = &data[5*m1+6]
        double* h = &data[6*m1+8]
    u[2*st] -= b[0]*u[0]
    u[3*st] -= b[1]*u[st]
    for k in range(4, n):
        u[k*st] -= (b[k-2]*u[(k-2)*st] + a[k-4]*u[(k-4)*st])
    u[(n-1)*st] /= d[n-1]
    u[(n-2)*st] /= d[n-2]
    u[(n-3)*st] = (u[(n-3)*st]-e[n-3]*u[(n-1)*st])/d[n-3]
    u[(n-4)*st] = (u[(n-4)*st]-e[n-4]*u[(n-2)*st])/d[n-4]
    u[(n-5)*st] = (u[(n-5)*st]-e[n-5]*u[(n-3)*st]-f[n-5]*u[(n-1)*st])/d[n-5]
    u[(n-6)*st] = (u[(n-6)*st]-e[n-6]*u[(n-4)*st]-f[n-6]*u[(n-2)*st])/d[n-6]
    u[(n-7)*st] = (u[(n-7)*st]-e[n-7]*u[(n-5)*st]-f[n-7]*u[(n-3)*st]-g[n-7]*u[(n-1)*st])/d[n-7]
    u[(n-8)*st] = (u[(n-8)*st]-e[n-8]*u[(n-6)*st]-f[n-8]*u[(n-4)*st]-g[n-8]*u[(n-2)*st])/d[n-8]
    for k in range(n-9, -1, -1):
        u[k*st] = (u[k*st]-e[k]*u[(k+2)*st]-f[k]*u[(k+4)*st]-g[k]*u[(k+6)*st]-h[k]*u[(k+8)*st])/d[k]

cpdef PDMA_inner_solve(T[:] u, double[:, ::1] data):
    PDMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void PDMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int n = m1
        int k
        double* a = &data[0]
        double* b = &data[m1]
        double* d = &data[2*m1]
        double* e = &data[3*m1+2]
        double* f = &data[4*m1+4]
    u[2*st] -= b[0]*u[0]
    u[3*st] -= b[1]*u[st]
    for k in range(4, n):
        u[k*st] -= (b[k-2]*u[(k-2)*st] + a[k-4]*u[(k-4)*st])
    u[(n-1)*st] /= d[n-1]
    u[(n-2)*st] /= d[n-2]
    u[(n-3)*st] = (u[(n-3)*st]-e[n-3]*u[(n-1)*st])/d[n-3]
    u[(n-4)*st] = (u[(n-4)*st]-e[n-4]*u[(n-2)*st])/d[n-4]
    for k in range(n-5, -1, -1):
        u[k*st] = (u[k*st]-e[k]*u[(k+2)*st]-f[k]*u[(k+4)*st])/d[k]

cpdef TDMA_inner_solve(T[:] u, double[:, ::1] data):
    TDMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void TDMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int n = m1
        int i
        double* ld = &data[0]
        double* d = &data[m1]
        double* ud = &data[m1*2+2]
    for i in range(2, n):
        u[i*st] -= ld[i-2]*u[(i-2)*st]
    u[(n-1)*st] = u[(n-1)*st]/d[n-1]
    u[(n-2)*st] = u[(n-2)*st]/d[n-2]
    for i in range(n - 3, -1, -1):
        u[i*st] = (u[i*st] - ud[i]*u[(i+2)*st])/d[i]

cpdef TDMA_O_inner_solve(T[:] u, double[:, ::1] data):
    TDMA_O_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void TDMA_O_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int n = m1
        int i
        double* ld = &data[0]
        double* d = &data[m1]
        double* ud = &data[2*m1+1]
    for i in range(1, n):
        u[i*st] -= ld[i-1]*u[(i-1)*st]
    u[(n-1)*st] = u[(n-1)*st]/d[n-1]
    for i in range(n-2, -1, -1):
        u[i*st] = (u[i*st] - ud[i]*u[(i+1)*st])/d[i]

cpdef TwoDMA_inner_solve(T[:] u, double[:, ::1] data):
    TwoDMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void TwoDMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int i, n = m1
        double* d = &data[0]
        double* u1 = &data[m1+2]
    u[(n-1)*st] = u[(n-1)*st]/d[n-1]
    u[(n-2)*st] = u[(n-2)*st]/d[n-2]
    for i in range(n - 3, -1, -1):
        u[i*st] = (u[i*st] - u1[i]*u[(i+2)*st])/d[i]

cpdef ThreeDMA_inner_solve(T[:] u, double[:, ::1] data):
    ThreeDMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void ThreeDMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int i, n = m1
        double* d = &data[0]
        double* u1 = &data[m1+2]
        double* u2 = &data[m1+4]
    u[(n-1)*st] = u[(n-1)*st]/d[n-1]
    u[(n-2)*st] = u[(n-2)*st]/d[n-2]
    u[(n-3)*st] = (u[(n-3)*st]-u1[n-3]*u[(n-1)*st])/d[n-3]
    u[(n-4)*st] = (u[(n-4)*st]-u1[n-4]*u[(n-2)*st])/d[n-4]
    for i in range(n - 5, -1, -1):
        u[i*st] = (u[i*st] - u1[i]*u[(i+2)*st] - u2[i]*u[(i+4)*st])/d[i]

cpdef DiagMA_inner_solve(T[:] u, double[:, ::1] data):
    DiagMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void DiagMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int i
    for i in range(m1):
        u[i*st] /= data[i]

cpdef FDMA_inner_solve(T[:] u, double[:, ::1] data):
    FDMA_inner_solve_ptr[T](&u[0], u.strides[0]/u.itemsize, &data[0, 0], data.shape[0], data.shape[1])

@cython.cdivision(True)
cdef void FDMA_inner_solve_ptr(T* u, int st, double* data, int m0, int m1):
    cdef:
        int i
        int n = m1
        double* ld = &data[0]
        double* d = &data[m1]
        double* u1 = &data[2*m1+2]
        double* u2 = &data[3*m1+4]
    for i in range(2, n):
        u[i*st] -= ld[i-2]*u[(i-2)*st]
    u[(n-1)*st] = u[(n-1)*st]/d[n-1]
    u[(n-2)*st] = u[(n-2)*st]/d[n-2]
    u[(n-3)*st] = (u[(n-3)*st] - u1[n-3]*u[(n-1)*st])/d[n-3]
    u[(n-4)*st] = (u[(n-4)*st] - u1[n-4]*u[(n-2)*st])/d[n-4]
    for i in range(n - 5, -1, -1):
        u[i*st] = (u[i*st] - u1[i]*u[(i+2)*st] - u2[i]*u[(i+4)*st])/d[i]


def Poisson_Solve_ADD(A, b, u, axis=0):
    cdef:
        double[::1] a0 = A[0]
        double[::1] a2 = A[2]
        double sc = A.scale
        int n
    n = u.ndim
    if n == 1:
        Poisson_Solve_ADD_1D(a0, a2, sc, b, u)
    elif n == 2:
        Poisson_Solve_ADD_2D_ptr(a0, a2, sc, b, u, axis)
    elif n == 3:
        Poisson_Solve_ADD_3D_ptr(a0, a2, sc, b, u, axis)

    return u

def Poisson_Solve_ADD_2D_ptr(double[::1] d,
                             double[::1] d1,
                             double scale,
                             T[:, ::1] b,
                             T[:, ::1] u,
                             int axis):
    cdef:
        int i, j, strides, N

    strides = u.strides[axis]/u.itemsize
    N = d.shape[0]
    if axis == 0:
        for j in range(u.shape[1]):
            Poisson_Solve_ADD_1D_ptr(&d[0], &d1[0], scale, &b[0, j], &u[0, j], N, strides)
    elif axis == 1:
        for i in range(u.shape[0]):
            Poisson_Solve_ADD_1D_ptr(&d[0], &d1[0], scale, &b[i, 0], &u[i, 0], N, strides)

def Poisson_Solve_ADD_3D_ptr(double[::1] d,
                             double[::1] d1,
                             double scale,
                             T[:, :, ::1] b,
                             T[:, :, ::1] u,
                             int axis):
    cdef:
        int i, j, k, strides, N

    strides = u.strides[axis]/u.itemsize
    N = d.shape[0]
    if axis == 0:
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                Poisson_Solve_ADD_1D_ptr(&d[0], &d1[0], scale, &b[0, j, k], &u[0, j, k], N, strides)
    elif axis == 1:
        for i in range(u.shape[0]):
            for k in range(u.shape[2]):
                Poisson_Solve_ADD_1D_ptr(&d[0], &d1[0], scale, &b[i, 0, k], &u[i, 0, k], N, strides)
    elif axis == 2:
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                Poisson_Solve_ADD_1D_ptr(&d[0], &d1[0], scale, &b[i, j, 0], &u[i, j, 0], N, strides)


cdef void Poisson_Solve_ADD_1D(double[::1] d,
                               double[::1] d1,
                               double scale,
                               double[::1] b,
                               double[::1] u):
    Poisson_Solve_ADD_1D_ptr(&d[0], &d1[0], scale, &b[0], &u[0], d.shape[0], 1)

cdef void Poisson_Solve_ADD_1D_ptr(double* d,
                                   double* d1,
                                   double scale,
                                   T* b,
                                   T* u,
                                   int N,
                                   int st):
    cdef:
        int k, ip, ii
        T se, so
    se = 0.0
    so = 0.0
    u[(N-1)*st] = b[(N-1)*st] / d[N-1]
    u[(N-2)*st] = b[(N-2)*st] / d[N-2]
    for k in range(N-3, -1, -1):
        ii = k*st
        ip = (k+2)*st
        if k%2 == 0:
            se += u[ip]
            u[ii] = b[ii] - d1[k]*se
        else:
            so += u[ip]
            u[ii] = b[ii] - d1[k]*so
        u[ii] /= d[k]
    if not abs(scale-1) < 1e-8:
        for k in range(N):
            u[k*st] /= scale

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
                    np.ndarray[double, ndim=1] d0,
                    np.ndarray[double, ndim=1] d1,
                    np.ndarray[double, ndim=1] d2,
                    np.ndarray[double, ndim=1] L):
    cdef:
        int i, N
        np.ndarray[double, ndim=1] A_0 = A[0].copy()
        np.ndarray[double, ndim=1] A_2 = A[2].copy()
        np.ndarray[double, ndim=1] A_4 = A[4].copy()
        np.ndarray[double, ndim=1] B_m2 = B.get(-2).copy()
        np.ndarray[double, ndim=1] B_0 = B[0].copy()
        np.ndarray[double, ndim=1] B_2 = B[2].copy()

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
                    np.ndarray[double, ndim=3] A_scale,
                    np.ndarray[double, ndim=3] B_scale,
                    bint neumann,
                    np.ndarray[double, ndim=3] d0,
                    np.ndarray[double, ndim=3] d1,
                    np.ndarray[double, ndim=3] d2,
                    np.ndarray[double, ndim=3] L):
    cdef:
        int i, j, k

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
                    np.ndarray[double, ndim=2] A_scale,
                    np.ndarray[double, ndim=2] B_scale,
                    bint neumann,
                    np.ndarray[double, ndim=2] d0,
                    np.ndarray[double, ndim=2] d1,
                    np.ndarray[double, ndim=2] d2,
                    np.ndarray[double, ndim=2] L):
    cdef:
        int i

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
        uu = np.ascontiguousarray(u)
        bb = np.ascontiguousarray(b)
        Solve_Helmholtz_1D(b, u, neumann, d0, d1, d2, L)
        if not u.flags['C_CONTIGUOUS']:
            u[:] = uu
    elif n == 2:
        Solve_Helmholtz_2D_ptr(axis, b, u, neumann, d0, d1, d2, L)
    elif n == 3:
        Solve_Helmholtz_3D_ptr(axis, b, u, neumann, d0, d1, d2, L)

def Solve_Helmholtz_1D(T[::1] fk,
                       T[::1] u_hat,
                       bint neumann,
                       double[::1] d0,
                       double[::1] d1,
                       double[::1] d2,
                       double[::1] L):
    cdef:
        vector[T] y
        int N = d0.shape[0]-2
    y.resize(N)
    Solve_Helmholtz_1D_ptr(&fk[0], &u_hat[0], neumann, &d0[0], &d1[0], &d2[0], &L[0], &y[0], N, 1)

cdef void Solve_Helmholtz_1D_ptr(T* fk,
                                 T* u_hat,
                                 bint neumann,
                                 double* d0,
                                 double* d1,
                                 double* d2,
                                 double* L,
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
                           double[:,:,::1] d0,
                           double[:,:,::1] d1,
                           double[:,:,::1] d2,
                           double[:,:,::1] L):
    cdef:
        vector[T] y
        int i, j, k, strides, N

    strides = fk.strides[axis]/fk.itemsize
    N = d0.shape[axis] - 2
    y.resize(N)
    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                Solve_Helmholtz_1D_ptr(&fk[0,j,k], &u_hat[0,j,k], neumann, &d0[0,j,k],
                                       &d1[0,j,k], &d2[0,j,k], &L[0,j,k], &y[0], N,
                                       strides)
    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                Solve_Helmholtz_1D_ptr(&fk[i,0,k], &u_hat[i,0,k], neumann, &d0[i,0,k],
                                       &d1[i,0,k], &d2[i,0,k], &L[i,0,k], &y[0], N,
                                       strides)

    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                Solve_Helmholtz_1D_ptr(&fk[i,j,0], &u_hat[i,j,0], neumann, &d0[i,j,0],
                                       &d1[i,j,0], &d2[i,j,0], &L[i,j,0], &y[0], N,
                                       strides)

def Solve_Helmholtz_2D_ptr(np.int64_t axis,
                           T[:,::1] fk,
                           T[:,::1] u_hat,
                           bint neumann,
                           double[:,::1] d0,
                           double[:,::1] d1,
                           double[:,::1] d2,
                           double[:,::1] L):
    cdef:
        vector[T] y
        int i, j, strides, N

    strides = fk.strides[axis]/fk.itemsize
    N = d0.shape[axis] - 2
    y.resize(N)
    if axis == 0:
        for j in range(d0.shape[1]):
            Solve_Helmholtz_1D_ptr(&fk[0,j], &u_hat[0,j], neumann, &d0[0,j],
                                   &d1[0,j], &d2[0,j], &L[0,j], &y[0], N,
                                   strides)
    elif axis == 1:
        for i in range(d0.shape[0]):
            Solve_Helmholtz_1D_ptr(&fk[i,0], &u_hat[i,0], neumann, &d0[i,0],
                                   &d1[i,0], &d2[i,0], &L[i,0], &y[0], N,
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
                     np.ndarray[double, ndim=1] sii,
                     np.ndarray[double, ndim=1] siu,
                     np.ndarray[double, ndim=1] siuu,
                     # All 3 diagonals of ABB
                     np.ndarray[double, ndim=1] ail,
                     np.ndarray[double, ndim=1] aii,
                     np.ndarray[double, ndim=1] aiu,
                     # All 5 diagonals of BBB
                     np.ndarray[double, ndim=1] bill,
                     np.ndarray[double, ndim=1] bil,
                     np.ndarray[double, ndim=1] bii,
                     np.ndarray[double, ndim=1] biu,
                     np.ndarray[double, ndim=1] biuu,
                     # Three upper and two lower diagonals of LU decomposition
                     np.ndarray[double, ndim=2] u0,
                     np.ndarray[double, ndim=2] u1,
                     np.ndarray[double, ndim=2] u2,
                     np.ndarray[double, ndim=2] l0,
                     np.ndarray[double, ndim=2] l1):

    LU_oe_Biharmonic_1D(0, a, b, c, sii[::2], siu[::2], siuu[::2], ail[::2], aii[::2], aiu[::2], bill[::2], bil[::2], bii[::2], biu[::2], biuu[::2], u0[0], u1[0], u2[0], l0[0], l1[0])
    LU_oe_Biharmonic_1D(1, a, b, c, sii[1::2], siu[1::2], siuu[1::2], ail[1::2], aii[1::2], aiu[1::2], bill[1::2], bil[1::2], bii[1::2], biu[1::2], biuu[1::2], u0[1], u1[1], u2[1], l0[1], l1[1])

def LU_oe_Biharmonic_1D(bint odd,
                        np.float_t a,
                        np.float_t b,
                        np.float_t c,
                        # 3 upper diagonals of SBB
                        np.ndarray[double, ndim=1] sii,
                        np.ndarray[double, ndim=1] siu,
                        np.ndarray[double, ndim=1] siuu,
                        # All 3 diagonals of ABB
                        np.ndarray[double, ndim=1] ail,
                        np.ndarray[double, ndim=1] aii,
                        np.ndarray[double, ndim=1] aiu,
                        # All 5 diagonals of BBB
                        np.ndarray[double, ndim=1] bill,
                        np.ndarray[double, ndim=1] bil,
                        np.ndarray[double, ndim=1] bii,
                        np.ndarray[double, ndim=1] biu,
                        np.ndarray[double, ndim=1] biuu,
                        # Two upper and two lower diagonals of LU decomposition
                        np.ndarray[double, ndim=1] u0,
                        np.ndarray[double, ndim=1] u1,
                        np.ndarray[double, ndim=1] u2,
                        np.ndarray[double, ndim=1] l0,
                        np.ndarray[double, ndim=1] l1):

    cdef:
        int i, j, kk
        long long int m, k
        double pi = np.pi
        vector[double] c0, c1, c2

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
                     np.ndarray[double, ndim=1] l0,
                     np.ndarray[double, ndim=1] l1,
                     np.ndarray[T, ndim=1] fk):
    # Solve Forward Ly = f
    cdef np.intp_t i, N
    y[0] = fk[0]
    y[1] = fk[1] - l0[0]*y[0]
    N = l0.shape[0]
    for i in xrange(2, N):
        y[i] = fk[i] - l0[i-1]*y[i-1] - l1[i-2]*y[i-2]

def Biharmonic_factor_pr_3D(np.int64_t axis,
                            np.ndarray[double, ndim=4] a,
                            np.ndarray[double, ndim=4] b,
                            np.ndarray[double, ndim=4] l0,
                            np.ndarray[double, ndim=4] l1):

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
                            np.ndarray[double, ndim=3] a,
                            np.ndarray[double, ndim=3] b,
                            np.ndarray[double, ndim=3] l0,
                            np.ndarray[double, ndim=3] l1):

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

def Biharmonic_factor_pr_1D(np.ndarray[double, ndim=2] a,
                            np.ndarray[double, ndim=2] b,
                            np.ndarray[double, ndim=2] l0,
                            np.ndarray[double, ndim=2] l1):

    Biharmonic_factor_oe_pr(0, a[0], b[0], l0[0], l1[0])
    Biharmonic_factor_oe_pr(1, a[1], b[1], l0[1], l1[1])

def Biharmonic_factor_oe_pr(bint odd,
                            np.ndarray[double, ndim=1] a,
                            np.ndarray[double, ndim=1] b,
                            np.ndarray[double, ndim=1] l0,
                            np.ndarray[double, ndim=1] l1):
    cdef:
        int i, j, M
        double pi = np.pi
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
                        np.ndarray[double, ndim=2] u0,
                        np.ndarray[double, ndim=2] u1,
                        np.ndarray[double, ndim=2] u2,
                        np.ndarray[double, ndim=2] l0,
                        np.ndarray[double, ndim=2] l1,
                        np.ndarray[double, ndim=2] a,
                        np.ndarray[double, ndim=2] b,
                        np.float_t ac):
    Solve_oe_Biharmonic_1D(0, fk[::2], uk[::2], u0[0], u1[0], u2[0], l0[0], l1[0], a[0], b[0], ac)
    Solve_oe_Biharmonic_1D(1, fk[1::2], uk[1::2], u0[1], u1[1], u2[1], l0[1], l1[1], a[1], b[1], ac)

cdef BackBsolve_U(int M,
                  bint odd,
                  np.ndarray[T, ndim=1] f,  # Uc = f
                  np.ndarray[T, ndim=1] uk,
                  np.ndarray[double, ndim=1] u0,
                  np.ndarray[double, ndim=1] u1,
                  np.ndarray[double, ndim=1] u2,
                  np.ndarray[double, ndim=1] l0,
                  np.ndarray[double, ndim=1] l1,
                  np.ndarray[double, ndim=1] a,
                  np.ndarray[double, ndim=1] b,
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
                           np.ndarray[double, ndim=1] u0,
                           np.ndarray[double, ndim=1] u1,
                           np.ndarray[double, ndim=1] u2,
                           np.ndarray[double, ndim=1] l0,
                           np.ndarray[double, ndim=1] l1,
                           np.ndarray[double, ndim=1] a,
                           np.ndarray[double, ndim=1] b,
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
                          np.ndarray[double, ndim=4, mode='c'] u0,
                          np.ndarray[double, ndim=4, mode='c'] u1,
                          np.ndarray[double, ndim=4, mode='c'] u2,
                          np.ndarray[double, ndim=4, mode='c'] l0,
                          np.ndarray[double, ndim=4, mode='c'] l1,
                          np.ndarray[double, ndim=4, mode='c'] a,
                          np.ndarray[double, ndim=4, mode='c'] b,
                          double a0):

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
                          np.ndarray[double, ndim=3, mode='c'] u0,
                          np.ndarray[double, ndim=3, mode='c'] u1,
                          np.ndarray[double, ndim=3, mode='c'] u2,
                          np.ndarray[double, ndim=3, mode='c'] l0,
                          np.ndarray[double, ndim=3, mode='c'] l1,
                          np.ndarray[double, ndim=3, mode='c'] a,
                          np.ndarray[double, ndim=3, mode='c'] b,
                          double a0):

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
                       double alfa,
                       np.ndarray[double, ndim=3] beta,
                       np.ndarray[double, ndim=3] ceta,
                       # 3 upper diagonals of SBB
                       np.ndarray[double, ndim=1, mode='c'] sii,
                       np.ndarray[double, ndim=1, mode='c'] siu,
                       np.ndarray[double, ndim=1, mode='c'] siuu,
                       # All 3 diagonals of ABB
                       np.ndarray[double, ndim=1, mode='c'] ail,
                       np.ndarray[double, ndim=1, mode='c'] aii,
                       np.ndarray[double, ndim=1, mode='c'] aiu,
                       # All 5 diagonals of BBB
                       np.ndarray[double, ndim=1, mode='c'] bill,
                       np.ndarray[double, ndim=1, mode='c'] bil,
                       np.ndarray[double, ndim=1, mode='c'] bii,
                       np.ndarray[double, ndim=1, mode='c'] biu,
                       np.ndarray[double, ndim=1, mode='c'] biuu,
                       np.ndarray[double, ndim=4, mode='c'] u0,
                       np.ndarray[double, ndim=4, mode='c'] u1,
                       np.ndarray[double, ndim=4, mode='c'] u2,
                       np.ndarray[double, ndim=4, mode='c'] l0,
                       np.ndarray[double, ndim=4, mode='c'] l1):
    cdef:
        unsigned int ii, jj, N1, N2, odd, i, j, k, kk, M, ll
        long long int m, n, p, dd, w0
        double a, b, c, pp
        double pi = np.pi
        vector[double] c0, c1, c2
        #double* pc0, pc1, pc2
        #np.ndarray[double, ndim=1] c0 = np.zeros(sii.shape[0]//2)
        #np.ndarray[double, ndim=1] c1 = np.zeros(sii.shape[0]//2)
        #np.ndarray[double, ndim=1] c2 = np.zeros(sii.shape[0]//2)

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
                       double alfa,
                       np.ndarray[double, ndim=2] beta,
                       np.ndarray[double, ndim=2] ceta,
                       # 3 upper diagonals of SBB
                       np.ndarray[double, ndim=1, mode='c'] sii,
                       np.ndarray[double, ndim=1, mode='c'] siu,
                       np.ndarray[double, ndim=1, mode='c'] siuu,
                       # All 3 diagonals of ABB
                       np.ndarray[double, ndim=1, mode='c'] ail,
                       np.ndarray[double, ndim=1, mode='c'] aii,
                       np.ndarray[double, ndim=1, mode='c'] aiu,
                       # All 5 diagonals of BBB
                       np.ndarray[double, ndim=1, mode='c'] bill,
                       np.ndarray[double, ndim=1, mode='c'] bil,
                       np.ndarray[double, ndim=1, mode='c'] bii,
                       np.ndarray[double, ndim=1, mode='c'] biu,
                       np.ndarray[double, ndim=1, mode='c'] biuu,
                       np.ndarray[double, ndim=3, mode='c'] u0,
                       np.ndarray[double, ndim=3, mode='c'] u1,
                       np.ndarray[double, ndim=3, mode='c'] u2,
                       np.ndarray[double, ndim=3, mode='c'] l0,
                       np.ndarray[double, ndim=3, mode='c'] l1):
    cdef:
        unsigned int ii, jj, N1, N2, odd, i, j, k, kk, M, ll
        long long int m, n, p, dd, w0
        double a, b, c, pp
        double pi = np.pi
        vector[double] c0, c1, c2
        #double* pc0, pc1, pc2
        #np.ndarray[double, ndim=1] c0 = np.zeros(sii.shape[0]//2)
        #np.ndarray[double, ndim=1] c1 = np.zeros(sii.shape[0]//2)
        #np.ndarray[double, ndim=1] c2 = np.zeros(sii.shape[0]//2)

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

