#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

import numpy as np
cimport cython
cimport numpy as np
import cython

ctypedef fused T:
    np.float64_t
    np.complex128_t

ctypedef np.float64_t real_t

ctypedef void (*func_ptr)(T*, T*, int, real_t*, real_t*, real_t*, real_t*, real_t*, int)

def scalar_product(input_array, output_array, x, w, axis):
    cdef:
        int st, n
        np.ndarray[real_t, ndim=1] Lnm = np.ones_like(x)
        np.ndarray[real_t, ndim=1] Ln = np.zeros_like(x)
        np.ndarray[real_t, ndim=1] Lnp = np.zeros_like(x)

    n = input_array.ndim
    st = input_array.strides[axis]/input_array.itemsize
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _scalar_product[np.float64_t](input_array, output_array, x, w, Lnm, Ln, Lnp)
        elif n == 2:
            fun_2D[np.float64_t](_scalar_product_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 3:
            fun_3D[np.float64_t](_scalar_product_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 4:
            fun_4D[np.float64_t](_scalar_product_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)

    else:
        if n == 1:
            _scalar_product[np.complex128_t](input_array, output_array, x, w, Lnm, Ln, Lnp)
        elif n == 2:
            fun_2D[np.complex128_t](_scalar_product_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 3:
            fun_3D[np.complex128_t](_scalar_product_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 4:
            fun_4D[np.complex128_t](_scalar_product_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)

def evaluate_expansion_all(input_array, output_array, x, axis):
    cdef:
        int st, n
        np.ndarray[real_t, ndim=1] Lnm = np.ones_like(x)
        np.ndarray[real_t, ndim=1] Ln = np.zeros_like(x)
        np.ndarray[real_t, ndim=1] Lnp = np.zeros_like(x)
        np.ndarray[real_t, ndim=1] w = np.zeros_like(x)  # dummy
    n = input_array.ndim
    st = input_array.strides[axis]/input_array.itemsize
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _evaluate_expansion_all[np.float64_t](input_array, output_array, x, w, Lnm, Ln, Lnp)
        elif n == 2:
            fun_2D[np.float64_t](_evaluate_expansion_all_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 3:
            fun_3D[np.float64_t](_evaluate_expansion_all_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 4:
            fun_4D[np.float64_t](_evaluate_expansion_all_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)

    else:
        if n == 1:
            _evaluate_expansion_all[np.complex128_t](input_array, output_array, x, w, Lnm, Ln, Lnp)
        elif n == 2:
            fun_2D[np.complex128_t](_evaluate_expansion_all_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 3:
            fun_3D[np.complex128_t](_evaluate_expansion_all_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)
        elif n == 4:
            fun_4D[np.complex128_t](_evaluate_expansion_all_ptr, input_array, output_array, st, x, w, Lnm, Ln, Lnp, axis)

cdef void fun_2D(func_ptr fun, T[:, ::1] ui, T[:, ::1] uo, int st, real_t[::1] x, real_t[::1] w, real_t[::1] Lnm, real_t[::1] Ln, real_t[::1] Lnp, int axis):
    cdef:
        int i
    if axis == 0:
        for i in range(ui.shape[1]):
            fun(&ui[0, i], &uo[0, i], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])
    elif axis == 1:
        for i in range(ui.shape[0]):
            fun(&ui[i, 0], &uo[i, 0], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])

cdef void fun_3D(func_ptr fun, T[:, :, ::1] ui, T[:, :, ::1] uo, int st, real_t[::1] x, real_t[::1] w, real_t[::1] Lnm, real_t[::1] Ln, real_t[::1] Lnp, int axis):
    cdef:
        int i, j, k
    if axis == 0:
        for j in range(ui.shape[1]):
            for k in range(ui.shape[2]):
                fun(&ui[0, j, k], &uo[0, j, k], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])
    elif axis == 1:
        for i in range(ui.shape[0]):
            for k in range(ui.shape[2]):
                fun(&ui[i, 0, k], &uo[i, 0, k], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])
    elif axis == 2:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                fun(&ui[i, j, 0], &uo[i, j, 0], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])

cdef void fun_4D(func_ptr fun, T[:, :, :, ::1] ui, T[:, :, :, ::1] uo, int st, real_t[::1] x, real_t[::1] w, real_t[::1] Lnm, real_t[::1] Ln, real_t[::1] Lnp, int axis):
    cdef:
        int i, j, k, l
    if axis == 0:
        for j in range(ui.shape[1]):
            for k in range(ui.shape[2]):
                for l in range(ui.shape[3]):
                    fun(&ui[0, j, k, l], &uo[0, j, k, l], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])
    elif axis == 1:
        for i in range(ui.shape[0]):
            for k in range(ui.shape[2]):
                for l in range(ui.shape[3]):
                    fun(&ui[i, 0, k, l], &uo[i, 0, k, l], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])
    elif axis == 2:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                for l in range(ui.shape[3]):
                    fun(&ui[i, j, 0, l], &uo[i, j, 0, l], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])
    elif axis == 3:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                for k in range(ui.shape[2]):
                    fun(&ui[i, j, k, 0], &uo[i, j, k, 0], st, &x[0], &w[0], &Lnm[0], &Ln[0], &Lnp[0], x.shape[0])

cpdef _scalar_product(T[:] ui, T[:] uo, real_t[::1] xj, real_t[::1] wj, real_t[::1] Lnm, real_t[::1] Ln, real_t[::1] Lnp):
    _scalar_product_ptr[T](&ui[0], &uo[0], ui.strides[0]/ui.itemsize, &xj[0], &wj[0], &Lnm[0], &Ln[0], &Lnp[0], xj.shape[0])

cpdef _evaluate_expansion_all(T[:] ui, T[:] uo, real_t[::1] xj, real_t[::1] wj, real_t[::1] Lnm, real_t[::1] Ln, real_t[::1] Lnp):
    _evaluate_expansion_all_ptr[T](&ui[0], &uo[0], ui.strides[0]/ui.itemsize, &xj[0], &wj[0], &Lnm[0], &Ln[0], &Lnp[0], xj.shape[0])

cdef void _evaluate_expansion_all_ptr(T* ui,
                                      T* uo,
                                      int st,
                                      real_t* xj,
                                      real_t* wj,
                                      real_t* Lnm,
                                      real_t* Ln,
                                      real_t* Lnp,
                                      int N):
    cdef:
        int i, j
        T s
        real_t s1, s2
    for i in range(N):
        Lnm[i] = 1.
        Ln[i] = xj[i]
        uo[i*st] = 0.
        Lnp[i] = (3.*xj[i]*Ln[i] - Lnm[i])/2.

    for i in range(N):
        s2 = (i+2.)/(i+3.)
        s1 = (2.*(i+2.)+1.)/(i+3.)
        for j in range(N):
            uo[j*st] += Lnm[j]*ui[i*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]

cdef void _scalar_product_ptr(T* ui,
                              T* uo,
                              int st,
                              real_t* xj,
                              real_t* wj,
                              real_t* Lnm,
                              real_t* Ln,
                              real_t* Lnp,
                              int N):
    cdef:
        int i, j
        T s
        real_t s1, s2
    for i in range(N):
        Lnm[i] = 1.
        Ln[i] = xj[i]
        uo[i*st] = 0.
        Lnp[i] = (3*xj[i]*Ln[i] - Lnm[i])/2

    for i in range(N):
        s = 0.0
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        for j in range(N):
            s += Lnm[j]*wj[j]*ui[j*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]
        uo[i*st] = s
