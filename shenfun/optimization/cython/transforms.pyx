#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

from cpython cimport array
import array
import numpy as np
cimport cython
cimport numpy as np
import cython

ctypedef fused T:
    double
    complex

ctypedef void (*funcX)(T*, T*, double*, int, int, double*, int)

cdef array.array darray = array.array('d', [])

def scalar_product(input_array, output_array, x, w, axis, a):
    cdef:
        int n
        np.ndarray[double, ndim=2] data = np.zeros((2, x.shape[0]))

    data[0] = x
    data[1] = w
    n = input_array.ndim
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _scalar_product[double](input_array, output_array, data, a)
        elif n == 2:
            fun_2D[double](_scalar_product_ptr, input_array, output_array, data, axis, a)
        elif n == 3:
            fun_3D[double](_scalar_product_ptr, input_array, output_array, data, axis, a)
        elif n == 4:
            fun_4D[double](_scalar_product_ptr, input_array, output_array, data, axis, a)

    else:
        if n == 1:
            _scalar_product[complex](input_array, output_array, data, a)
        elif n == 2:
            fun_2D[complex](_scalar_product_ptr, input_array, output_array, data, axis, a)
        elif n == 3:
            fun_3D[complex](_scalar_product_ptr, input_array, output_array, data, axis, a)
        elif n == 4:
            fun_4D[complex](_scalar_product_ptr, input_array, output_array, data, axis, a)

def evaluate_expansion_all(input_array, output_array, x, axis, a):
    cdef:
        int st, n
    n = input_array.ndim
    x = x.reshape((1, x.shape[0]))
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _evaluate_expansion_all[double](input_array, output_array, x, a)
        elif n == 2:
            fun_2D[double](_evaluate_expansion_all_ptr, input_array, output_array, x, axis, a)
        elif n == 3:
            fun_3D[double](_evaluate_expansion_all_ptr, input_array, output_array, x, axis, a)
        elif n == 4:
            fun_4D[double](_evaluate_expansion_all_ptr, input_array, output_array, x, axis, a)

    else:
        if n == 1:
            _evaluate_expansion_all[complex](input_array, output_array, x, a)
        elif n == 2:
            fun_2D[complex](_evaluate_expansion_all_ptr, input_array, output_array, x, axis, a)
        elif n == 3:
            fun_3D[complex](_evaluate_expansion_all_ptr, input_array, output_array, x, axis, a)
        elif n == 4:
            fun_4D[complex](_evaluate_expansion_all_ptr, input_array, output_array, x, axis, a)

cdef void fun_2D(funcX fun, T[:, ::1] ui, T[:, ::1] uo, double[:, ::1] data, int axis, double[:, ::1] a):
    cdef:
        int i, st
    st = ui.strides[axis]/ui.itemsize
    if axis == 0:
        for i in range(ui.shape[1]):
            fun(&ui[0, i], &uo[0, i], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])
    elif axis == 1:
        for i in range(ui.shape[0]):
            fun(&ui[i, 0], &uo[i, 0], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])

cdef void fun_3D(funcX fun, T[:, :, ::1] ui, T[:, :, ::1] uo, double[:, ::1] data, int axis, double[:, ::1] a):
    cdef:
        int i, j, k
        int st = ui.strides[axis]/ui.itemsize
    if axis == 0:
        for j in range(ui.shape[1]):
            for k in range(ui.shape[2]):
                fun(&ui[0, j, k], &uo[0, j, k], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])
    elif axis == 1:
        for i in range(ui.shape[0]):
            for k in range(ui.shape[2]):
                fun(&ui[i, 0, k], &uo[i, 0, k], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])
    elif axis == 2:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                fun(&ui[i, j, 0], &uo[i, j, 0], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])

cdef void fun_4D(funcX fun, T[:, :, :, ::1] ui, T[:, :, :, ::1] uo, double[:, ::1] data, int axis, double[:, ::1] a):
    cdef:
        int i, j, k, l
        int st = ui.strides[axis]/ui.itemsize
    if axis == 0:
        for j in range(ui.shape[1]):
            for k in range(ui.shape[2]):
                for l in range(ui.shape[3]):
                    fun(&ui[0, j, k, l], &uo[0, j, k, l], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])
    elif axis == 1:
        for i in range(ui.shape[0]):
            for k in range(ui.shape[2]):
                for l in range(ui.shape[3]):
                    fun(&ui[i, 0, k, l], &uo[i, 0, k, l], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])
    elif axis == 2:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                for l in range(ui.shape[3]):
                    fun(&ui[i, j, 0, l], &uo[i, j, 0, l], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])
    elif axis == 3:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                for k in range(ui.shape[2]):
                    fun(&ui[i, j, k, 0], &uo[i, j, k, 0], &data[0, 0], st, ui.shape[axis], &a[0, 0], a.shape[0])

cpdef _scalar_product(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _scalar_product_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], &a[0, 0], a.shape[0])

cpdef _evaluate_expansion_all(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _evaluate_expansion_all_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], &a[0, 0], a.shape[0])

@cython.cdivision(True)
cdef void _evaluate_expansion_all_ptr(T* ui,
                                      T* uo,
                                      double* data,
                                      int st,
                                      int N,
                                      double* a,
                                      int M):
    cdef:
        int i, j
        T s
        double s1, s2, a00
        double* xj = &data[0]
        double* anm = &a[0]
        double* anp
        double* ann
        array.array[double] an = array.clone(darray, N+3, zero=True)
        array.array[double] Lnm = array.clone(darray, N, zero=False)
        array.array[double] Ln = array.clone(darray, N, zero=False)
        array.array[double] Lnp = array.clone(darray, N, zero=False)

    if M == 2:
        ann = an.data.as_doubles
        anp = &a[N+3]
    else:
        ann = &a[N+3]
        anp = &a[2*(N+3)]
    for i in range(N):
        Lnm[i] = 1
        Ln[i] = (xj[i]-ann[0])/anm[0]
        uo[i*st] = 0
        Lnp[i] = (xj[i]-ann[1])/anm[1]*Ln[i] - anp[1]/anm[1]*Lnm[i]

    for i in range(N):
        s1 = 1/anm[i+2]
        s2 = anp[i+2]/anm[i+2]
        a00 = ann[i+2]
        for j in range(N):
            uo[j*st] += Lnm[j]*ui[i*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]

@cython.cdivision(True)
cdef void _scalar_product_ptr(T* ui,
                              T* uo,
                              double* data,
                              int st,
                              int N,
                              double* a,
                              int M):
    cdef:
        int i, j
        T s
        double s1, s2
        double* xj = &data[0]
        double* wj = &data[N]
        double* anm = &a[0]
        double* ann
        double* anp
        array.array[double] an = array.clone(darray, N+3, zero=True)
        array.array[double] Lnm = array.clone(darray, N, zero=False)
        array.array[double] Ln = array.clone(darray, N, zero=False)
        array.array[double] Lnp = array.clone(darray, N, zero=False)

    if M == 2:
        ann = an.data.as_doubles
        anp = &a[N+3]
    else:
        ann = &a[N+3]
        anp = &a[2*(N+3)]

    for i in range(N):
        Lnm[i] = 1
        Ln[i] = (xj[i]-ann[0])/anm[0]
        uo[i*st] = 0
        Lnp[i] = (xj[i]-ann[1])/anm[1]*Ln[i] - anp[1]/anm[1]*Lnm[i]

    for i in range(N):
        s1 = 1/anm[i+2]
        s2 = anp[i+2]/anm[i+2]
        a00 = ann[i+2]
        s = 0.0
        for j in range(N):
            s += Lnm[j]*wj[j]*ui[j*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]
        uo[i*st] = s
