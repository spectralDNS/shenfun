#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

from libcpp.vector cimport vector
from cpython cimport array
import array
import numpy as np
cimport cython
cimport numpy as np
import cython
from scipy.special import gammaln

ctypedef fused T:
    double
    complex

ctypedef void (*funcX)(T*, T*, double*, int, int, int, double*, int, int)

cdef array.array darray = array.array('d', [])

def restricted_product(L, input_array, output_array, x, i0, i1, a0, axis, a):
    cdef:
        int n
        int aM = output_array.shape[axis]
        np.ndarray[double, ndim=2] data = np.zeros((4, i1-i0))

    data[0] = x[i0:i1]
    data[1] = L.evaluate_basis(x[i0:i1], i=a0)
    data[2] = L.evaluate_basis(x[i0:i1], i=a0+1)
    data[3] = L.evaluate_basis(x[i0:i1], i=a0+2)
    ax = a[:, slice(a0, None)].copy()
    n = input_array.ndim
    sl = [slice(None)]*n
    sl[axis] = slice(i0, i1)
    input = input_array[tuple(sl)]
    if input.flags['C_CONTIGUOUS'] is False:
        input = input.copy()
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _restricted_product[double](input, output_array, data, ax)
        elif n == 2:
            fun_2D[double](_restricted_product_ptr, input, output_array, data, axis, ax)
        elif n == 3:
            fun_3D[double](_restricted_product_ptr, input, output_array, data, axis, ax)
        elif n == 4:
            fun_4D[double](_restricted_product_ptr, input, output_array, data, axis, ax)

    else:
        if n == 1:
            _restricted_product[complex](input, output_array, data, ax)
        elif n == 2:
            fun_2D[complex](_restricted_product_ptr, input, output_array, data, axis, ax)
        elif n == 3:
            fun_3D[complex](_restricted_product_ptr, input, output_array, data, axis, ax)
        elif n == 4:
            fun_4D[complex](_restricted_product_ptr, input, output_array, data, axis, ax)
    return output_array

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
    if not input_array.flags['C_CONTIGUOUS']:
        input_array = input_array.copy()

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

def Omega(z, a):
    a[:] = np.exp(gammaln(z+0.5) - gammaln(z+1))
    return a

def leg2cheb(input_array, output_array, axis=0, transpose=False):
    cdef:
        int n = input_array.ndim
        int N = input_array.shape[axis]
        np.ndarray[long int, ndim=1] k = np.arange(N)
        np.ndarray[double, ndim=2] a = np.zeros((1, N))
        np.ndarray[double, ndim=2] x = np.zeros((1, 1))

    a[0] = Omega(k, a[0])
    x[0] = transpose
    output_array.fill(0)
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _leg2cheb[double](input_array, output_array, x, a)
        elif n == 2:
            fun_2D[double](_leg2cheb_ptr, input_array, output_array, x, axis, a)
        elif n == 3:
            fun_3D[double](_leg2cheb_ptr, input_array, output_array, x, axis, a)
        elif n == 4:
            fun_4D[double](_leg2cheb_ptr, input_array, output_array, x, axis, a)

    else:
        if n == 1:
            _leg2cheb[complex](input_array, output_array, x, a)
        elif n == 2:
            fun_2D[complex](_leg2cheb_ptr, input_array, output_array, x, axis, a)
        elif n == 3:
            fun_3D[complex](_leg2cheb_ptr, input_array, output_array, x, axis, a)
        elif n == 4:
            fun_4D[complex](_leg2cheb_ptr, input_array, output_array, x, axis, a)
    return output_array

def cheb2leg(input_array, output_array, axis=0):
    cdef:
        int n = input_array.ndim
        int N = input_array.shape[axis]
        np.ndarray[long int, ndim=1] k = np.arange(N)
        np.ndarray[double, ndim=2] x = np.zeros((1, N//2))
        np.ndarray[double, ndim=2] a = np.zeros((1, N))

    k[0] = 1
    x[0] = Omega((k[::2]-2)/2, x[0])/k[::2]
    a[0] = 1/(2*Omega(k, a[0])*k*(k+0.5))
    a[0, 0] = 2/np.sqrt(np.pi)
    output_array.fill(0)
    if input_array.dtype.char in 'fdg':
        if n == 1:
            _cheb2leg[double](input_array.copy(), output_array, x, a)
        elif n == 2:
            fun_2D[double](_cheb2leg_ptr, input_array.copy(), output_array, x, axis, a)
        elif n == 3:
            fun_3D[double](_cheb2leg_ptr, input_array.copy(), output_array, x, axis, a)
        elif n == 4:
            fun_4D[double](_cheb2leg_ptr, input_array.copy(), output_array, x, axis, a)

    else:
        if n == 1:
            _cheb2leg[complex](input_array.copy(), output_array, x, a)
        elif n == 2:
            fun_2D[complex](_cheb2leg_ptr, input_array.copy(), output_array, x, axis, a)
        elif n == 3:
            fun_3D[complex](_cheb2leg_ptr, input_array.copy(), output_array, x, axis, a)
        elif n == 4:
            fun_4D[complex](_cheb2leg_ptr, input_array.copy(), output_array, x, axis, a)
    return output_array

cdef void fun_2D(funcX fun, T[:, ::1] ui, T[:, ::1] uo, double[:, ::1] data, int axis, double[:, ::1] a):
    cdef:
        int i, st
    st = ui.strides[axis]/ui.itemsize
    if axis == 0:
        for i in range(ui.shape[1]):
            fun(&ui[0, i], &uo[0, i], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])
    elif axis == 1:
        for i in range(ui.shape[0]):
            fun(&ui[i, 0], &uo[i, 0], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])

cdef void fun_3D(funcX fun, T[:, :, ::1] ui, T[:, :, ::1] uo, double[:, ::1] data, int axis, double[:, ::1] a):
    cdef:
        int i, j, k
        int st = ui.strides[axis]/ui.itemsize
    if axis == 0:
        for j in range(ui.shape[1]):
            for k in range(ui.shape[2]):
                fun(&ui[0, j, k], &uo[0, j, k], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])
    elif axis == 1:
        for i in range(ui.shape[0]):
            for k in range(ui.shape[2]):
                fun(&ui[i, 0, k], &uo[i, 0, k], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])
    elif axis == 2:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                fun(&ui[i, j, 0], &uo[i, j, 0], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])

cdef void fun_4D(funcX fun, T[:, :, :, ::1] ui, T[:, :, :, ::1] uo, double[:, ::1] data, int axis, double[:, ::1] a):
    cdef:
        int i, j, k, l
        int st = ui.strides[axis]/ui.itemsize
    if axis == 0:
        for j in range(ui.shape[1]):
            for k in range(ui.shape[2]):
                for l in range(ui.shape[3]):
                    fun(&ui[0, j, k, l], &uo[0, j, k, l], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])
    elif axis == 1:
        for i in range(ui.shape[0]):
            for k in range(ui.shape[2]):
                for l in range(ui.shape[3]):
                    fun(&ui[i, 0, k, l], &uo[i, 0, k, l], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])
    elif axis == 2:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                for l in range(ui.shape[3]):
                    fun(&ui[i, j, 0, l], &uo[i, j, 0, l], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])
    elif axis == 3:
        for i in range(ui.shape[0]):
            for j in range(ui.shape[1]):
                for k in range(ui.shape[2]):
                    fun(&ui[i, j, k, 0], &uo[i, j, k, 0], &data[0, 0], st, ui.shape[axis], uo.shape[axis], &a[0, 0], a.shape[0], a.shape[1])

cpdef _restricted_product(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _restricted_product_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], uo.shape[0], &a[0, 0], a.shape[0], a.shape[1])

cpdef _scalar_product(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _scalar_product_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], uo.shape[0], &a[0, 0], a.shape[0], a.shape[1])

cpdef _evaluate_expansion_all(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _evaluate_expansion_all_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], uo.shape[0], &a[0, 0], a.shape[0], a.shape[1])

cpdef _leg2cheb(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _leg2cheb_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], uo.shape[0], &a[0, 0], a.shape[0], a.shape[1])

cpdef _cheb2leg(T[:] ui, T[:] uo, double[:, ::1] data, double[:, ::1] a):
    _cheb2leg_ptr[T](&ui[0], &uo[0], &data[0, 0], ui.strides[0]/ui.itemsize, ui.shape[0], uo.shape[0], &a[0, 0], a.shape[0], a.shape[1])

@cython.cdivision(True)
cdef void _evaluate_expansion_all_ptr(T* ui,
                                      T* uo,
                                      double* data,
                                      int st,
                                      int N,
                                      int Nx,
                                      double* a,
                                      int M,
                                      int Mx):
    cdef:
        int i, j
        T s
        double s1, s2, a00
        double* xj = &data[0]
        double* anm = &a[0]
        double* anp
        double* ann
        array.array[double] an = array.clone(darray, Mx, zero=True)
        array.array[double] Lnm = array.clone(darray, Nx, zero=False)
        array.array[double] Ln = array.clone(darray, Nx, zero=False)
        array.array[double] Lnp = array.clone(darray, Nx, zero=False)

    if M == 2:
        ann = an.data.as_doubles
        anp = &a[Mx]
    else:
        ann = &a[Mx]
        anp = &a[2*Mx]
    for i in range(Nx):
        Lnm[i] = 1
        Ln[i] = (xj[i]-ann[0])/anm[0]
        uo[i*st] = 0
        Lnp[i] = (xj[i]-ann[1])/anm[1]*Ln[i] - anp[1]/anm[1]*Lnm[i]
    for i in range(N):
        s1 = 1/anm[i+2]
        s2 = anp[i+2]/anm[i+2]
        a00 = ann[i+2]
        for j in range(Nx):
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
                              int Nx,
                              double* a,
                              int M,
                              int Mx):
    cdef:
        int i, j
        T s
        double s1, s2
        double* xj = &data[0]
        double* wj = &data[N]
        double* anm = &a[0]
        double* ann
        double* anp
        array.array[double] an = array.clone(darray, Mx, zero=True)
        array.array[double] Lnm = array.clone(darray, Nx, zero=False)
        array.array[double] Ln = array.clone(darray, Nx, zero=False)
        array.array[double] Lnp = array.clone(darray, Nx, zero=False)

    if M == 2:
        ann = an.data.as_doubles
        anp = &a[Mx]
    else:
        ann = &a[Mx]
        anp = &a[2*Mx]

    for i in range(Nx):
        Lnm[i] = 1
        Ln[i] = (xj[i]-ann[0])/anm[0]
        uo[i*st] = 0
        Lnp[i] = (xj[i]-ann[1])/anm[1]*Ln[i] - anp[1]/anm[1]*Lnm[i]

    for i in range(N):
        s1 = 1/anm[i+2]
        s2 = anp[i+2]/anm[i+2]
        a00 = ann[i+2]
        s = 0.0
        for j in range(Nx):
            s += Lnm[j]*wj[j]*ui[j*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]
        uo[i*st] = s

@cython.cdivision(True)
cdef void _leg2cheb_ptr(T* c,
                        T* v,
                        double* transpose,
                        int st,
                        int N,
                        int Nx,
                        double* a,
                        int M,
                        int Mx):
    cdef:
        int n
        vector[T] cx

    if transpose[0] < 0.5:
        for n in range(0, N, 2):
            for i in range(0, N-n):
                v[i*st] += a[n//2]*a[n//2+i]*c[(n+i)*st]
        v[0] /= 2
        for n in range(0, N):
            v[n*st] *= (2/np.pi)
    else:
        cx.resize(N)
        for i in range(0, N):
            cx[i] = c[i*st]*2/np.pi
        cx[0] /= 2
        for n in range(0, N, 2):
            for i in range(0, N-n):
                v[(i+n)*st] += a[n//2]*a[n//2+i]*cx[i]


@cython.cdivision(True)
cdef void _cheb2leg_ptr(T* v,
                        T* c,
                        double* dn,
                        int st,
                        int N,
                        int Nx,
                        double* a,
                        int M,
                        int Mx):
    cdef:
        int n
        double SPI = np.sqrt(np.pi)

    for n in range(1, N):
        v[n*st] = v[n*st]*n
    for n in range(N):
        c[n*st] = SPI*a[n]*v[n*st]
    for n in range(2, N, 2):
        for i in range(0, N-n):
            c[i*st] -= dn[n//2]*a[n//2+i]*v[(n+i)*st]
    for n in range(N):
        c[n*st] *= (n+0.5)

@cython.cdivision(True)
cdef void _restricted_product_ptr(T* input_array,
                                  T* output_array,
                                  double* data,
                                  int st,
                                  int N,
                                  int No,
                                  double* a,
                                  int M,
                                  int Mx):

    cdef:
        int k, kp, i
        double* xi = &data[0]
        double* anm = &a[0]
        double* ann
        double* anp
        array.array[double] an = array.clone(darray, Mx, zero=True)
        array.array[double] Lnm = array.clone(darray, N, zero=False)
        array.array[double] Ln = array.clone(darray, N, zero=False)
        array.array[double] Lnp = array.clone(darray, N, zero=False)

    if M == 2:
        ann = an.data.as_doubles
        anp = &a[Mx]
    else:
        ann = &a[Mx]
        anp = &a[2*Mx]
    for i in range(N):
        Lnm[i] = data[N+i]
        Ln[i] = data[2*N+i]
        Lnp[i] = data[3*N+i]
    for k in range(No):
        s1 = 1/anm[k+2]
        s2 = anp[k+2]/anm[k+2]
        a00 = ann[k+2]
        s = 0.0
        for i in range(N):
            s += Lnm[i]*input_array[i]
            Lnm[i] = Ln[i]
            Ln[i] = Lnp[i]
            Lnp[i] = s1*(xi[i]-a00)*Ln[i] - s2*Lnm[i]
        output_array[k] = s
