import numba as nb
import numpy as np

__all__ = ['scalar_product', 'evaluate_expansion_all']

def scalar_product(input_array, output_array, x, w, axis, a):
    n = input_array.ndim
    if n == 1:
        _scalar_product(input_array, output_array, x, w, a)
    elif n == 2:
        fun_2D(_scalar_product, input_array, output_array, axis, x, w, a)
    elif n == 3:
        fun_3D(_scalar_product, input_array, output_array, axis, x, w, a)
    elif n == 4:
        fun_4D(_scalar_product, input_array, output_array, axis, x, w, a)
    else:
        if axis > 0:
            input_array = np.moveaxis(input_array, axis, 0)
            output_array = np.moveaxis(output_array, axis, 0)
        _scalar_product(input_array, output_array, x, w, a)
        if axis > 0:
            input_array = np.moveaxis(input_array, 0, axis)
            output_array = np.moveaxis(output_array, 0, axis)

def evaluate_expansion_all(input_array, output_array, x, axis, a):
    n = input_array.ndim
    if n == 1:
        _evaluate_expansion_all(input_array, output_array, x, a)
    elif n == 2:
        fun_2D(_evaluate_expansion_all, input_array, output_array, axis, x, a)
    elif n == 3:
        fun_3D(_evaluate_expansion_all, input_array, output_array, axis, x, a)
    elif n == 4:
        fun_4D(_evaluate_expansion_all, input_array, output_array, axis, x, a)
    else:
        if axis > 0:
            input_array = np.moveaxis(input_array, axis, 0)
            output_array = np.moveaxis(output_array, axis, 0)
        _evaluate_expansion_all(input_array, output_array, x, a)
        if axis > 0:
            input_array = np.moveaxis(input_array, 0, axis)
            output_array = np.moveaxis(output_array, 0, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def fun_2D(fun, input_array, output_array, axis, *args):
    if axis == 0:
        for j in range(input_array.shape[1]):
            fun(input_array[:, j], output_array[:, j], *args)
    elif axis == 1:
        for i in range(input_array.shape[0]):
            fun(input_array[i], output_array[i], *args)

@nb.jit(nopython=True, fastmath=True, cache=True)
def fun_3D(fun, input_array, output_array, axis, *args):
    if axis == 0:
        for j in range(input_array.shape[1]):
            for k in range(input_array.shape[2]):
                fun(input_array[:, j, k], output_array[:, j, k], *args)
    elif axis == 1:
        for i in range(input_array.shape[0]):
            for k in range(input_array.shape[2]):
                fun(input_array[i, :, k], output_array[i, :, k], *args)
    elif axis == 2:
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                fun(input_array[i, j], output_array[i, j], *args)

@nb.jit(nopython=True, fastmath=True, cache=True)
def fun_4D(fun, input_array, output_array, axis, *args):
    if axis == 0:
        for j in range(input_array.shape[1]):
            for k in range(input_array.shape[2]):
                for l in range(input_array.shape[3]):
                    fun(input_array[:, j, k, l], output_array[:, j, k, l], *args)
    elif axis == 1:
        for i in range(input_array.shape[0]):
            for k in range(input_array.shape[2]):
                for l in range(input_array.shape[3]):
                    fun(input_array[i, :, k, l], output_array[i, :, k, l], *args)
    elif axis == 2:
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                for l in range(input_array.shape[3]):
                    fun(input_array[i, j, :, l], output_array[i, j, :, l], *args)
    elif axis == 3:
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                for k in range(input_array.shape[2]):
                    fun(input_array[i, j, k, :], output_array[i, j, k, :], *args)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _scalar_product(input_array, output_array, xj, wj, a):
    M = output_array.shape[0]
    N = xj.shape[0]
    Lnm = np.ones(N)
    if a.shape[0] == 3:
        anm = a[0] # a_{n-1, n}
        ann = a[1] # a_{n, n}
        anp = a[2] # a_{n+1, n}
    else:
        anm = a[0]
        anp = a[1]
        ann = np.zeros(N+2)
    Ln = (xj-ann[0])/anm[0]
    Lnp = (xj-ann[1])/anm[1]*Ln - anp[1]/anm[1]*Lnm
    for k in range(M):
        s1 = 1/anm[k+2]
        s2 = anp[k+2]/anm[k+2]
        a00 = ann[k+2]
        s = 0.0
        for j in range(N):
            s += Lnm[j]*wj[j]*input_array[j]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]
        output_array[k] = s

@nb.jit(nopython=True, fastmath=True, cache=True)
def _evaluate_expansion_all(input_array, output_array, xj, a):
    M = input_array.shape[0]
    N = output_array.shape[0]
    Lnm = np.ones(N)
    if a.shape[0] == 3:
        anm = a[0] # a_{n-1, n}
        ann = a[1] # a_{n, n}
        anp = a[2] # a_{n+1, n}
    else:
        anm = a[0]
        anp = a[1]
        ann = np.zeros(N+2)
    Ln = (xj-ann[0])/anm[0]
    Lnp = (xj-ann[1])/anm[1]*Ln - anp[1]/anm[1]*Lnm
    output_array[:] = 0
    for k in range(M):
        s1 = 1/anm[k+2]
        s2 = anp[k+2]/anm[k+2]
        a00 = ann[k+2]
        for j in range(N):
            output_array[j] += Lnm[j]*input_array[k]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]
