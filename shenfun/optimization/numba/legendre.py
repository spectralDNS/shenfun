import numba as nb
import numpy as np

__all__ = ['scalar_product', 'evaluate_expansion_all']

def scalar_product(input_array, output_array, x, w, axis):
    n = input_array.ndim
    if n == 1:
        _scalar_product(input_array, output_array, x, w)
    elif n == 2:
        fun_2D(_scalar_product, input_array, output_array, axis, x, w)
    elif n == 3:
        fun_3D(_scalar_product, input_array, output_array, axis, x, w)
    elif n == 4:
        fun_4D(_scalar_product, input_array, output_array, axis, x, w)

def evaluate_expansion_all(input_array, output_array, x, axis):
    n = input_array.ndim
    if n == 1:
        _evaluate_expansion_all(input_array, output_array, x)
    elif n == 2:
        fun_2D(_evaluate_expansion_all, input_array, output_array, axis, x)
    elif n == 3:
        fun_3D(_evaluate_expansion_all, input_array, output_array, axis, x)
    elif n == 4:
        fun_4D(_evaluate_expansion_all, input_array, output_array, axis, x)

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
def _scalar_product(input_array, output_array, xj, wj):
    N = xj.shape[0]
    Lnm = np.ones_like(xj)
    Ln = xj.copy()
    Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
    for i in range(N):
        s = 0.0
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        for j in range(N):
            s += Lnm[j]*wj[j]*input_array[j]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]
        output_array[i] = s

@nb.jit(nopython=True, fastmath=True, cache=True)
def _evaluate_expansion_all(input_array, output_array, xj):
    N = xj.shape[0]
    Lnm = np.ones_like(xj)
    Ln = xj.copy()
    Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
    output_array[:] = 0
    for i in range(N):
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        for j in range(N):
            output_array[j] += Lnm[j]*input_array[i]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]
