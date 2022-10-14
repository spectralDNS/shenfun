import numba as nb
import numpy as np
from scipy.special import gammaln

__all__ = ['scalar_product', 'evaluate_expansion_all', 'cheb2leg', 'leg2cheb', 'restricted_product']

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

def restricted_product(space, input_array, output_array, x, i0, i1, a0, axis, a):
    n = input_array.ndim
    Lnm = space.evaluate_basis(x[i0:i1], i=a0)
    Ln = space.evaluate_basis(x[i0:i1], i=a0+1)
    if n == 1:
        _restricted_product(input_array, output_array, x, Lnm, Ln, i0, i1, a0, a)
    elif n == 2:
        fun_2D(_restricted_product, input_array, output_array, axis, x, Lnm, Ln, i0, i1, a0, a)
    elif n == 3:
        fun_3D(_restricted_product, input_array, output_array, axis, x, Lnm, Ln, i0, i1, a0, a)
    elif n == 4:
        fun_4D(_restricted_product, input_array, output_array, axis, x, Lnm, Ln, i0, i1, a0, a)
    else:
        if axis > 0:
            input_array = np.moveaxis(input_array, axis, 0)
            output_array = np.moveaxis(output_array, axis, 0)
        _restricted_product(input_array, output_array, x, Lnm, Ln, i0, i1, a0, a)
        if axis > 0:
            input_array = np.moveaxis(input_array, 0, axis)
            output_array = np.moveaxis(output_array, 0, axis)
    return output_array

@nb.jit(nopython=True, fastmath=True, cache=False)
def fun_2D(fun, input_array, output_array, axis, *args):
    if axis == 0:
        for j in range(input_array.shape[1]):
            fun(input_array[:, j], output_array[:, j], *args)
    elif axis == 1:
        for i in range(input_array.shape[0]):
            fun(input_array[i], output_array[i], *args)

@nb.jit(nopython=True, fastmath=True, cache=False)
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

@nb.jit(nopython=True, fastmath=True, cache=False)
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

@nb.jit(nopython=True, fastmath=True, cache=False)
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

@nb.jit(nopython=True, fastmath=True, cache=False)
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
        ann = np.zeros(M+2)
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

@nb.jit(nopython=True, fastmath=True, cache=False)
def _restricted_product(input_array, output_array, xj, Lnm0, Ln0, i0, i1, a0, a):
    N = xj.shape[0]
    Lnm = Lnm0.copy()
    Ln = Ln0.copy()
    if a.shape[0] == 3:
        anm = a[0] # a_{n-1, n}
        ann = a[1] # a_{n, n}
        anp = a[2] # a_{n+1, n}
    else:
        anm = a[0]
        anp = a[1]
        ann = np.zeros(N+2)
    Lnp = (xj[i0:i1]-ann[1+a0])/anm[1+a0]*Ln - anp[1+a0]/anm[1+a0]*Lnm
    for k in range(len(output_array)):
        kp = k+a0
        s1 = 1/anm[kp+2]
        s2 = anp[kp+2]/anm[kp+2]
        a00 = ann[kp+2]
        s = 0.0
        for j in range(len(Ln)):
            s += Lnm[j]*input_array[i0+j]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[i0+j]-a00)*Ln[j] - s2*Lnm[j]
        output_array[k] = s

Omega = lambda z: np.exp(gammaln(z+0.5) - gammaln(z+1))

def leg2cheb(input_array, output_array, axis=0, transpose=False):
    n = input_array.ndim
    N = input_array.shape[axis]
    k = np.arange(N)
    a = Omega(k)
    output_array.fill(0)
    if n == 1:
        _leg2cheb(input_array, output_array, a, N, transpose)
    elif n == 2:
        fun_2D(_leg2cheb, input_array, output_array, axis, a, N, transpose)
    elif n == 3:
        fun_3D(_leg2cheb, input_array, output_array, axis, a, N, transpose)
    elif n == 4:
        fun_4D(_leg2cheb, input_array, output_array, axis, a, N, transpose)
    else:
        if axis > 0:
            input_array = np.moveaxis(input_array, axis, 0)
            output_array = np.moveaxis(output_array, axis, 0)
        _leg2cheb(input_array, output_array, a, N, transpose)
        if axis > 0:
            input_array = np.moveaxis(input_array, 0, axis)
            output_array = np.moveaxis(output_array, 0, axis)
    return output_array

@nb.jit(nopython=True, fastmath=True, cache=False)
def _leg2cheb(c, v, a, N, transpose):
    if transpose is False:
        for n in range(0, N, 2):
            v[:(N-n)] += a[n//2]*a[n//2:(N-n//2)]*c[n:]
        v[0] /= 2
        v *= 2/np.pi
    else:
        cx = c.copy()
        cx[0] /= 2
        cx *= 2/np.pi
        for n in range(0, N, 2):
            v[n:] += a[n//2]*a[n//2:(N-n//2)]*cx[:(N-n)]

def cheb2leg(input_array, output_array, axis=0):
    n = input_array.ndim
    N = input_array.shape[axis]
    k = np.arange(N)
    k[0] = 1
    dn = Omega((k[::2]-2)/2)/k[::2]
    a = 1/(2*Omega(k)*k*(k+0.5))
    a[0] = 2/np.sqrt(np.pi)
    output_array.fill(0)
    if n == 1:
        _cheb2leg(input_array, output_array, a, dn, N)
    elif n == 2:
        fun_2D(_cheb2leg, input_array, output_array, axis, a, dn, N)
    elif n == 3:
        fun_3D(_cheb2leg, input_array, output_array, axis, a, dn, N)
    elif n == 4:
        fun_4D(_cheb2leg, input_array, output_array, axis, a, dn, N)
    else:
        if axis > 0:
            input_array = np.moveaxis(input_array, axis, 0)
            output_array = np.moveaxis(output_array, axis, 0)
        _cheb2leg(input_array, output_array, a, dn, N)
        if axis > 0:
            input_array = np.moveaxis(input_array, 0, axis)
            output_array = np.moveaxis(output_array, 0, axis)

@nb.jit(nopython=True, fastmath=True, cache=False)
def _cheb2leg(v, c, a, dn, N):
    vn = v.copy()
    for i in range(1, N):
        vn[i] = v[i]*i
    c[:] = np.sqrt(np.pi)*a*vn
    for n in range(2, N, 2):
        c[:(N-n)] -= dn[n//2]*a[n//2:(N-n//2)]*vn[n:]
    for i in range(N):
        c[i] *= (i+0.5)
