import warnings
from functools import wraps
import numba as nb
import numpy as np
from scipy.special import gammaln

warnings.simplefilter('ignore', category=nb.core.errors.NumbaPerformanceWarning)

__all__ = ['scalar_product', 'evaluate_expansion_all', 'cheb2leg', 'leg2cheb',
           'restricted_product', '_leg2cheb', 'FMMdirect1', 'FMMdirect2',
           'FMMdirect3', 'FMMdirect4']

def NDim(func):
    @wraps(func)
    def wrapped_function(input_array, output_array, axis, *args):
        n = input_array.ndim
        if n == 1:
            func(input_array, output_array, *args)
        elif n == 2:
            fun_2D(func, input_array, output_array, axis, *args)
        elif n == 3:
            fun_3D(func, input_array, output_array, axis, *args)
        elif n == 4:
            fun_4D(func, input_array, output_array, axis, *args)
    return wrapped_function

@nb.jit(nopython=True, fastmath=True, cache=False)
def _matvec(A, x, b, m, n, transpose):
    if transpose == 0:
        for i in range(m):
            s = 0.0
            a = A[i]
            for j in range(n):
                s = s+a[j]*x[j]
            b[i] = s
    else:
        for j in range(n):
            b[j] = 0
        for i in range(m):
            s = x[i]
            a = A[i]
            for j in range(n):
                b[j] = b[j] + s*a[j]
    return b

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

@nb.jit(nopython=True, fastmath=True, cache=False)
def _leg2cheb(c, v, a, N, transpose):
    M_2_PI = 2/np.pi
    if transpose is False:
        for n in range(0, N, 2):
            #v[:(N-n)] += a[n//2]*a[n//2:(N-n//2)]*c[n:]
            nh = n//2
            for i in range(N-n):
                v[i] += a[nh]*a[nh+i]*c[n+i]
        v[0] /= 2
        v *= M_2_PI
    else:
        cv = c*M_2_PI
        cv[0] /= 2
        for n in range(0, N, 2):
            nh = n//2
            for i in range(N-n):
                v[i+n] += a[nh]*a[nh+i]*cv[i]
            #v[n:] += (a[n//2]*M_2_PI)*a[n//2:(N-n//2)]*c[:(N-n)]

def leg2cheb(input_array, output_array=None, axis=0, transpose=False):
    if output_array is None:
        output_array = np.zeros_like(input_array)
    else:
        output_array.fill(0)
    n = input_array.ndim
    N = input_array.shape[axis]
    k = np.arange(N)
    a = Omega(k)
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
def _cheb2leg(v, c, a, dn, N):
    vn = v.copy()
    for i in range(1, N):
        vn[i] = v[i]*i
    c[:] = np.sqrt(np.pi)*a*vn
    for n in range(2, N, 2):
        c[:(N-n)] -= dn[n//2]*a[n//2:(N-n//2)]*vn[n:]
    for i in range(N):
        c[i] *= (i+0.5)

def cheb2leg(input_array, output_array=None, axis=0):
    if output_array is None:
        output_array = np.zeros_like(input_array)
    else:
        output_array.fill(0)
    n = input_array.ndim
    N = input_array.shape[axis]
    k = np.arange(N)
    k[0] = 1
    dn = Omega((k[::2]-2)/2)/k[::2]
    a = 1/(2*Omega(k)*k*(k+0.5))
    a[0] = 2/np.sqrt(np.pi)
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
    return output_array


@NDim
@nb.jit(nopython=True, fastmath=True, cache=True)
def FMMdirect1(u, v, a, n0, trans):
    N = u.shape[0]
    if trans is False:
        for n in range(0, n0):
            for i in range(0, N-2*n):
                v[i] += a[n]*a[n+i]*u[2*n+i]
    else:
        for n in range(0, n0):
            for i in range(N-2*n):
                v[i+2*n] += a[n]*a[n+i]*u[i]

@NDim
@nb.jit(nopython=True, fastmath=True, cache=True)
def FMMdirect2(u, v, a, h, Nd, n0, trans):
    N = v.shape[0]
    for k in range(Nd):
        i0 = k*h
        j0 = n0+i0
        for n in range(0, h, 2):
            a0 = i0+(n0+n)//2
            if trans is True:
                for j in range(min(N-(j0+n), h-n)):
                    v[j0+n+j] += a[(n+n0)//2]*a[a0+j]*u[i0+j]
            else:
                for j in range(min(N-(j0+n), h-n)):
                    v[i0+j] += a[(n+n0)//2]*a[a0+j]*u[j0+n+j]

@NDim
@nb.jit(nopython=True, fastmath=True, cache=True)
def FMMdirect3(u, v, dn, a, h, Nd, n0):
    N = v.shape[0]
    for k in range(Nd):
        i0 = k*h
        j0 = n0+i0
        for n in range(0, h, 2):
            a0 = i0+(n0+n)//2
            for j in range(min(N-(j0+n), h-n)):
                v[i0+j] -= dn[(n+n0)//2]*a[a0+j]*u[j0+n+j]

@NDim
@nb.jit(nopython=True, fastmath=True, cache=True)
def FMMdirect4(u, v, dn, a, n0):
    N = u.shape[0]
    v[:] += np.sqrt(np.pi)*a*u
    for n in range(1, n0):
        for i in range(0, N-2*n):
            v[i] -= dn[n]*a[n+i]*u[2*n+i]
