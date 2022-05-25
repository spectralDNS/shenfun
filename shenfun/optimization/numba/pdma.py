import numba as nb
import numpy as np
from .la import Solve_axis_2D, Solve_axis_3D, Solve_axis_4D

__all__ = ['PDMA_LU', 'PDMA_Solve', 'PDMA_inner_solve']

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_LU(data):
    """LU decomposition"""
    a = data[0, :-4]
    b = data[1, :-2]
    d = data[2, :]
    e = data[3, 2:]
    f = data[4, 4:]
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

def PDMA_Solve(x, data, axis=0):
    n = x.ndim
    if n == 1:
        PDMA_inner_solve(x, data)
    elif n == 2:
        Solve_axis_2D(data, x, PDMA_inner_solve, axis)
    elif n == 3:
        Solve_axis_3D(data, x, PDMA_inner_solve, axis)
    elif n == 4:
        Solve_axis_4D(data, x, PDMA_inner_solve, axis)
    else:
        if axis > 0:
            x = np.moveaxis(x, axis, 0)
        PDMA_inner_solve(x, data)
        if axis > 0:
            x = np.moveaxis(x, 0, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_inner_solve(u, data):
    a = data[0, :-4]
    b = data[1, :-2]
    d = data[2, :]
    e = data[3, 2:]
    f = data[4, 4:]
    n = d.shape[0]
    u[2] -= b[0]*u[0]
    u[3] -= b[1]*u[1]
    for k in range(4, n):
        u[k] -= (b[k-2]*u[k-2] + a[k-4]*u[k-4])
    u[n-1] /= d[n-1]
    u[n-2] /= d[n-2]
    u[n-3] = (u[n-3]-e[n-3]*u[n-1])/d[n-3]
    u[n-4] = (u[n-4]-e[n-4]*u[n-2])/d[n-4]
    for k in range(n-5, -1, -1):
        u[k] = (u[k]-e[k]*u[k+2]-f[k]*u[k+4])/d[k]
