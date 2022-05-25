import numba as nb
import numpy as np
from .la import Solve_axis_2D, Solve_axis_3D, Solve_axis_4D

__all__ = ['TDMA_LU', 'TDMA_Solve',
           'TDMA_O_LU', 'TDMA_O_Solve',
           'TDMA_inner_solve', 'TDMA_O_inner_solve']

def TDMA_Solve(x, data, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_inner_solve(x, data)
    elif n == 2:
        Solve_axis_2D(data, x, TDMA_inner_solve, axis)
    elif n == 3:
        Solve_axis_3D(data, x, TDMA_inner_solve, axis)
    elif n == 4:
        Solve_axis_4D(data, x, TDMA_inner_solve, axis)
    else:
        if axis > 0:
            x = np.moveaxis(x, axis, 0)
        TDMA_inner_solve(x, data)
        if axis > 0:
            x = np.moveaxis(x, 0, axis)

def TDMA_O_Solve(x, data, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_O_inner_solve(x, data)
    elif n == 2:
        Solve_axis_2D(data, x, TDMA_O_inner_solve, axis)
    elif n == 3:
        Solve_axis_3D(data, x, TDMA_O_inner_solve, axis)
    elif n == 4:
        Solve_axis_4D(data, x, TDMA_O_inner_solve, axis)
    else:
        if axis > 0:
            x = np.moveaxis(x, axis, 0)
        TDMA_O_inner_solve(x, data)
        if axis > 0:
            x = np.moveaxis(x, 0, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_LU(data):
    ld = data[0, :-2]
    d = data[1, :]
    ud = data[2, 2:]
    n = d.shape[0]
    for i in range(2, n):
        ld[i-2] = ld[i-2]/d[i-2]
        d[i] -= ld[i-2]*ud[i-2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_LU(data):
    ld = data[0, :-1]
    d = data[1, :]
    ud = data[2, 1:]
    n = d.shape[0]
    for i in range(1, n):
        ld[i-1] = ld[i-1]/d[i-1]
        d[i] -= ld[i-1]*ud[i-1]

@nb.njit
def TDMA_inner_solve(u, data):
    ld = data[0, :-2]
    d = data[1, :]
    ud = data[2, 2:]
    n = d.shape[0]
    for i in range(2, n):
        u[i] -= ld[i-2]*u[i-2]
    u[n-1] = u[n-1]/d[n-1]
    u[n-2] = u[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        u[i] = (u[i] - ud[i]*u[i+2])/d[i]

@nb.njit
def TDMA_O_inner_solve(u, data):
    ld = data[0, :-1]
    d = data[1, :]
    ud = data[2, 1:]
    n = d.shape[0]
    for i in range(1, n):
        u[i] -= ld[i-1]*u[i-1]
    u[n-1] = u[n-1]/d[n-1]
    for i in range(n-2, -1, -1):
        u[i] = (u[i] - ud[i]*u[i+1])/d[i]
