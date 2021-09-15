import numba as nb
import numpy as np

__all__ = ['FDMA_LU', 'FDMA_Solve', 'FDMA_inner_solve']

@nb.jit(nopython=True, fastmath=True, cache=True)
def FDMA_LU(data):
    ld = data[0, :-2]
    d = data[1, :]
    u1 = data[2, 2:]
    u2 = data[3, 4:]
    n = int(d.shape[0])
    for i in range(2, n):
        ld[i-2] = ld[i-2]/d[i-2]
        d[i] -= ld[i-2]*u1[i-2]
        if i < n-2:
            u1[i] -= ld[i-2]*u2[i-2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def FDMA_Solve(x, data, axis=0):
    ld = data[0, :-2]
    d = data[1, :]
    u1 = data[2, 2:]
    u2 = data[3, 4:]
    n = d.shape[0]
    for i in range(2, n):
        x[i] -= ld[i-2]*x[i-2]
    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    x[n-3] = (x[n-3] - u1[n-3]*x[n-1])/d[n-3]
    x[n-4] = (x[n-4] - u1[n-4]*x[n-2])/d[n-4]
    for i in range(n - 5, -1, -1):
        x[i] = (x[i] - u1[i]*x[i+2] - u2[i]*x[i+4])/d[i]

@nb.njit
def FDMA_inner_solve(u, data):
    ld = data[0, :-2]
    d = data[1, :]
    u1 = data[2, 2:]
    u2 = data[3, 4:]
    n = d.shape[0]
    for i in range(2, n):
        u[i] -= ld[i-2]*u[i-2]
    u[n-1] = u[n-1]/d[n-1]
    u[n-2] = u[n-2]/d[n-2]
    u[n-3] = (u[n-3] - u1[n-3]*u[n-1])/d[n-3]
    u[n-4] = (u[n-4] - u1[n-4]*u[n-2])/d[n-4]
    for i in range(n - 5, -1, -1):
        u[i] = (u[i] - u1[i]*u[i+2] - u2[i]*u[i+4])/d[i]
