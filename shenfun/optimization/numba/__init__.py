import numba as nb
from .tdma import *
from .pdma import *
from .helmholtz import *
from .biharmonic import *

@nb.jit(nopython=True, fastmath=True, cache=True)
def outer2D(a, b, c, symmetric):
    N, M = a.shape[1:]
    if symmetric:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]**2           # (0, 0)
                c[1, i, j] = a[0, i, j]*a[1, i, j]   # (0, 1)
                c[2, i, j] = c[1, i, j]              # (1, 0)
                c[3, i, j] = a[1, i, j]**2           # (1, 1)
    else:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]*b[0, i, j]   # (0, 0)
                c[1, i, j] = a[0, i, j]*b[1, i, j]   # (0, 1)
                c[2, i, j] = a[1, i, j]*b[0, i, j]   # (1, 0)
                c[3, i, j] = a[1, i, j]*b[1, i, j]   # (1, 1)


@nb.jit(nopython=True, fastmath=True, cache=True)
def outer3D(a, b, c, symmetric):
    N, M = a.shape[1:]
    if symmetric:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]**2           # (0, 0)
                c[1, i, j] = a[0, i, j]*a[1, i, j]   # (0, 1)
                c[2, i, j] = a[0, i, j]*a[2, i, j]   # (0, 2)
                c[3, i, j] = c[1, i, j]              # (1, 0)
                c[4, i, j] = a[1, i, j]**2           # (1, 1)
                c[5, i, j] = a[1, i, j]*a[2, i, j]   # (1, 2)
                c[6, i, j] = c[2, i, j]              # (2, 0)
                c[7, i, j] = c[5, i, j]              # (2, 1)
                c[8, i, j] = a[2, i, j]**2           # (2, 2)
    else:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]*b[0, i, j]   # (0, 0)
                c[1, i, j] = a[0, i, j]*b[1, i, j]   # (0, 1)
                c[2, i, j] = a[0, i, j]*b[2, i, j]   # (0, 2)
                c[3, i, j] = a[1, i, j]*b[0, i, j]   # (1, 0)
                c[4, i, j] = a[1, i, j]*b[1, i, j]   # (1, 1)
                c[5, i, j] = a[1, i, j]*b[2, i, j]   # (1, 2)
                c[6, i, j] = a[2, i, j]*b[0, i, j]   # (2, 0)
                c[7, i, j] = a[2, i, j]*b[1, i, j]   # (2, 1)
                c[8, i, j] = a[2, i, j]*b[2, i, j]   # (2, 2)
