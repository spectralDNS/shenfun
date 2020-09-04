import numpy as np
from numba import jit

__all__ = ['chebval']

def chebval(x, c):
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    y = np.zeros_like(x)
    c0 = np.zeros_like(x)
    c1 = np.zeros_like(x)
    tmp = np.zeros_like(x)
    x2 = np.zeros_like(x)
    y = _chebval(x, c, y, c0, c1, tmp, x2)
    return y

@jit
def _chebval(x, c, y, c0, c1, tmp, x2):
    N = c.shape[0]
    M = x.shape[0]
    if N == 1:
        c0[:] = c[0]
        c1[:] = 0
    elif N == 2:
        c0[:] = c[0]
        c1[:] = c[1]
    else:
        for j in range(M):
            x2[j] = 2*x[j]
            c0[j] = c[-2]
            c1[j] = c[-1]
        for i in range(3, N + 1):
            for j in range(M):
                tmp[j] = c0[j]
                c0[j] = c[-i] - c1[j]
                c1[j] = tmp[j] + c1[j]*x2[j]
    for j in range(M):
        y[j] = c0[j] + x[j]*c1[j]
    return y
