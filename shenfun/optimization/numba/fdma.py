import numba as nb

__all__ = ['FDMA_LU', 'FDMA_Solve', 'TwoDMA_Solve']

@nb.jit(nopython=True, fastmath=True, cache=True)
def FDMA_LU(ld, d, u1, u2):
    n = d.shape[0]
    for i in range(2, n):
        ld[i-2] = ld[i-2]/d[i-2]
        d[i] = d[i] - ld[i-2]*u1[i-2]
        if i < n-2:
            u1[i] = u1[i] - ld[i-2]*u2[i-2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def FDMA_Solve(d, u1, u2, l, x, axis=0):
    n = d.shape[0]
    for i in range(2, n):
        x[i] -= l[i-2]*x[i-2]
    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    x[n-3] = (x[n-3] - u1[n-3]*x[n-1])/d[n-3]
    x[n-4] = (x[n-4] - u1[n-4]*x[n-2])/d[n-4]
    for i in range(n - 5, -1, -1):
        x[i] = (x[i] - u1[i]*x[i+2] - u2[i]*x[i+4])/d[i]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TwoDMA_Solve(d, u1, x, axis=0):
    n = d.shape[0]
    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - u1[i]*x[i+2])/d[i]
