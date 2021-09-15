import numba as nb

__all__ = ['TwoDMA_Solve', 'TwoDMA_inner_solve']

@nb.jit(nopython=True, fastmath=True, cache=True)
def TwoDMA_Solve(x, data, axis=0):
    d = data[0, :]
    u1 = data[1, 2:]
    n = d.shape[0]
    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - u1[i]*x[i+2])/d[i]

@nb.njit
def TwoDMA_inner_solve(u, data):
    d = data[0, :]
    u1 = data[1, 2:]
    n = d.shape[0]
    u[n-1] = u[n-1]/d[n-1]
    u[n-2] = u[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        u[i] = (u[i] - u1[i]*u[i+2])/d[i]