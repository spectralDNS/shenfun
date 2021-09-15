import numba as nb

__all__ = ['DiagMA_inner_solve', 'DiagMA_Solve']

@nb.njit
def DiagMA_inner_solve(u, data):
    d = data[0]
    for i in range(d.shape[0]):
        u[i] /= d[i]

@nb.njit
def DiagMA_Solve1D(u, data):
    for i in range(data.shape[0]):
        u[i] /= data[i]
    return u

def DiagMA_Solve(u, data, axis=0):
    n = u.ndim
    if n == 1:
        DiagMA_Solve1D(u, data[0])
    elif n == 2:
        DiagMA_Solve2D(u, data[0], axis)
    elif n == 3:
        DiagMA_Solve3D(u, data[0], axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def DiagMA_Solve2D(u, d, axis):
    if axis == 0:
        for j in range(u.shape[1]):
            DiagMA_Solve1D(u[:, j], d)
    elif axis == 1:
        for i in range(u.shape[0]):
            DiagMA_Solve1D(u[i], d)

@nb.jit(nopython=True, fastmath=True, cache=True)
def DiagMA_Solve3D(u, d, axis):
    if axis == 0:
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                DiagMA_Solve1D(u[:, j, k], d)
    elif axis == 1:
        for i in range(u.shape[0]):
            for k in range(u.shape[2]):
                DiagMA_Solve1D(u[i, :, k], d)
    elif axis == 2:
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                DiagMA_Solve1D(u[i, j], d)
