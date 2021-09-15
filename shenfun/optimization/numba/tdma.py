import numba as nb

__all__ = ['TDMA_LU', 'TDMA_Solve',
           'TDMA_O_LU', 'TDMA_O_Solve',
           'TDMA_inner_solve', 'TDMA_O_inner_solve']

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_LU(data):
    ld = data[0, :-2]
    d = data[1, :]
    ud = data[2, 2:]
    n = d.shape[0]
    for i in range(2, n):
        ld[i-2] = ld[i-2]/d[i-2]
        d[i] -= ld[i-2]*ud[i-2]

def TDMA_Solve(x, data, axis=0):
    ld = data[0, :-2]
    d = data[1, :]
    ud = data[2, 2:]
    n = x.ndim
    if n == 1:
        TDMA_Solve1D(ld, d, ud, x)
    elif n == 2:
        TDMA_Solve2D(ld, d, ud, x, axis)
    elif n == 3:
        TDMA_Solve3D(ld, d, ud, x, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_Solve1D(ld, d, ud, x):
    n = d.shape[0]
    for i in range(2, n):
        x[i] -= ld[i-2]*x[i-2]

    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - ud[i]*x[i+2])/d[i]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_Solve2D(ld, d, ud, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                x[i, j] -= ld[i-2]*x[i-2, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1]
            x[n-2, j] = x[n-2, j]/d[n-2]

        for i in range(n - 3, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - ud[i]*x[i+2, j])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            TDMA_Solve1D(ld, d, ud, x[i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_Solve3D(ld, d, ud, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= ld[i-2]*x[i-2, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1]
                x[n-2, j, k] = x[n-2, j, k]/d[n-2]

        for i in range(n - 3, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - ud[i]*x[i+2, j, k])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= ld[j-2]*x[i, j-2, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]
                x[i, n-2, k] = x[i, n-2, k]/d[n-2]

            for j in range(n - 3, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - ud[j]*x[i, j+2, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                TDMA_Solve1D(ld, d, ud, x[i, j])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_Solve1D(ld, d, ud, x, axis=0):
    n = d.shape[0]
    for i in range(1, n):
        x[i] -= ld[i-1]*x[i-1]
    x[n-1] = x[n-1]/d[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = (x[i] - ud[i]*x[i+1])/d[i]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_LU(data):
    ld = data[0, :-1]
    d = data[1, :]
    ud = data[2, 1:]
    n = d.shape[0]
    for i in range(1, n):
        ld[i-1] = ld[i-1]/d[i-1]
        d[i] -= ld[i-1]*ud[i-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_Solve2D(ld, d, ud, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(1, n):
            for j in range(x.shape[1]):
                x[i, j] -= ld[i-1]*x[i-1, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1]

        for i in range(n - 2, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - ud[i]*x[i+1, j])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            TDMA_O_Solve1D(ld, d, ud, x[i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_Solve3D(ld, d, ud, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(1, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= ld[i-1]*x[i-1, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1]

        for i in range(n - 2, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - ud[i]*x[i+1, j, k])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(1, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= ld[j-1]*x[i, j-1, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]

            for j in range(n - 2, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - ud[j]*x[i, j+1, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                TDMA_O_Solve1D(ld, d, ud, x[i, j])

def TDMA_O_Solve(x, data, axis=0):
    ld = data[0, :-1]
    d = data[1, :]
    ud = data[2, 1:]
    n = x.ndim
    if n == 1:
        TDMA_O_Solve1D(ld, d, ud, x)
    elif n == 2:
        TDMA_O_Solve2D(ld, d, ud, x, axis)
    elif n == 3:
        TDMA_O_Solve3D(ld, d, ud, x, axis)

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
def TDMA_O_inner_solve(u, lu):
    ld = lu[0, :-1]
    d = lu[1, :]
    ud = lu[2, 1:]
    n = d.shape[0]
    for i in range(1, n):
        u[i] -= ld[i-1]*u[i-1]

    u[n-1] = u[n-1]/d[n-1]
    for i in range(n-2, -1, -1):
        u[i] = (u[i] - ud[i]*u[i+1])/d[i]
