import numba as nb

__all__ = ['PDMA_LU', 'PDMA_Solve',
           'PDMA_inner_solve']

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

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_Solve1D(a, b, d, e, f, u):
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

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_LU2D(a, b, d, e, f, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            PDMA_LU(a[:-8, j], b[:-6, j], d[:-4, j], e[:-6, j], f[:-8, j])
    elif axis == 1:
        for i in range(d.shape[0]):
            PDMA_LU(a[i, :-8], b[i, :-6], d[i, :-4], e[i, :-6], f[i, :-8])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_LU3D(a, b, d, e, f, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                PDMA_LU(a[:-8, j, k], b[:-6, j, k], d[:-4, j, k], e[:-6, j, k], f[:-8, j, k])
    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                PDMA_LU(a[i, :-8, k], b[i, :-6, k], d[i, :-4, k], e[i, :-6, k], f[i, :-8, k])
    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                PDMA_LU(a[i, j, :-8], b[i, j, :-6], d[i, j, :-4], e[i, j, :-6], f[i, j, :-8])

def PDMA_Solve(x, data, axis=0):
    a = data[0, :-4]
    b = data[1, :-2]
    d = data[2, :]
    e = data[3, 2:]
    f = data[4, 4:]
    n = x.ndim
    if n == 1:
        PDMA_Solve1D(a, b, d, e, f, x)
    elif n == 2:
        PDMA_Solve2D(a, b, d, e, f, x, axis)
    elif n == 3:
        PDMA_Solve3D(a, b, d, e, f, x, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_Solve2D(a, b, d, e, f, x, axis):
    if axis == 0:
        for j in range(x.shape[1]):
            PDMA_Solve1D(a, b, d, e, f, x[:, j])
    elif axis == 1:
        for i in range(x.shape[0]):
            PDMA_Solve1D(a, b, d, e, f, x[i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_Solve3D(a, b, d, e, f, x, axis):
    if axis == 0:
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                PDMA_Solve1D(a, b, d, e, f, x[:, j, k])
    elif axis == 1:
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                PDMA_Solve1D(a, b, d, e, f, x[i, :, k])
    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                PDMA_Solve1D(a, b, d, e, f, x[i, j])

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
