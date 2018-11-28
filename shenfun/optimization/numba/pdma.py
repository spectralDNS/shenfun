import numba as nb

__all__ = ['PDMA_SymLU', 'PDMA_SymLU_VC', 'PDMA_SymSolve', 'PDMA_SymLU2D',
           'PDMA_SymLU3D', 'PDMA_SymSolve_VC']

def PDMA_SymLU_VC(d, a, l, axis=0):
    n = d.ndim
    if n == 1:
        PDMA_SymLU(d, a, l)
    elif n == 2:
        PDMA_SymLU2D(d, a, l, axis)
    elif n == 3:
        PDMA_SymLU3D(d, a, l, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymLU(d, e, f):
    n = d.shape[0]
    m = e.shape[0]
    k = n - m

    for i in range(n-2*k):
        lam = e[i]/d[i]
        d[i+k] -= lam*e[i]
        e[i+k] -= lam*f[i]
        e[i] = lam
        lam = f[i]/d[i]
        d[i+2*k] -= lam*f[i]
        f[i] = lam

    lam = e[n-4]/d[n-4]
    d[n-2] -= lam*e[n-4]
    e[n-4] = lam
    lam = e[n-3]/d[n-3]
    d[n-1] -= lam*e[n-3]
    e[n-3] = lam

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymLU2D(d, e, f, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            PDMA_SymLU(d[:-4, j], e[:-6, j], f[:-8, j])
    elif axis == 1:
        for i in range(d.shape[0]):
            PDMA_SymLU(d[i, :-4], e[i, :-6], f[i, :-8])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymLU3D(d, e, f, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                PDMA_SymLU(d[:-4, j, k], e[:-6, j, k], f[:-8, j, k])
    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                PDMA_SymLU(d[i, :-4, k], e[i, :-6, k], f[i, :-8, k])
    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                PDMA_SymLU(d[i, j, :-4], e[i, j, :-6], f[i, j, :-8])

def PDMA_SymSolve(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        PDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        PDMA_SymSolve2D(d, a, l, x, axis)
    elif n == 3:
        PDMA_SymSolve3D(d, a, l, x, axis)

def PDMA_SymSolve_VC(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        PDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        PDMA_SymSolve2D_VC(d, a, l, x, axis)
    elif n == 3:
        PDMA_SymSolve3D_VC(d, a, l, x, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymSolve1D(d, e, f, b):
    n = d.shape[0]
    b[2] -= e[0]*b[0]
    b[3] -= e[1]*b[1]
    for k in range(4, n):
        b[k] -= (e[k-2]*b[k-2] + f[k-4]*b[k-4])

    b[n-1] /= d[n-1]
    b[n-2] /= d[n-2]
    b[n-3] /= d[n-3]
    b[n-3] -= e[n-3]*b[n-1]
    b[n-4] /= d[n-4]
    b[n-4] -= e[n-4]*b[n-2]
    for k in range(n-5, -1, -1):
        b[k] /= d[k]
        b[k] -= (e[k]*b[k+2] + f[k]*b[k+4])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymSolve2D(d, e, f, b, axis):
    if axis == 0:
        for j in range(b.shape[1]):
            PDMA_SymSolve1D(d, e, f, b[:, j])
    elif axis == 1:
        for i in range(b.shape[0]):
            PDMA_SymSolve1D(d, e, f, b[i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymSolve3D(d, e, f, b, axis):
    if axis == 0:
        for j in range(b.shape[1]):
            for k in range(b.shape[2]):
                PDMA_SymSolve1D(d, e, f, b[:, j, k])
    elif axis == 1:
        for i in range(b.shape[0]):
            for k in range(b.shape[2]):
                PDMA_SymSolve1D(d, e, f, b[i, :, k])
    elif axis == 2:
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                PDMA_SymSolve1D(d, e, f, b[i, j])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymSolve3D_VC(d, e, f, x, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                PDMA_SymSolve1D(d[:-4, j, k], e[:-6, j, k], f[:-8, j, k], x[:, j, k])
    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                PDMA_SymSolve1D(d[i, :-4, k], e[i, :-6, k], f[i, :-8, k], x[i, :, k])
    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                PDMA_SymSolve1D(d[i, j, :-4], e[i, j, :-6], f[i, j, :-8], x[i, j, :])

@nb.jit(nopython=True, fastmath=True, cache=True)
def PDMA_SymSolve2D_VC(d, e, f, x, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            PDMA_SymSolve1D(d[:-4, j], e[:-6, j], f[:-8, j], x[:, j])
    elif axis == 1:
        for i in range(d.shape[0]):
            PDMA_SymSolve1D(d[i, :-4], e[i, :-6], f[i, :-8], x[i, :])
