import numba as nb

__all__ = ['TDMA_SymLU', 'TDMA_SymSolve', 'TDMA_SymSolve2D',
           'TDMA_SymSolve3D', 'TDMA_SymLU_VC', 'TDMA_SymSolve_VC',
           'TDMA_O_SymLU', 'TDMA_O_SymSolve']

#@nb.jit((float[:], float[:], float[:]), cache=True, nopython=True, fastmath=True)
@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymLU(d, ud, ld):
    n = d.shape[0]
    for i in range(2, n):
        ld[i-2] = ud[i-2]/d[i-2]
        d[i] -= ld[i-2]*ud[i-2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_SymLU(d, ud, ld):
    n = d.shape[0]
    for i in range(1, n):
        ld[i-1] = ud[i-1]/d[i-1]
        d[i] = d[i] - ld[i-1]*ud[i-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymLU_2D(d, ud, ld, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            TDMA_SymLU(d[:, j], ud[:, j], ld[:, j])
    elif axis == 1:
        for i in range(d.shape[0]):
            TDMA_SymLU(d[i], ud[i], ld[i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymLU_3D(d, ud, ld, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                TDMA_SymLU(d[:, j, k], ud[:, j, k], ld[:, j, k])
    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                TDMA_SymLU(d[i, :, k], ud[i, :, k], ld[i, :, k])
    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                TDMA_SymLU(d[i, j], ud[i, j], ld[i, j])

def TDMA_SymLU_VC(d, a, l, axis=0):
    n = d.ndim
    if n == 1:
        TDMA_SymLU(d, a, l)
    elif n == 2:
        TDMA_SymLU_2D(d, a, l, axis)
    elif n == 3:
        TDMA_SymLU_3D(d, a, l, axis)

def TDMA_SymSolve(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        TDMA_SymSolve2D(d, a, l, x, axis)
    elif n == 3:
        TDMA_SymSolve3D(d, a, l, x, axis)

def TDMA_SymSolve_VC(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_SymSolve1D(d, a, l, x)
    elif n == 2:
        TDMA_SymSolve_VC_2D(d, a, l, x, axis)
    elif n == 3:
        TDMA_SymSolve_VC_3D(d, a, l, x, axis)

#@nb.jit((float[::1], float[::1], float[::1], float[::1]), nopython=True, fastmath=True)
@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymSolve1D(d, a, l, x):
    n = x.shape[0]-2
    for i in range(2, n):
        x[i] -= l[i-2]*x[i-2]

    x[n-1] = x[n-1]/d[n-1]
    x[n-2] = x[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - a[i]*x[i+2])/d[i]

#@nb.jit((float[:], float[:], float[:], complex[:, :], nb.int64), cache=True, nopython=True, fastmath=True)
@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymSolve2D(d, a, l, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                x[i, j] -= l[i-2]*x[i-2, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1]
            x[n-2, j] = x[n-2, j]/d[n-2]

        for i in range(n - 3, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - a[i]*x[i+2, j])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            TDMA_SymSolve1D(d, a, l, x[i])

#@nb.jit([(nb.float32[:], nb.float32[:], nb.float32[:], nb.complex64[:, :, :], nb.int64),
#         (nb.float64[:], nb.float64[:], nb.float64[:], nb.complex128[:, :, :], nb.int64)], cache=True, nopython=True, fastmath=True)
@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymSolve3D(d, a, l, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(2, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[i-2]*x[i-2, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1]
                x[n-2, j, k] = x[n-2, j, k]/d[n-2]

        for i in range(n - 3, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i]*x[i+2, j, k])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(2, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[j-2]*x[i, j-2, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]
                x[i, n-2, k] = x[i, n-2, k]/d[n-2]

            for j in range(n - 3, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[j]*x[i, j+2, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                TDMA_SymSolve1D(d, a, l, x[i, j])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymSolve_VC_3D(d, a, l, x, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                TDMA_SymSolve1D(d[:, j, k], a[:, j, k], l[:, j, k], x[:, j, k])
    elif axis == 1:
        for i in range(d.shape[0]):
            for k in range(d.shape[2]):
                TDMA_SymSolve1D(d[i, :, k], a[i, :, k], l[i, :, k], x[i, :, k])
    elif axis == 2:
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                TDMA_SymSolve1D(d[i, j], a[i, j], l[i, j], x[i, j])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_SymSolve_VC_2D(d, a, l, x, axis):
    if axis == 0:
        for j in range(d.shape[1]):
            TDMA_SymSolve1D(d[:, j], a[:, j], l[:, j], x[:, j])
    elif axis == 1:
        for i in range(d.shape[0]):
            TDMA_SymSolve1D(d[i], a[i], l[i], x[i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_SymSolve1D(d, a, l, x, axis=0):
    n = d.shape[0]
    for i in range(1, n):
        x[i] -= l[i-1]*x[i-1]
    x[n-1] = x[n-1]/d[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = (x[i] - a[i]*x[i+1])/d[i]

@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_SymSolve2D(d, a, l, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(1, n):
            for j in range(x.shape[1]):
                x[i, j] -= l[i-1]*x[i-1, j]

        for j in range(x.shape[1]):
            x[n-1, j] = x[n-1, j]/d[n-1]

        for i in range(n - 2, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                x[i, j] = (x[i, j] - a[i]*x[i+1, j])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            TDMA_O_SymSolve1D(d, a, l, x[i])

#@nb.jit([(nb.float32[:], nb.float32[:], nb.float32[:], nb.complex64[:, :, :], nb.int64),
#         (nb.float64[:], nb.float64[:], nb.float64[:], nb.complex128[:, :, :], nb.int64)], cache=True, nopython=True, fastmath=True)
@nb.jit(nopython=True, fastmath=True, cache=True)
def TDMA_O_SymSolve3D(d, a, l, x, axis):
    n = d.shape[0]
    if axis == 0:
        for i in range(1, n):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[i-1]*x[i-1, j, k]

        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[n-1, j, k] = x[n-1, j, k]/d[n-1]

        for i in range(n - 2, -1, -1):
            d1 = 1./d[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[i]*x[i+1, j, k])*d1

    elif axis == 1:
        for i in range(x.shape[0]):
            for j in range(1, n):
                for k in range(x.shape[2]):
                    x[i, j, k] -= l[j-1]*x[i, j-1, k]

            for k in range(x.shape[2]):
                x[i, n-1, k] = x[i, n-1, k]/d[n-1]

            for j in range(n - 2, -1, -1):
                for k in range(x.shape[2]):
                    x[i, j, k] = (x[i, j, k] - a[j]*x[i, j+1, k])/d[j]

    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                TDMA_O_SymSolve1D(d, a, l, x[i, j])

def TDMA_O_SymSolve(d, a, l, x, axis=0):
    n = x.ndim
    if n == 1:
        TDMA_O_SymSolve1D(d, a, l, x)
    elif n == 2:
        TDMA_O_SymSolve2D(d, a, l, x, axis)
    elif n == 3:
        TDMA_O_SymSolve3D(d, a, l, x, axis)
