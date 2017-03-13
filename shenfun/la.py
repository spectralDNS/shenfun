from shenfun.optimization import la
from shenfun.matrixbase import ShenMatrix
import numpy as np

class TDMA(object):
    """Tridiagonal matrix solver

    args:
        mat    Symmetric tridiagonal matrix with diagonals in offsets -2, 0, 2

    """

    def __init__(self, mat):
        assert isinstance(mat, ShenMatrix)
        self.mat = mat
        self.N = 0
        self.dd = np.zeros(0)

    def init(self, N):
        self.N = N
        M = self.mat.shape[0]
        B = self.mat
        self.dd = B[0].copy()*np.ones(M)
        self.ud = B[2].copy()*np.ones(M-2)
        self.L = np.zeros(M-2)
        self.s = self.mat.testfunction[0].slice()
        la.TDMA_SymLU(self.dd[self.s], self.ud[self.s], self.L)

    def __call__(self, b, u=None, axis=0):

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b

        N = u.shape[axis]
        if not N == self.N:
            self.init(N)

        if len(u.shape) == 3:
            la.TDMA_SymSolve3D(self.dd[self.s], self.ud[self.s], self.L,
                               u[self.s], axis)
        elif len(u.shape) == 1:
            la.TDMA_SymSolve(self.dd[self.s], self.ud[self.s], self.L,
                             u[self.s])

        else:
            raise NotImplementedError

        return u

class PDMA(object):
    """Pentadiagonal matrix solver

    args:
        mat       Symmetric pentadiagonal matrix with diagonals in offsets
                  -4, -2, 0, 2, 4

    kwargs:
        solver      ('cython', 'python')     Choose implementation

    """

    def __init__(self, mat, solver="cython"):
        assert isinstance(mat, ShenMatrix)
        self.mat = mat
        self.solver = solver
        self.N = 0

    def init(self, N):
        self.N = N
        B = self.mat
        if self.solver == "cython":
            self.d0, self.d1, self.d2 = B[0].copy(), B[2].copy(), B[4].copy()
            la.PDMA_SymLU(self.d0, self.d1, self.d2)
            #self.SymLU(self.d0, self.d1, self.d2)
            ##self.d0 = self.d0.astype(float)
            ##self.d1 = self.d1.astype(float)
            ##self.d2 = self.d2.astype(float)
        else:
            #self.L = lu_factor(B.diags().toarray())
            self.d0, self.d1, self.d2 = B[0].copy(), B[2].copy(), B[4].copy()
            #self.A = np.zeros((9, N-4))
            #self.A[0, 4:] = self.d2
            #self.A[2, 2:] = self.d1
            #self.A[4, :] = self.d0
            #self.A[6, :-2] = self.d1
            #self.A[8, :-4] = self.d2
            self.A = np.zeros((5, N-4))
            self.A[0, 4:] = self.d2
            self.A[2, 2:] = self.d1
            self.A[4, :] = self.d0
            self.L = decomp_cholesky.cholesky_banded(self.A)

    def SymLU(self, d, e, f):
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

    def SymSolve(self, d, e, f, b):
        n = d.shape[0]
        #bc = array(map(decimal.Decimal, b))
        bc = b

        bc[2] -= e[0]*bc[0]
        bc[3] -= e[1]*bc[1]
        for k in range(4, n):
            bc[k] -= (e[k-2]*bc[k-2] + f[k-4]*bc[k-4])

        bc[n-1] /= d[n-1]
        bc[n-2] /= d[n-2]
        bc[n-3] /= d[n-3]
        bc[n-3] -= e[n-3]*bc[n-1]
        bc[n-4] /= d[n-4]
        bc[n-4] -= e[n-4]*bc[n-2]
        for k in range(n-5,-1,-1):
            bc[k] /= d[k]
            bc[k] -= (e[k]*bc[k+2] + f[k]*bc[k+4])
        b[:] = bc.astype(float)

    def __call__(self, b, u=None, axis=0):

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b

        N = u.shape[0]
        if not N == self.N:
            self.init(N)
        if len(u.shape) == 3:
            la.PDMA_Symsolve3D(self.d0, self.d1, self.d2, u[:-4], axis)

        elif len(u.shape) == 1:
            la.PDMA_Symsolve(self.d0, self.d1, self.d2, u[:-4])

        else:
            raise NotImplementedError

        return u

class Helmholtz(object):
    """Helmholtz solver -u'' + alfa*u = b

    args:

    """

    def __init__(self, A, B, T=None, scale_A=True):
        self.B = B
        self.A = A
        B[2] = np.broadcast_to(B[2], A[2].shape)
        B[-2] = np.broadcast_to(B[-2], A[2].shape)
        A.scale = np.broadcast_to(A.scale, B.scale.shape)
        if scale_A:
            A *= -1.

        A_scale = A.scale
        B_scale = B.scale
        N = A.testfunction[0].N
        if T is None:
            self.u0 = np.zeros(N-2, float)     # Diagonal entries of U
            self.u1 = np.zeros(N-4, float)     # Diagonal+1 entries of U
            self.u2 = np.zeros(N-6, float)     # Diagonal+2 entries of U
            self.L  = np.zeros(N-4, float)     # The single nonzero row of L
            la.LU_Helmholtz_1D(A, B, A_scale, B_scale, self.u0, self.u1, self.u2, self.L)

        else:
            axis = A.axis
            assert A_scale.shape[axis] == 1
            assert B_scale.shape[axis] == 1

            shape = list(T.local_shape())
            shape[axis] = N-2
            self.u0 = np.zeros(shape, float)     # Diagonal entries of U
            shape[axis] = N-4
            self.u1 = np.zeros(shape, float)     # Diagonal+2 entries of U
            self.L  = np.zeros(shape, float)     # The single nonzero row of L
            shape[axis] = N-6
            self.u2 = np.zeros(shape, float)     # Diagonal+4 entries of U
            if len(T) == 2:
                la.LU_Helmholtz_2D(A, B, axis, A_scale, B_scale, self.u0, self.u1, self.u2, self.L)

            elif len(T) == 3:
                la.LU_Helmholtz_3D(A, B, axis, A_scale, B_scale, self.u0, self.u1, self.u2, self.L)

    def __call__(self, u, b):
        if len(u.shape) == 3:
            la.Solve_Helmholtz_3D(self.A.axis, b, u, self.u0, self.u1, self.u2, self.L)

        elif len(u.shape) == 2:
            la.Solve_Helmholtz_2D(self.A.axis, b, u, self.u0, self.u1, self.u2, self.L)

        else:
            la.Solve_Helmholtz_1D(b, u, self.u0, self.u1, self.u2, self.L)

        return u

    #def matvec(self, v, c):
        #assert self.neumann is False
        #c[:] = 0
        #if len(v.shape) > 1:
            #Matvec.Helmholtz_matvec3D(v, c, 1.0, self.alfa**2, self.A[0], self.A[2], self.B[0])
        #else:
            #Matvec.Helmholtz_matvec(v, c, 1.0, self.alfa**2, self.A[0], self.A[2], self.B[0])
        #return c
