from shenfun.optimization import la
from shenfun.spectralbase import SpectralBase
import numpy as np

class TDMA(object):
    """Tridiagonal matrix solver

    args:
        basis       ShenNeumannBasis, ShenDirichletBasis or ShenLegendreDirichletBasis

    """

    def __init__(self, basis):
        assert isinstance(basis, SpectralBase)
        self.basis = basis
        self.dd = np.zeros(0)

    def init(self, N):
        M = self.basis.get_shape(N)
        B = self.basis.get_mass_matrix()(np.arange(N).astype(float), self.basis.quad)
        self.dd = B[0].copy()*np.ones(M)
        self.ud = B[2].copy()*np.ones(M-2)
        self.L = np.zeros(M-2)
        self.s = self.basis.slice(N)
        la.TDMA_SymLU(self.dd[self.s], self.ud[self.s], self.L)

    def __call__(self, u):
        N = u.shape[0]
        if not self.dd.shape[0] == u.shape[0]:
            self.init(N)
        if len(u.shape) == 3:
            #la.TDMA_3D(self.ud, self.dd, self.dd.copy(), self.ud.copy(), u[self.s])
            la.TDMA_SymSolve3D(self.dd[self.s], self.ud[self.s], self.L, u[self.s])
        elif len(u.shape) == 1:
            #la.TDMA_1D(self.ud, self.dd, self.dd.copy(), self.ud.copy(), u[self.s])
            la.TDMA_SymSolve(self.dd[self.s], self.ud[self.s], self.L, u[self.s])
        else:
            raise NotImplementedError
        return u

class PDMA(object):
    """Pentadiagonal matrix solver

    args:
        basis       (ShenBiharmonicBasis)

    kwargs:
        solver      ('cython', 'python')     Choose implementation

    """

    def __init__(self, basis, solver="cython"):
        self.basis = basis
        self.B = np.zeros(0)
        self.solver = solver

    def init(self, N):
        self.B = self.basis.get_mass_matrix()(np.arange(N), self.basis.quad)
        if self.solver == "cython":
            self.d0, self.d1, self.d2 = self.B[0].copy(), self.B[2].copy(), self.B[4].copy()
            la.PDMA_SymLU(self.d0, self.d1, self.d2)
            #self.SymLU(self.d0, self.d1, self.d2)
            ##self.d0 = self.d0.astype(float)
            ##self.d1 = self.d1.astype(float)
            ##self.d2 = self.d2.astype(float)
        else:
            #self.L = lu_factor(self.B.diags().toarray())
            self.d0, self.d1, self.d2 = self.B[0].copy(), self.B[2].copy(), self.B[4].copy()
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

    def __call__(self, u):
        N = u.shape[0]
        if not self.B.shape[0] == u.shape[0]:
            self.init(N)
        if len(u.shape) == 3:
            if self.solver == "cython":
                la.PDMA_Symsolve3D(self.d0, self.d1, self.d2, u[:-4])
            else:
                b = u.copy()
                for i in range(u.shape[1]):
                    for j in range(u.shape[2]):
                        #u[:-4, i, j] = lu_solve(self.L, b[:-4, i, j])
                        u[:-4, i, j] = la_solve.spsolve(self.B.diags(), b[:-4, i, j])

        elif len(u.shape) == 1:
            if self.solver == "cython":
                la.PDMA_Symsolve(self.d0, self.d1, self.d2, u[:-4])
                #self.SymSolve(self.d0, self.d1, self.d2, u[:-4])
            else:
                b = u.copy()
                #u[:-4] = lu_solve(self.L, b[:-4])
                #u[:-4] = la_solve.spsolve(self.B.diags(), b[:-4])
                #u[:-4] = solve_banded((4, 4), self.A, b[:-4])
                u[:-4] = decomp_cholesky.cho_solve_banded((self.L, False), b[:-4])
        else:
            raise NotImplementedError

        return u
