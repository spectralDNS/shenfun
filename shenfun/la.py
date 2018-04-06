r"""
This module contains linear algebra solvers for SparseMatrixes
"""
import numpy as np
from scipy.linalg import decomp_cholesky
from shenfun.optimization import la
from shenfun.matrixbase import SparseMatrix

class TDMA(object):
    """Tridiagonal matrix solver

    Parameters
    ----------
        mat : SparseMatrix
              Symmetric tridiagonal matrix with diagonals in offsets -2, 0, 2

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, mat):
        assert isinstance(mat, SparseMatrix)
        self.mat = mat
        self.N = 0
        self.dd = np.zeros(0)
        self.ud = None
        self.L = None

    def init(self):
        """Initialize and allocate solver"""
        M = self.mat.shape[0]
        B = self.mat
        self.dd = B[0].copy()*np.ones(M)
        self.ud = B[2].copy()*np.ones(M-2)
        self.L = np.zeros(M-2)
        la.TDMA_SymLU(self.dd, self.ud, self.L)

    def __call__(self, b, u=None, axis=0):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        if not self.dd.shape[0] == self.mat.shape[0]:
            self.init()

        if len(u.shape) == 3:
            la.TDMA_SymSolve3D(self.dd, self.ud, self.L, u, axis)
            #la.TDMA_SymSolve3D_ptr(self.dd[self.s], self.ud[self.s], self.L,
                                   #u[self.s], axis)
        elif len(u.shape) == 2:
            la.TDMA_SymSolve2D(self.dd, self.ud, self.L, u, axis)

        elif len(u.shape) == 1:
            la.TDMA_SymSolve(self.dd, self.ud, self.L, u)

        else:
            raise NotImplementedError

        u /= self.mat.scale
        return u

class PDMA(object):
    """Pentadiagonal matrix solver

    Parameters
    ----------
        mat : SparseMatrix
              Symmetric pentadiagonal matrix with diagonals in offsets
              -4, -2, 0, 2, 4
        solver : str, optional
                 Choose implementation

                 - cython - Use efficient cython implementation
                 - python - Use python/scipy

    """

    def __init__(self, mat, solver="cython"):
        assert isinstance(mat, SparseMatrix)
        self.mat = mat
        self.solver = solver
        self.N = 0
        self.d0 = np.zeros(0)
        self.d1 = None
        self.d2 = None
        self.A = None
        self.L = None

    def init(self):
        """Initialize and allocate solver"""
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
            #self.A = np.zeros((9, B[0].shape[0]))
            #self.A[0, 4:] = self.d2
            #self.A[2, 2:] = self.d1
            #self.A[4, :] = self.d0
            #self.A[6, :-2] = self.d1
            #self.A[8, :-4] = self.d2
            self.A = np.zeros((5, B[0].shape[0]))
            self.A[0, 4:] = self.d2
            self.A[2, 2:] = self.d1
            self.A[4, :] = self.d0
            self.L = decomp_cholesky.cholesky_banded(self.A)

    @staticmethod
    def SymLU(d, e, f): # pragma: no cover
        """Symmetric LU decomposition"""
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

    @staticmethod
    def SymSolve(d, e, f, b): # pragma: no cover
        """Symmetric solve (for testing only)"""
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
        for k in range(n-5, -1, -1):
            bc[k] /= d[k]
            bc[k] -= (e[k]*bc[k+2] + f[k]*bc[k+4])
        b[:] = bc.astype(float)

    def __call__(self, b, u=None, axis=0):

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b

        if not self.d0.shape[0] == self.mat[0].shape[0]:
            self.init()

        if len(u.shape) == 3:
            #la.PDMA_Symsolve3D(self.d0, self.d1, self.d2, u, axis)
            la.PDMA_Symsolve3D_ptr(self.d0, self.d1, self.d2, u, axis)

        elif len(u.shape) == 2:
            la.PDMA_Symsolve2D(self.d0, self.d1, self.d2, u, axis)

        elif len(u.shape) == 1:
            la.PDMA_Symsolve(self.d0, self.d1, self.d2, u[:-4])

        else:
            raise NotImplementedError

        u /= self.mat.scale
        return u

class DiagonalMatrix(np.ndarray):
    """Matrix type with only diagonal matrices in all dimensions

    Typically used for Fourier spaces
    """
    # pylint: disable=too-few-public-methods, unused-argument

    def __new__(cls, buffer):
        assert isinstance(buffer, np.ndarray)
        obj = np.ndarray.__new__(cls,
                                 buffer.shape,
                                 dtype=buffer.dtype,
                                 buffer=buffer)

        return obj

    def solve(self, b, u=None, axis=0, neglect_zero_wavenumber=True):
        """Solve for diagonal matrix

        Parameters
        ----------
            b : array
                Right hand side on entry, solution on exit if no u
            u : array, optional
                Output array
            neglect_zero_wavenumber : bool, optional
                                      Whether or not to neglect zeros on
                                      diagonal
        """
        diagonal_array = self
        if neglect_zero_wavenumber:
            d = np.where(diagonal_array == 0, 1, diagonal_array)

        if u is not None:
            u[:] = b / d
            return u
        return b / d

    #def matvec(self, v, c):
        #c[:] = self*v
        #return c
