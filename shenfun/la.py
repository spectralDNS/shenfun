r"""
This module contains linear algebra solvers for SparseMatrixes
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from shenfun.optimization import optimizer
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
        self.TDMA_SymLU(self.dd, self.ud, self.L)

    @staticmethod
    @optimizer
    def TDMA_SymLU(d, ud, ld):
        n = d.shape[0]
        for i in range(2, n):
            ld[i-2] = ud[i-2]/d[i-2]
            d[i] = d[i] - ld[i-2]*ud[i-2]

    @staticmethod
    @optimizer
    def TDMA_SymSolve(d, a, l, x, axis=0):
        assert x.ndim == 1, "Use optimized version for multidimensional solve"
        n = d.shape[0]
        for i in range(2, n):
            x[i] -= l[i-2]*x[i-2]

        x[n-1] = x[n-1]/d[n-1]
        x[n-2] = x[n-2]/d[n-2]
        for i in range(n - 3, -1, -1):
            x[i] = (x[i] - a[i]*x[i+2])/d[i]

    def __call__(self, b, u=None, axis=0):
        """Solve matrix problem self u = b

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array, optional
                Output array
            axis : int, optional
                   The axis over which to solve for if b and u are multidimensional

        If u is not provided, then b is overwritten with the solution and returned
        """

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        if not self.dd.shape[0] == self.mat.shape[0]:
            self.init()

        self.TDMA_SymSolve(self.dd, self.ud, self.L, u, axis=axis)

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
        self.d0, self.d1, self.d2 = B[0].copy(), B[2].copy(), B[4].copy()
        self.PDMA_SymLU(self.d0, self.d1, self.d2)

        #self.A = np.zeros((5, B[0].shape[0]))
        #self.A[0, 4:] = self.d2
        #self.A[2, 2:] = self.d1
        #self.A[4, :] = self.d0
        #self.L = decomp_cholesky.cholesky_banded(self.A)

    @staticmethod
    @optimizer
    def PDMA_SymLU(d, e, f): # pragma: no cover
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
    @optimizer
    def PDMA_SymSolve(d, e, f, b, axis=0): # pragma: no cover
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
        """Solve matrix problem self u = b

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array, optional
                Output array
            axis : int, optional
                   The axis over which to solve for if b and u are multidimensional

        If u is not provided, then b is overwritten with the solution and returned
        """

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b

        if not self.d0.shape[0] == self.mat[0].shape[0]:
            self.init()

        self.PDMA_SymSolve(self.d0, self.d1, self.d2, u, axis)

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

class Solve(object):
    """Solver class for matrix created by Dirichlet bases

    Possibly with inhomogeneous boundary values

    Parameters
    ----------
        A : SparseMatrix
        test : BasisFunction

   """
    def __init__(self, A, test):
        assert A.shape[0] == A.shape[1]
        assert isinstance(A, SparseMatrix)
        self.s = test.slice()
        self.A = A
        if hasattr(test, 'bc'):
            self.bc = test.bc

    def __call__(self, b, u=None, axis=0):
        """Solve matrix problem Au = b

        Parameters
        ----------
        b : array
            Array of right hand side on entry and solution on exit unless
            u is provided.
        u : array, optional
            Output array
        axis : int, optional
               The axis over which to solve for if b and u are multidimensional

        If u is not provided, then b is overwritten with the solution and returned

        """
        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        s = self.s
        assert self.A.shape[0] == b[s].shape[0]
        A = self.A.diags('csr')
        if b.ndim == 1:
            u[s] = spsolve(A, b[s])
        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))

            if b.dtype is np.dtype('complex'):
                u.real[s] = spsolve(A, br.real).reshape(u[s].shape)
                u.imag[s] = spsolve(A, br.imag).reshape(u[s].shape)
            else:
                u[s] = spsolve(A, br).reshape(u[s].shape)
        if hasattr(self, 'bc'):
            self.bc.apply_after(u, True)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)

        u /= self.A.scale
        return u

class NeumannSolve(object):
    """Solver class for matrix created by Neumann bases

    Assuming Neumann test- and trialfunction, where index k=0 is used only
    to fix the mean value.

    Parameters
    ----------
        A : SparseMatrix
        test : BasisFunction

    """
    def __init__(self, A, test):
        assert A.shape[0] == A.shape[1]
        assert isinstance(A, SparseMatrix)
        self.mean = test.mean
        self.s = test.slice()
        self.A = A

    def __call__(self, b, u=None, axis=0):
        """Solve matrix problem A u = b

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array, optional
                Output array
            axis : int, optional
                   The axis over which to solve for if b and u are multidimensional

        If u is not provided, then b is overwritten with the solution and returned
        """
        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        b[0] = self.mean
        s = self.s
        self.A[0][0] = 1
        A = self.A.diags('csr')
        if b.ndim == 1:
            u[s] = spsolve(A, b[s])
        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))

            if b.dtype is np.dtype('complex'):
                u.real[s] = spsolve(A, br.real).reshape(u[s].shape)
                u.imag[s] = spsolve(A, br.imag).reshape(u[s].shape)
            else:
                u[s] = spsolve(A, br).reshape(u[s].shape)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)
        u /= self.A.scale
        return u
