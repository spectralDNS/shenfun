r"""
This module contains linear algebra solvers for SparseMatrixes
"""
import numpy as np
import scipy.sparse as scp
from scipy.sparse.linalg import spsolve, splu
from shenfun.optimization import optimizer
from shenfun.matrixbase import SparseMatrix

class TDMA:
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
        self.dd = B[0]*np.ones(M)
        self.ud = B[2]*np.ones(M-2)
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

        Note
        ----
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

class PDMA:
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

        Note
        ----
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

class Solve:
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
        self.test = test

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

        Note
        ----
        If u is not provided, then b is overwritten with the solution and returned

        """
        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if u is not b:
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
        #if self.test.has_nonhomogeneous_bcs:
        #    self.test.bc.set_boundary_dofs(u, True)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)

        return u

class NeumannSolve:
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
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        b[0] = self.mean
        s = self.s

        A = self.A.diags('csr')
        _, zerorow = A[0].nonzero()
        A[(0, zerorow)] = 0
        A[0, 0] = 1

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
            if u is not b:
                b = np.moveaxis(b, 0, axis)
        #u /= self.A.scale
        return u

class SolverGeneric2ND:
    """Generic solver for problems consisting of tensorproduct matrices
    containing two non-diagonal submatrices.

    Parameters
    ----------
    mats : sequence
        sequence of instances of :class:`.TPMatrix`

    Note
    ----
    In addition to two non-diagonal matrices, the solver can also handle one
    additional diagonal matrix (one Fourier matrix).
    """

    def __init__(self, tpmats):
        self.tpmats = tpmats
        self.T = T = tpmats[0].space
        self.M = None

    def matvec(self, u, c):
        c.fill(0)
        if u.ndim == 2:
            s0 = tuple(base.slice() for base in self.T)
            c[s0] = self.M.dot(u[s0].flatten()).reshape(self.T.dims())
        return c

    def get_diagonal_axis(self):
        naxes = self.T.get_nondiagonal_axes()
        diagonal_axis = np.setxor1d([0, 1, 2], naxes)
        assert len(diagonal_axis) == 1
        return diagonal_axis[0]

    def diags(self, i, format='csr'):
        """Return matrix for given index `i` in diagonal direction"""
        if self.T.dimensions == 2:
            # In 2D there's just 1 matrix, store and reuse
            if self.M is not None:
                return self.M
            m = self.tpmats[0]
            M0 = scp.kron(m.mats[0].diags(format), m.mats[1].diags(format))
            M0 *= np.atleast_1d(m.scale).item()
            for m in self.tpmats[1:]:
                M1 = scp.kron(m.mats[0].diags(format), m.mats[1].diags(format))
                M1 *= np.atleast_1d(m.scale).item()
                M0 = M0 + M1
            self.M = M0
            return self.M

        else:
            # 1 matrix per Fourier coefficient
            naxes = self.T.get_nondiagonal_axes()
            m = self.tpmats[0]
            diagonal_axis = self.get_diagonal_axis()
            sc = [0, 0, 0]
            sc[diagonal_axis] = i if m.scale.shape[diagonal_axis] > 1 else 0
            M0 = scp.kron(m.mats[naxes[0]].diags(format), m.mats[naxes[1]].diags(format))
            M0 *= m.scale[tuple(sc)]
            for m in self.tpmats[1:]:
                M1 = scp.kron(m.mats[naxes[0]].diags(format), m.mats[naxes[1]].diags(format))
                sc[diagonal_axis] = i if m.scale.shape[diagonal_axis] > 1 else 0
                M1 *= m.scale[tuple(sc)]
                M0 = M0 + M1
            return M0

    def __call__(self, b, u=None, format='csr'):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
        if u.ndim == 2:
            s0 = self.T.slice()
            M = self.diags(0, format=format)
            u[s0] = scp.linalg.spsolve(M, b[s0].flatten()).reshape(self.T.dims())

        elif u.ndim == 3:
            naxes = self.T.get_nondiagonal_axes()
            diagonal_axis = self.get_diagonal_axis()
            s0 = list(self.T.slice())
            for i in range(self.T.shape(True)[diagonal_axis]):
                m = self.tpmats[0]
                M0 = self.diags(i, format=format)
                s0[diagonal_axis] = i
                shape = np.take(self.T.dims(), naxes)
                u[tuple(s0)] = scp.linalg.spsolve(M0, b[tuple(s0)].flatten()).reshape(shape)
        return u

class Solver2D:
    """Generic solver for tensorproductspaces in 2D

    Parameters
    ----------
    mats : sequence
        sequence of instances of :class:`.TPMatrix`

    """

    def __init__(self, tpmats):
        self.tpmats = tpmats
        m = tpmats[0]
        self.T = T = m.space
        ndim = T.dimensions
        assert ndim == 2
        assert np.atleast_1d(m.scale).size == 1, "Use level = 2 with :func:`.inner`"
        M0 = scp.kron(m.mats[0].diags(), m.mats[1].diags())
        M0 *= np.atleast_1d(m.scale).item()
        for m in tpmats[1:]:
            M1 = scp.kron(m.mats[0].diags(), m.mats[1].diags())
            assert np.atleast_1d(m.scale).size == 1, "Use level = 2 with :func:`.inner`"
            M1 *= np.atleast_1d(m.scale).item()
            M0 = M0 + M1
        self.M = M0

    def matvec(self, u, c):
        c.fill(0)
        s0 = tuple(base.slice() for base in self.T)
        c[s0] = self.M.dot(u[s0].flatten()).reshape(self.T.dims())
        return c

    def __call__(self, b, u=None):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
        s0 = tuple(base.slice() for base in self.T)
        u[s0] = scp.linalg.spsolve(self.M, b[s0].flatten()).reshape(self.T.dims())
        return u


class TDMA_O:
    """Tridiagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix
        Symmetric tridiagonal matrix with diagonals in offsets -1, 0, 1

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
        self.dd = B[0]*np.ones(M)
        self.ud = B[1]*np.ones(M-1)
        self.L = np.zeros(M-1)
        self.TDMA_O_SymLU(self.dd, self.ud, self.L)

    @staticmethod
    @optimizer
    def TDMA_O_SymLU(d, ud, ld):
        n = d.shape[0]
        for i in range(1, n):
            ld[i-1] = ud[i-1]/d[i-1]
            d[i] = d[i] - ld[i-1]*ud[i-1]

    @staticmethod
    @optimizer
    def TDMA_O_SymSolve(d, a, l, x, axis=0):
        assert x.ndim == 1, "Use optimized version for multidimensional solve"
        n = d.shape[0]
        for i in range(1, n):
            x[i] -= l[i-1]*x[i-1]

        x[n-1] = x[n-1]/d[n-1]
        for i in range(n-2, -1, -1):
            x[i] = (x[i] - a[i]*x[i+1])/d[i]

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

        Note
        ----
        If u is not provided, then b is overwritten with the solution and returned

        """

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        if not self.dd.shape[0] == self.mat.shape[0]:
            self.init()

        self.TDMA_O_SymSolve(self.dd, self.ud, self.L, u, axis=axis)

        u /= self.mat.scale
        return u


class SolverGeneric1ND:
    """Generic solver for tensorproduct matrices consisting of
    non-diagonal matrices along only one axis.

    Parameters
    ----------
    mats : sequence
        sequence of instances of :class:`.TPMatrix`

    Note
    ----
    In addition to the one non-diagonal direction, the solver can also handle
    up to two diagonal (Fourier) directions. Also, this Python version of the
    solver is not very efficient. Consider implementing in Cython.

    """

    def __init__(self, mats):
        assert isinstance(mats, list)
        m = mats[0]
        if m.naxes == []:
            for tpmat in mats:
                tpmat.simplify_diagonal_matrices()

        self.mats = mats

        # For time-dependent solver, store all generated matrices and reuse
        # This takes a lot of memory, so for now it's only implemented for 2D
        self.MM = None

    def __call__(self, b, u=None):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
        m = self.mats[0]

        if u.ndim == 2:
            if m.naxes[0] == 0:
                # non-diagonal in axis=0
                if self.MM is None:
                    self.MM = []
                    for i in range(b.shape[1]):
                        MM = None
                        for mat in self.mats:
                            sc = mat.scale[0, i] if mat.scale.shape[1] > 1 else mat.scale[0, 0]
                            if MM:
                                MM += sc*mat.mats[0]
                            else:
                                MM = sc*mat.mats[0]
                        sl = m.space.bases[0].slice()
                        MM._lu = splu(MM.diags('csc'))
                        u[sl, i] = MM.solve(b[sl, i], u[sl, i], use_lu=True)
                        self.MM.append(MM)

                else:
                    for i in range(b.shape[1]):
                        sl = m.space.bases[0].slice()
                        u[sl, i] = self.MM[i].solve(b[sl, i], u[sl, i], use_lu=True)

            else:
                if self.MM is None:
                    # non-diagonal in axis=1
                    self.MM = []
                    for i in range(b.shape[0]):
                        MM = None
                        for mat in self.mats:
                            sc = mat.scale[i, 0] if mat.scale.shape[0] > 1 else mat.scale[0, 0]
                            if MM:
                                MM += sc*mat.mats[1]
                            else:
                                MM = sc*mat.mats[1]
                        sl = m.space.bases[1].slice()
                        u[i, sl] = MM.solve(b[i, sl], u[i, sl])
                        MM._lu = splu(MM.diags('csc'))
                        MM.solve(b[i, sl], u[i, sl], use_lu=True)
                        self.MM.append(MM)

                else:
                    for i in range(b.shape[0]):
                        sl = m.space.bases[1].slice()
                        u[i, sl] = self.MM[i].solve(b[i, sl], u[i, sl], use_lu=True)


        elif u.ndim == 3:
            if m.naxes[0] == 0:
                # non-diagonal in axis=0
                for i in range(b.shape[1]):
                    for j in range(b.shape[2]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, u.shape)[0, i, j]
                            if MM:
                                MM += sc*mat.mats[0]
                            else:
                                MM = sc*mat.mats[0]
                        sl = mat.space.bases[0].slice()
                        u[sl, i, j] = MM.solve(b[sl, i, j], u[sl, i, j])

            elif m.naxes[0] == 1:
                # non-diagonal in axis=1
                for i in range(b.shape[0]):
                    for j in range(b.shape[2]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, u.shape)[i, 0, j]
                            if MM:
                                MM += sc*mat.mats[1]
                            else:
                                MM = sc*mat.mats[1]
                        sl = mat.space.bases[1].slice()
                        u[i, sl, j] = MM.solve(b[i, sl, j], u[i, sl, j])

            elif m.naxes[0] == 2:
                # non-diagonal in axis=2
                for i in range(b.shape[0]):
                    for j in range(b.shape[1]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, u.shape)[i, j, 0]
                            if MM:
                                MM += sc*mat.mats[2]
                            else:
                                MM = sc*mat.mats[2]
                        sl = mat.space.bases[2].slice()
                        u[i, j, sl] = MM.solve(b[i, j, sl], u[i, j, sl])

        return u
