r"""
This module contains linear algebra solvers for SparseMatrices,
TPMatrices and BlockMatrices.
"""
import numpy as np
from numbers import Number, Integral
from scipy.sparse import spmatrix, kron
from scipy.sparse.linalg import spsolve, splu
from shenfun.config import config
from shenfun.optimization import optimizer
from shenfun.matrixbase import SparseMatrix, extract_bc_matrices, \
    SpectralMatrix, BlockMatrix, TPMatrix, get_simplified_tpmatrices
from shenfun.forms.arguments import Function
from mpi4py import MPI

comm = MPI.COMM_WORLD

def Solver(mats):
    """Return appropriate solver for `mats`

    Parameters
    ----------
    mats : SparseMatrix or list of SparseMatrices

    Returns
    -------
    Matrix solver

    Note
    ----
    The list of matrices may include boundary matrices. The returned solver
    will incorporate these boundary matrices automatically on the right hand
    side of the equation system.
    """
    assert isinstance(mats, (SparseMatrix, list))
    bc_mats = []
    mat = mats
    if isinstance(mats, list):
        bc_mats = extract_bc_matrices([mats])
        mat = sum(mats[1:], mats[0])
    return mat.get_solver()([mat]+bc_mats)

class SparseMatrixSolver:
    """SparseMatrix solver

    Parameters
    ----------
    mat : SparseMatrix or list of SparseMatrices

    Note
    ----
    The list of matrices may include boundary matrices. The returned solver
    will incorporate these boundary matrices automatically on the right hand
    side of the equation system.

    """
    def __init__(self, mat):
        assert isinstance(mat, (SparseMatrix, list))
        self.bc_mats = []
        if isinstance(mat, list):
            bc_mats = extract_bc_matrices([mat])
            mat = sum(mat[1:], mat[0])
            self.bc_mats = bc_mats
        self.mat = mat
        self._lu = None
        assert self.mat.shape[0] == self.mat.shape[1]

    def apply_bcs(self, b, axis=0):
        if len(self.bc_mats) > 0:
            assert isinstance(b, Function)
            b.set_boundary_dofs()
            w0 = np.zeros_like(b)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(b, w0, axis=axis)
        return b

    def apply_constraints(self, b, constraints, axis=0):
        """Apply constraints to matrix `self.mat` and rhs vector `b`

        Parameters
        ----------
        b : array
        constraints : tuple of 2-tuples
            The 2-tuples represent (row, val)
            The constraint indents the matrix row and sets b[row] = val
        axis : int
            The axis we are solving over
        """
        # Only apply constraint to matrix first time around

        if len(constraints) > 0:
            if b.ndim > 1:
                T = b.function_space().bases[axis]

            A = self.mat
            if isinstance(A, spmatrix):
                for (row, val) in constraints:
                    if self._lu is None:
                        A = A.tolil()
                        _, zerorow = A[row].nonzero()
                        A[(row, zerorow)] = 0
                        A[row, row] = 1
                        self.mat = A.tocsc()

                    if b.ndim > 1:
                        b[T.si[row]] = val
                    else:
                        b[row] = val

            elif isinstance(A, SparseMatrix):
                for (row, val) in constraints:
                    if self._lu is None:
                        for key, vals in A.items():
                            if key >= 0:
                                M = A.shape[0]-key
                                v = np.broadcast_to(np.atleast_1d(vals), M).copy()
                                if row < M:
                                    v[row] = int(key == 0)/A.scale
                            elif key < 0:
                                M = A.shape[0]+key
                                v = np.broadcast_to(np.atleast_1d(vals), M).copy()
                                if row+key < M and row+key > 0:
                                    v[row+key] = 0
                            A[key] = v
                    if b.ndim > 1:
                        b[T.si[row]] = val
                    else:
                        b[row] = val
        return b

    def perform_lu(self):
        """Perform LU-decomposition"""
        if self._lu is None:
            if isinstance(self.mat, SparseMatrix):
                self.mat = self.mat.diags('csc')
            self._lu = splu(self.mat, permc_spec=config['matrix']['sparse']['permc_spec'])
        return self._lu

    def solve(self, b, u, axis, lu):

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        s = slice(0, self.mat.shape[0])
        if b.ndim == 1:
            if b.dtype.char in 'fdg' or lu.U.dtype.char in 'FDG':
                u[s] = lu.solve(b[s])
            else: # complex b and real matrix
                u.real[s] = lu.solve(b[s].real)
                u.imag[s] = lu.solve(b[s].imag)

        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))
            if b.dtype.char in 'fdg' or lu.U.dtype.char in 'FDG':
                u[s] = lu.solve(br).reshape(u[s].shape)
            else: # complex b and real matrix
                u.real[s] = lu.solve(br.real).reshape(u[s].shape)
                u.imag[s] = lu.solve(br.imag).reshape(u[s].shape)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)

        return u

    def __call__(self, b, u=None, axis=0, constraints=()):
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
        constraints : tuple of 2-tuples
            The 2-tuples represent (row, val)
            The constraint indents the matrix row and sets b[row] = val

        Note
        ----
        If u is not provided, then b is overwritten with the solution and returned

        """
        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Boundary conditions
        b = self.apply_bcs(b, axis=axis)

        b = self.apply_constraints(b, constraints, axis=axis)

        lu = self.perform_lu()

        u = self.solve(b, u, axis, lu)

        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()

        return u


class TDMA(SparseMatrixSolver):
    """Tridiagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix or list of SparseMatrices
        Tridiagonal matrix with diagonals in offsets -2, 0, 2

    """
    def __init__(self, mat):
        SparseMatrixSolver.__init__(self, mat)
        N = self.mat.shape[0]

        A = self.mat
        #self.issymmetric = np.all(A[2] == A[-2])
        self.issymmetric = A.issymmetric
        self.dd = A[0]*np.ones(N)*A.scale
        self.ud = A[2]*np.ones(N-2)*A.scale
        self.ld = np.zeros(N-2) if self.issymmetric else A[-2]*np.ones(N-2)*A.scale

    @staticmethod
    @optimizer
    def TDMA_LU(ld, d, ud):
        n = d.shape[0]
        for i in range(2, n):
            ld[i-2] = ld[i-2]/d[i-2]
            d[i] = d[i] - ld[i-2]*ud[i-2]

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

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of TDMA'
            self.dd[0] = 1
            self.ud[0] = 0
            s = [slice(None)]*len(b.shape)
            s[axis] = 0
            b[tuple(s)] = constraints[0][1]
        return b

    def perform_lu(self):
        if self._lu is None:
            if self.issymmetric:
                self.TDMA_SymLU(self.dd, self.ud, self.ld)
            else:
                self.TDMA_LU(self.ld, self.dd, self.ud)
            self._lu = 1
        return 1

    def solve(self, b, u, axis, lu):
        if u is not b:
            u[:] = b
        self.TDMA_SymSolve(self.dd, self.ud, self.ld, u, axis=axis)
        return u

class TDMA_O(TDMA):
    """Tridiagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix
        Symmetric tridiagonal matrix with diagonals in offsets -1, 0, 1

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, mat):
        SparseMatrixSolver.__init__(self, mat)
        B = self.mat
        M = B.shape[0]
        self.dd = B[0]*np.ones(M)*B.scale
        self.ud = B[1]*np.ones(M-1)*B.scale
        self.L = np.zeros(M-1)

    def perform_lu(self):
        if self._lu is None:
            self.TDMA_O_SymLU(self.dd, self.ud, self.L)
            self._lu = 1
        return 1

    def solve(self, b, u, axis, lu):
        if u is not b:
            u[:] = b
        self.TDMA_O_SymSolve(self.dd, self.ud, self.L, u, axis=axis)
        return u

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


class PDMA(SparseMatrixSolver):
    """Pentadiagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix or list of SparseMatrices
        Pentadiagonal matrix with diagonals in offsets
        -4, -2, 0, 2, 4

    """

    def __init__(self, mat):
        SparseMatrixSolver.__init__(self, mat)
        assert len(self.mat) == 5

        # Broadcast in case diagonal is simply a constant.
        self.issymmetric = self.mat.issymmetric
        #assert self.issymmetric, 'Unsymmetric is broken'
        A = self.mat
        #self.issymmetric = np.all(A[2] == A[-2]) and np.all(A[4] == A[-4])
        shape = self.mat.shape[1]
        self.d0 = np.broadcast_to(np.atleast_1d(self.mat[0]), shape).copy()*self.mat.scale
        self.d1 = np.broadcast_to(np.atleast_1d(self.mat[2]), shape-2).copy()*self.mat.scale
        self.d2 = np.broadcast_to(np.atleast_1d(self.mat[4]), shape-4).copy()*self.mat.scale

        self.l1 = None
        self.l2 = None
        self.A = None
        self.L = None
        self._lu = None

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of PDMA'
            self.d0[0] = 1
            self.d1[0] = 0
            self.d2[0] = 0
            if b.ndim > 1:
                s = [slice(None)]*len(b.shape)
                s[axis] = 0
                b[tuple(s)] = constraints[0][1]
                #T = b.function_space().bases[axis]
                #b[T.si[0]] = constraints[0][1]
            else:
                b[0] = constraints[0][1]
        return b

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
    def PDMA_LU(a, b, d, e, f): # pragma: no cover
        """LU decomposition"""
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

    @staticmethod
    @optimizer
    def PDMA_SymSolve(d, e, f, u, axis=0):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)

        n = d.shape[0]
        bc = u

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
        u[:] = bc.astype(float)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)


    @staticmethod
    @optimizer
    def PDMA_Solve(a, b, d, e, f, u, axis=0): # pragma: no cover
        if axis > 0:
            u = np.moveaxis(u, axis, 0)

        n = d.shape[0]
        bc = u

        bc[2] -= b[0]*bc[0]
        bc[3] -= b[1]*bc[1]
        for k in range(4, n):
            bc[k] -= (b[k-2]*bc[k-2] + a[k-4]*bc[k-4])

        bc[n-1] /= d[n-1]
        bc[n-2] /= d[n-2]
        bc[n-3] = (bc[n-3]-e[n-3]*bc[n-1])/d[n-3]
        bc[n-4] = (bc[n-4]-e[n-4]*bc[n-2])/d[n-4]
        for k in range(n-5, -1, -1):
            bc[k] = (bc[k]-e[k]*bc[k+2]-f[k]*bc[k+4])/d[k]

        if axis > 0:
            u = np.moveaxis(u, 0, axis)

    def perform_lu(self):
        if self._lu is None:
            if self.issymmetric:
                self.PDMA_SymLU(self.d0, self.d1, self.d2)

            else:
                shape = self.mat.shape[1]
                self.l1 = np.broadcast_to(np.atleast_1d(self.mat[-2]), shape-2).copy()*self.mat.scale
                self.l2 = np.broadcast_to(np.atleast_1d(self.mat[-4]), shape-4).copy()*self.mat.scale
                self.PDMA_LU(self.l2, self.l1, self.d0, self.d1, self.d2)
            self._lu = 1
        return 1

    def solve(self, b, u, axis, lu):
        if u is not b:
            u[:] = b

        if self.issymmetric:
            self.PDMA_SymSolve(self.d0, self.d1, self.d2, u, axis)
        else:
            self.PDMA_Solve(self.l2, self.l1, self.d0, self.d1, self.d2, u, axis)
        return u

class FDMA(SparseMatrixSolver):
    """4-diagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix or list of SparseMatrices
        4-diagonal matrix with diagonals in offsets -2, 0, 2, 4

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, mat):
        SparseMatrixSolver.__init__(self, mat)
        N = self.mat.shape[0]
        self.dd = self.mat[0]*np.ones(N)*self.mat.scale
        self.u1 = self.mat[2]*np.ones(N-2)*self.mat.scale
        self.u2 = self.mat[4]*np.ones(N-4)*self.mat.scale
        self.ld = self.mat[-2]*np.ones(N-2)*self.mat.scale

    def perform_lu(self):
        if self._lu is None:
            self.FDMA_LU(self.ld, self.dd, self.u1, self.u2)
            self._lu = 1
        return 1

    @staticmethod
    @optimizer
    def FDMA_LU(ld, d, u1, u2):
        n = d.shape[0]
        for i in range(2, n):
            ld[i-2] = ld[i-2]/d[i-2]
            d[i] = d[i] - ld[i-2]*u1[i-2]
            if i < n-2:
                u1[i] = u1[i] - ld[i-2]*u2[i-2]

    @staticmethod
    @optimizer
    def FDMA_Solve(d, u1, u2, l, x, axis=0):
        assert x.ndim == 1, "Use optimized version for multidimensional solve"
        n = d.shape[0]
        for i in range(2, n):
            x[i] -= l[i-2]*x[i-2]

        x[n-1] = x[n-1]/d[n-1]
        x[n-2] = x[n-2]/d[n-2]
        x[n-3] = (x[n-3] - u1[n-3]*x[n-1])/d[n-3]
        x[n-4] = (x[n-4] - u1[n-4]*x[n-2])/d[n-4]
        for i in range(n - 5, -1, -1):
            x[i] = (x[i] - u1[i]*x[i+2] - u2[i]*x[i+4])/d[i]

    def solve(self, b, u, axis, lu):
        if u is not b:
            u[:] = b
        self.FDMA_Solve(self.dd, self.u1, self.u2, self.ld, u, axis=axis)
        return u

class TwoDMA(SparseMatrixSolver):
    """2-diagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix
        2-diagonal matrix with diagonals in offsets 0, 2

    """
    def __init__(self, mat):
        SparseMatrixSolver.__init__(self, mat)
        N = self.mat.shape[0]
        self.dd = self.mat[0]*np.ones(N)*self.mat.scale
        self.u1 = self.mat[2]*np.ones(N-2)*self.mat.scale

    @staticmethod
    @optimizer
    def TwoDMA_Solve(d, u1, x, axis=0):
        assert x.ndim == 1, "Use optimized version for multidimensional solve"
        n = d.shape[0]
        x[n-1] = x[n-1]/d[n-1]
        x[n-2] = x[n-2]/d[n-2]
        for i in range(n - 3, -1, -1):
            x[i] = (x[i] - u1[i]*x[i+2])/d[i]

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of TwoDMA'
            self.dd[0] = 1
            self.u1[0] = 0
            if b.ndim > 1:
                shape = b.shape
                b = b.flatten()
                b[0] = constraints[0][1]
                b = b.reshape(shape)
            else:
                b[0] = constraints[0][1]
        return b

    def perform_lu(self):
        return 1

    def solve(self, b, u, axis, lu):
        if u is not b:
            u[:] = b
        self.TwoDMA_Solve(self.dd, self.u1, u, axis=axis)
        return u


class Solve(SparseMatrixSolver):
    """Generic solver class for SparseMatrix

    Possibly with inhomogeneous boundary values

    Parameters
    ----------
        mat : SparseMatrix or list of SparseMatrices
        format : str, optional
            The format of the scipy.sparse.spmatrix to convert into
            before solving. Default is Compressed Sparse Column `csc`.

    Note
    ----
    This solver converts the matrix to a Scipy sparse matrix of choice and
    uses `scipy.sparse` methods `splu` and `spsolve`.

    """
    def __init__(self, mat, format=None):
        format = config['matrix']['sparse']['solve'] if format is None else format
        SparseMatrixSolver.__init__(self, mat)
        self.mat = self.mat.diags(format)


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
        tpmats = get_simplified_tpmatrices(tpmats)
        bc_mats = extract_bc_matrices([tpmats])
        self.tpmats = tpmats
        self.bc_mats = bc_mats
        self.T = tpmats[0].space
        self.M = None

    def matvec(self, u, c):
        c.fill(0)
        if u.ndim == 2:
            s0 = tuple(base.slice() for base in self.T)
            c[s0] = self.M.dot(u[s0].flatten()).reshape(self.T.dims())
        else:
            raise NotImplementedError
        return c

    def get_diagonal_axis(self):
        naxes = self.T.get_nondiagonal_axes()
        diagonal_axis = np.setxor1d([0, 1, 2], naxes)
        assert len(diagonal_axis) == 1
        return diagonal_axis[0]

    def diags(self, i):
        """Return matrix for given index `i` in diagonal direction"""
        if self.T.dimensions == 2:
            # In 2D there's just 1 matrix, store and reuse
            if self.M is not None:
                return self.M
            m = self.tpmats[0]
            M0 = m.diags('csc')
            for m in self.tpmats[1:]:
                M0 = M0 + m.diags('csc')
            self.M = M0
            return self.M

        else:
            # 1 matrix per Fourier coefficient
            naxes = self.T.get_nondiagonal_axes()
            m = self.tpmats[0]
            diagonal_axis = self.get_diagonal_axis()
            sc = [0, 0, 0]
            sc[diagonal_axis] = i if m.scale.shape[diagonal_axis] > 1 else 0
            A0 = m.mats[naxes[0]].diags('csc')
            A1 = m.mats[naxes[1]].diags('csc')
            M0 = kron(A0, A1, 'csc')
            M0 *= m.scale[tuple(sc)]
            for m in self.tpmats[1:]:
                A0 = m.mats[naxes[0]].diags('csc')
                A1 = m.mats[naxes[1]].diags('csc')
                M1 = kron(A0, A1, 'csc')
                sc[diagonal_axis] = i if m.scale.shape[diagonal_axis] > 1 else 0
                M1 *= m.scale[tuple(sc)]
                M0 = M0 + M1
            return M0

    def apply_constraints(self, A, b, constraints):
        """Apply constraints to matrix `A` and rhs vector `b`

        Parameters
        ----------
        A : Sparse matrix
        b : array
        constraints : tuple of 2-tuples
            The 2-tuples represent (row, val)
            The constraint indents the matrix row and sets b[row] = val
        """
        if len(constraints) > 0:
            A = A.tolil()
            for (row, val) in constraints:
                _, zerorow = A[row].nonzero()
                A[(row, zerorow)] = 0
                A[row, row] = 1
                b[row] = val
            A = A.tocsc()
        return A, b

    def __call__(self, b, u=None, constraints=()):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)

        if u.ndim == 2:
            s0 = self.T.slice()
            M = self.diags(0)
            bs = b[s0].flatten()
            M, bs = self.apply_constraints(M, bs, constraints)
            u[s0] = spsolve(M, bs).reshape(self.T.dims())

        elif u.ndim == 3:
            naxes = self.T.get_nondiagonal_axes()
            diagonal_axis = self.get_diagonal_axis()
            s0 = list(self.T.slice())
            for i in range(self.T.shape(True)[diagonal_axis]):
                M0 = self.diags(i)
                s0[diagonal_axis] = i
                bs = b[tuple(s0)].flatten()
                M0, bs = self.apply_constraints(M0, bs, constraints)
                shape = np.take(self.T.dims(), naxes)
                u[tuple(s0)] = spsolve(M0, bs).reshape(shape)
        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()
        return u

class SolverDiagonal:
    """Solver for purely diagonal matrices, like Fourier in Cartesian coordinates.

    Parameters
    ----------
    tpmats : sequence
        sequence of instances of :class:`.TPMatrix`
    """
    def __init__(self, tpmats):
        tpmats = get_simplified_tpmatrices(tpmats)
        assert len(tpmats) == 1
        self.mat = tpmats[0]

    def __call__(self, b, u=None, constraints=()):
        return self.mat.solve(b, u=u, constraints=constraints)

class Solver2D:
    """Generic solver for tensorproductspaces in 2D

    Parameters
    ----------
    mats : sequence
        sequence of instances of :class:`.TPMatrix`

    Note
    ----
    If there are boundary matrices in the list of mats, then
    these matrices are used to modify the right hand side before
    solving. If this is not the desired behaviour, then use
    :func:`.extract_bc_matrices` on tpmats before using this class.

    """

    def __init__(self, tpmats):
        bc_mats = extract_bc_matrices([tpmats])
        self.tpmats = tpmats
        self.bc_mats = bc_mats
        self._lu = None
        m = tpmats[0]
        self.T = T = m.space
        assert m._issimplified is False, "Cannot use simplified matrices with this solver"
        M0 = m.diags(format='csc')
        for m in tpmats[1:]:
            M0 = M0 + m.diags('csc')
        self.M = M0

    def matvec(self, u, c):
        c.fill(0)
        s0 = tuple(base.slice() for base in self.T)
        c[s0] = self.M.dot(u[s0].flatten()).reshape(self.T.dims())
        return c

    @staticmethod
    def apply_constraints(A, b, constraints):
        """Apply constraints to matrix `A` and rhs vector `b`

        Parameters
        ----------
        A : Sparse matrix
        b : array
        constraints : tuple of 2-tuples
            The 2-tuples represent (row, val)
            The constraint indents the matrix row and sets b[row] = val
        """
        if len(constraints) > 0:
            A = A.tolil()
            for (row, val) in constraints:
                _, zerorow = A[row].nonzero()
                A[(row, zerorow)] = 0
                A[row, row] = 1
                b[row] = val
            A = A.tocsc()
        return A, b

    def __call__(self, b, u=None, constraints=()):

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)

        s0 = tuple(base.slice() for base in self.T)
        assert b.dtype.char == u.dtype.char

        bs = b[s0].flatten()
        self.M, bs = self.apply_constraints(self.M, bs, constraints)
        if self._lu is None:
            self._lu = splu(self.M, permc_spec=config['matrix']['sparse']['permc_spec'])

        if b.dtype.char in 'fdg' or self.M.dtype.char in 'FDG':
            u[s0] = self._lu.solve(bs).reshape(self.T.dims())

        else:
            u.imag[s0] = self._lu.solve(bs.imag).reshape(self.T.dims())
            u.real[s0] = self._lu.solve(bs.real).reshape(self.T.dims())

        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()
        return u

class Solver3D(Solver2D):
    """Generic solver for tensorproductspaces in 3D

    Parameters
    ----------
    mats : sequence
        sequence of instances of :class:`.TPMatrix`

    Note
    ----
    If there are boundary matrices in the list of mats, then
    these matrices are used to modify the right hand side before
    solving. If this is not the desired behaviour, then use
    :func:`.extract_bc_matrices` on mats before using this class.

    """

    def __init__(self, tpmats):
        Solver2D.__init__(self, tpmats)

class SolverND(Solver2D):
    """Generic solver for tensorproductspaces in N dimensions

    Parameters
    ----------
    mats : sequence
        sequence of instances of :class:`.TPMatrix`

    Note
    ----
    If there are boundary matrices in the list of mats, then
    these matrices are used to modify the right hand side before
    solving. If this is not the desired behaviour, then use
    :func:`.extract_bc_matrices` on mats before using this class.

    """

    def __init__(self, tpmats):
        Solver2D.__init__(self, tpmats)


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
    up to two diagonal (Fourier) directions.
    Also note that if there are boundary matrices in the list of mats, then
    these matrices are used to modify the right hand side before
    solving. If this is not the desired behaviour, then use
    :func:`.extract_bc_matrices` on mats before using this class.

    """

    def __init__(self, mats):
        assert isinstance(mats, list)
        mats = get_simplified_tpmatrices(mats)
        assert len(mats[0].naxes) == 1
        self.naxes = mats[0].naxes.pop()
        bc_mats = extract_bc_matrices([mats])
        self.mats = mats
        self.bc_mats = bc_mats
        # For time-dependent solver, store all generated solvers and reuse
        self.assemble()

    def matvec(self, u, c):
        c.fill(0)
        w0 = np.zeros_like(u)
        for mat in self.mats:
            c += mat.matvec(u, w0)

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            for bc_mat in self.bc_mats:
                c += bc_mat.matvec(u, w0)
        return c

    def assemble(self):
        ndim = self.mats[0].dimensions
        shape = self.mats[0].space.shape(True)
        if ndim == 2:
            self.MM = []
            if self.naxes == 0:
                # non-diagonal in axis=0
                for i in range(shape[1]):
                    MM = None
                    for mat in self.mats:
                        sc = mat.scale[0, i] if mat.scale.shape[1] > 1 else mat.scale[0, 0]
                        if MM:
                            MM += mat.mats[0]*sc
                        else:
                            MM = mat.mats[0]*sc
                    self.MM.append(Solver(MM))

            else:
                # non-diagonal in axis=1
                for i in range(shape[0]):
                    MM = None
                    for mat in self.mats:
                        sc = mat.scale[i, 0] if mat.scale.shape[0] > 1 else mat.scale[0, 0]
                        if MM:
                            MM += mat.mats[1]*sc
                        else:
                            MM = mat.mats[1]*sc
                    self.MM.append(Solver(MM))

        elif ndim == 3:
            self.MM = []
            if self.naxes == 0:
                # non-diagonal in axis=0
                for i in range(shape[1]):
                    self.MM.append([])
                    for j in range(shape[2]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, shape)[0, i, j]
                            if MM:
                                MM += mat.mats[0]*sc
                            else:
                                MM = mat.mats[0]*sc
                        self.MM[-1].append(Solver(MM))

            elif self.naxes == 1:
                # non-diagonal in axis=1
                for i in range(shape[0]):
                    self.MM.append([])
                    for j in range(shape[2]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, shape)[i, 0, j]
                            if MM:
                                MM += mat.mats[1]*sc
                            else:
                                MM = mat.mats[1]*sc
                        self.MM[-1].append(Solver(MM))

            elif self.naxes == 2:
                # non-diagonal in axis=2
                for i in range(shape[0]):
                    self.MM.append([])
                    for j in range(shape[1]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, shape)[i, j, 0]
                            if MM:
                                MM += mat.mats[2]*sc
                            else:
                                MM = mat.mats[2]*sc
                        self.MM[-1].append(Solver(MM))

    def apply_constraints(self, b, constraints=()):
        """Apply constraints to solver

        Note
        ----
        The SolverGeneric1ND solver can only constrain the first dofs of
        the diagonal axes. For Fourier this is the zero dof with the
        constant basis function exp(0).
        """
        ndim = self.mats[0].dimensions
        space = self.mats[0].space
        z0 = space.local_slice()
        paxes = [i for i in range(ndim) if i != self.naxes]
        if ndim == 2:
            paxis = paxes[0]
            if z0[paxis].start != 0:
                return b
            s = space.bases[paxis].si[0]
            self.MM[0].apply_constraints(b[s], constraints)
        else:
            if z0[paxes[0]].start*z0[paxes[1]].start != 0:
                return b
            s = [0, 0, 0]
            s[self.naxes] = slice(None)
            self.MM[0][0].apply_constraints(b[tuple(s)], constraints)
        return b

    def __call__(self, b, u=None, constraints=()):
        """Solve problem with one non-diagonal direction

        Parameters
        ----------
        b : array, right hand side
        u : array, solution
        constraints : tuple of 2-tuples
            Each 2-tuple (row, value) is a constraint set for the non-periodic
            direction, for Fourier index 0 in 2D and (0, 0) in 3D

        """
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
        m = self.mats[0]

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)

        space = self.mats[0].space
        b = self.apply_constraints(b, constraints)

        if u.ndim == 2:
            if self.naxes == 0:
                for i in range(b.shape[1]):
                    u[:, i] = self.MM[i](b[:, i], u[:, i])

            else:
                for i in range(b.shape[0]):
                    u[i] = self.MM[i](b[i], u[i])

        elif u.ndim == 3:

            if self.naxes == 0:
                # non-diagonal in axis=0
                for i in range(b.shape[1]):
                    for j in range(b.shape[2]):
                        sol = self.MM[i][j]
                        u[:, i, j] = sol(b[:, i, j], u[:, i, j])

            elif self.naxes == 1:
                # non-diagonal in axis=1
                for i in range(b.shape[0]):
                    for j in range(b.shape[2]):
                        sol = self.MM[i][j]
                        u[i, :, j] = sol(b[i, :, j], u[i, :, j])

            elif self.naxes == 2:
                # non-diagonal in axis=2
                for i in range(b.shape[0]):
                    for j in range(b.shape[1]):
                        sol = self.MM[i][j]
                        u[i, j] = sol(b[i, j], u[i, j])

        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()
        return u

class BlockMatrixSolver:
    def __init__(self, mats):
        assert isinstance(mats, (BlockMatrix, list))
        self.bc_mat = None
        self._lu = None
        if isinstance(mats, BlockMatrix):
            mats = mats.get_mats()
        bc_mats = extract_bc_matrices([mats])
        assert len(mats) > 0
        self.mat = BlockMatrix(mats)
        if len(bc_mats) > 0:
            self.bc_mat = BlockMatrix(bc_mats)
        tps = self.mat.testbase.flatten()
        offset = [np.zeros(tps[0].dimensions, dtype=int)]
        for i, tp in enumerate(tps):
            dims = tp.dims()
            offset.append(np.array(dims + offset[i]))
        self.offset = offset
        self.global_shape = self.offset[-1]

    @staticmethod
    def apply_constraint(A, b, offset, i, constraint):
        if constraint is None or comm.Get_rank() > 0:
            return A, b

        if isinstance(i, int):
            if i > 0:
                return A, b

        if isinstance(i, tuple):
            if np.sum(np.array(i)) > 0:
                return A, b

        row = offset + constraint[1]
        assert isinstance(constraint, tuple)
        assert len(constraint) == 3
        val = constraint[2]
        b[row] = val
        if A is not None:
            A = A.tolil()
            r = A.getrow(row).nonzero()
            A[(row, r[1])] = 0
            A[row, row] = 1
            A = A.tocsc()
        return A, b

    def __call__(self, b, u=None, constraints=()):
        from .forms.arguments import Function
        import scipy.sparse as sp
        space = b.function_space()
        if u is None:
            u = Function(space)
        else:
            assert u.shape == b.shape

        if self.bc_mat: # Add contribution to right hand side due to inhomogeneous boundary conditions
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            b -= self.bc_mat.matvec(u, w0)

        B = self.mat
        tpmat = B.get_mats(True)
        axis = tpmat.naxes[0] if isinstance(tpmat, TPMatrix) else 0
        tp = space.flatten() if hasattr(space, 'flatten') else [space]
        nvars = b.shape[0] if len(b.shape) > space.dimensions else 1
        u = u.expand_dims(0) if nvars == 1 else u
        b = b.expand_dims(0) if nvars == 1 else b
        for con in constraints:
            assert len(con) == 3
            assert isinstance(con[0], Integral)
            assert isinstance(con[1], Integral)
            assert isinstance(con[2], Number)
        N = self.global_shape[axis]
        gi = np.zeros(N, dtype=b.dtype)
        go = np.zeros(N, dtype=b.dtype)
        if space.dimensions == 1:
            s = [0, 0]
            if self._lu is None:
                Ai = B.diags((0,), format='csr').tocsc()
                lu = self._lu = sp.linalg.splu(Ai, permc_spec=config['matrix']['sparse']['permc_spec'])
            else:
                lu = self._lu
            for k in range(nvars):
                s[0] = k
                s[1] = tp[k].slice()
                gi[self.offset[k][axis]:self.offset[k+1][axis]] = b[tuple(s)]
            go[:] = lu.solve(gi)
            for k in range(nvars):
                s[0] = k
                s[1] = tp[k].slice()
                u[tuple(s)] = go[self.offset[k][axis]:self.offset[k+1][axis]]

        elif space.dimensions == 2:
            if len(tpmat.naxes) == 2: # 2 non-periodic axes
                s = [0, 0, 0]
                gi = np.zeros(space.dim(), dtype=b.dtype)
                go = np.zeros(space.dim(), dtype=b.dtype)
                start = 0
                for k in range(nvars):
                    s[0] = k
                    s[1] = tp[k].bases[0].slice()
                    s[2] = tp[k].bases[1].slice()
                    gi[start:(start+tp[k].dim())] = b[tuple(s)].ravel()
                    start += tp[k].dim()

                if self._lu is None:
                    Ai = B.diags(format='csr').tocsc()
                    for con in constraints:
                        dim = 0
                        for i in range(con[0]):
                            dim += tp[i].dim()
                        Ai, gi = self.apply_constraint(Ai, gi, dim, 0, con)
                    self._lu = sp.linalg.splu(Ai, permc_spec=config['matrix']['sparse']['permc_spec'])
                else:
                    for con in constraints:
                        dim = 0
                        for i in range(con[0]):
                            dim += tp[i].dim()
                            gi[dim] = con[2]

                lu = self._lu
                go[:] = lu.solve(gi)
                start = 0
                for k in range(nvars):
                    s[0] = k
                    s[1] = tp[k].bases[0].slice()
                    s[2] = tp[k].bases[1].slice()
                    u[tuple(s)] = go[start:(start+tp[k].dim())].reshape((1, tp[k].bases[0].dim(), tp[k].bases[1].dim()))
                    start += tp[k].dim()

            else:
                if self._lu is None:
                    self._lu = {}
                s = [0]*3
                # ii is periodic axis
                ii = {0:2, 1:1}[axis]
                d0 = [0, 0]
                for i in range(b.shape[ii]):
                    s[ii] = i # periodic
                    for k in range(nvars):
                        s[0] = k # vector index
                        s[axis+1] = tp[k].bases[axis].slice() # non-periodic
                        gi[self.offset[k][axis]:self.offset[k+1][axis]] = b[tuple(s)]
                    d0 = list(d0)
                    d0[(axis+1)%2] = i
                    d0 = tuple(d0)
                    if d0 in self._lu:
                        lu = self._lu[d0]
                        for con in constraints:
                            _, gi = self.apply_constraint(None, gi, self.offset[con[0]][axis], i, con)
                    else:
                        Ai = B.diags(d0, format='csr').tocsc()
                        for con in constraints:
                            Ai, gi = self.apply_constraint(Ai, gi, self.offset[con[0]][axis], i, con)
                        lu = sp.linalg.splu(Ai, permc_spec=config['matrix']['sparse']['permc_spec'])
                        self._lu[d0] = lu
                    if gi.dtype.char in 'fdg' or lu.U.dtype.char in 'FDG':
                        go[:] = lu.solve(gi)
                    else:
                        go.imag[:] = lu.solve(gi.imag)
                        go.real[:] = lu.solve(gi.real)

                    for k in range(nvars):
                        s[0] = k
                        s[axis+1] = tp[k].bases[axis].slice()
                        u[tuple(s)] = go[self.offset[k][axis]:self.offset[k+1][axis]]

        elif space.dimensions == 3:
            if self._lu is None:
                self._lu = {}
            s = [0]*4

            if len(tpmat.naxes) == 1:
                ii, jj = {0:(2, 3), 1:(1, 3), 2:(1, 2)}[axis]
                d0 = [0, 0, 0]
                for i in range(b.shape[ii]):
                    for j in range(b.shape[jj]):
                        s[ii], s[jj] = i, j
                        for k in range(nvars):
                            s[0] = k
                            s[axis+1] = tp[k].bases[axis].slice()
                            gi[self.offset[k][axis]:self.offset[k+1][axis]] = b[tuple(s)]
                        d0 = list(d0)
                        d0[ii-1], d0[jj-1] = i, j
                        d0 = tuple(d0)
                        if d0 in self._lu:
                            lu = self._lu[d0]
                            for con in constraints:
                                _, gi = self.apply_constraint(None, gi, self.offset[con[0]][axis], (i, j), con)
                        else:
                            Ai = B.diags(d0, format='csr').tocsc() # Note - bug in scipy (https://github.com/scipy/scipy/issues/14551) such that we cannot use format='csc' directly
                            for con in constraints:
                                Ai, gi = self.apply_constraint(Ai, gi, self.offset[con[0]][axis], (i, j), con)
                            lu = sp.linalg.splu(Ai, permc_spec=config['matrix']['sparse']['permc_spec'])
                            self._lu[d0] = lu

                        go[:] = lu.solve(gi)
                        for k in range(nvars):
                            s[0] = k
                            s[axis+1] = tp[k].bases[axis].slice()
                            u[tuple(s)] = go[self.offset[k][axis]:self.offset[k+1][axis]]

            elif len(tpmat.naxes) == 2:
                ii = np.setxor1d(tpmat.naxes, range(3))[0]+1 # periodic axis
                d0 = [0, 0, 0]
                N = []
                for tpi in tp:
                    N.append(np.prod(np.take(tpi.dims(), tpmat.naxes)))
                gi = np.zeros(np.sum(N), dtype=b.dtype)
                go = np.zeros(np.sum(N), dtype=b.dtype)
                for i in range(b.shape[ii]):
                    s = [0, 0, 0, 0]
                    start = 0
                    s[ii] = i
                    for k in range(nvars):
                        s[0] = k
                        for n in tpmat.naxes:
                            s[n+1] = tp[k].bases[n].slice()
                        gi[start:(start+N[k])] = b[tuple(s)].ravel()
                        start += N[k]

                    if i not in self._lu:
                        d0 = list(d0)
                        d0[ii-1] = i
                        d0 = tuple(d0)
                        Ai = B.diags(d0, format='csr').tocsc()
                        for con in constraints:
                            dim = 0
                            for n in range(con[0]):
                                dim += N[n]
                            Ai, gi = self.apply_constraint(Ai, gi, dim, i, con)
                        self._lu[i] = sp.linalg.splu(Ai, permc_spec=config['matrix']['sparse']['permc_spec'])
                        lu = self._lu[i]
                    else:
                        lu = self._lu[i]
                        for con in constraints:
                            dim = 0
                            for n in range(con[0]):
                                dim += N[n]
                            gi[dim] = con[2]

                    go[:] = lu.solve(gi)
                    start = 0
                    for k in range(nvars):
                        s[0] = k
                        for n in tpmat.naxes:
                            s[n+1] = tp[k].bases[n].slice()
                        u[tuple(s)] = go[start:(start+N[k])].reshape(u[tuple(s)].shape)
                        start += N[k]

        u = u.reshape(u.shape[1:]) if nvars == 1 else u
        b = b.reshape(b.shape[1:]) if nvars == 1 else b
        return u

