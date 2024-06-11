r"""
This module contains linear algebra solvers for SparseMatrices,
TPMatrices and BlockMatrices.
"""
from numbers import Number, Integral
import numpy as np
from scipy.sparse import spmatrix, kron
from scipy.sparse.linalg import splu
from shenfun.config import config
from shenfun.optimization import optimizer, runtimeoptimizer
from shenfun.matrixbase import SparseMatrix, extract_bc_matrices, \
    BlockMatrix, get_simplified_tpmatrices
from shenfun.forms.arguments import Function
from mpi4py import MPI
comm = MPI.COMM_WORLD

def Solver(mats):
    """Return appropriate solver for `mats`

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances

    Returns
    -------
    Matrix solver (:class:`.SparseMatrixSolver`)

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
    """Solver for :class:`.SparseMatrix` matrices.

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances

    Note
    ----
    The list of matrices may include boundary matrices. The returned solver
    will incorporate these boundary matrices automatically on the right hand
    side of the equation system.

    """
    def __init__(self, mats):
        assert isinstance(mats, (SparseMatrix, list))
        self.bc_mats = []
        if isinstance(mats, list):
            bc_mats = extract_bc_matrices([mats])
            mats = sum(mats[1:], mats[0])
            self.bc_mats = bc_mats
        self.mat = mats
        self._lu = None
        self._inner_arg = None # argument to inner_solve
        self.dtype = None
        assert self.mat.shape[0] == self.mat.shape[1]

    def apply_bcs(self, b, u, axis=0):
        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(b)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0, axis=axis)
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
        if self._lu is None:
            if isinstance(self.mat, SparseMatrix):
                self.mat = self.mat.diags('csc')
            self.mat.eliminate_zeros()
            if self.mat.nnz < 2:
                self._lu = None
                return self._lu
            self._lu = splu(self.mat, permc_spec=config['matrix']['sparse']['permc_spec'])
            self.dtype = self.mat.dtype.char
            self._inner_arg = (self._lu, self.dtype)
        return self._lu

    def solve(self, b, u, axis, lu):
        """Solve Au=b

        Solve along axis if b and u are multidimensional arrays.

        Parameters
        ----------
        b, u : arrays of rhs and output
            Both can be multidimensional
        axis : int
            The axis we are solving over
        lu : LU-decomposition
            Can be either the output from splu, or a dia-matrix containing
            the L and U matrices. The latter is used in subclasses.

        """
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        s = slice(0, self.mat.shape[0])
        if b.ndim == 1:
            if b.dtype.char in 'fdg' or self.dtype in 'FDG':
                u[s] = lu.solve(b[s])

            else:
                u.real[s] = lu.solve(b[s].real)
                u.imag[s] = lu.solve(b[s].imag)

        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))
            if b.dtype.char in 'fdg' or self.dtype in 'FDG':
                u[s] = lu.solve(br).reshape(u[s].shape)
            else:
                u.real[s] = lu.solve(br.real).reshape(u[s].shape)
                u.imag[s] = lu.solve(br.imag).reshape(u[s].shape)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)

        return u

    @staticmethod
    def inner_solve(u, lu):
        """Solve Au=b for one-dimensional u

        On entry u is the rhs b, on exit it contains the solution.

        Parameters
        ----------
        u : array 1D
            rhs on entry and solution on exit
        lu : LU-decomposition
            Can be either a 2-tuple with (output from splu, dtype), or a scipy
            dia-matrix containing the L and U matrices. The latter is used in
            subclasses.

        """
        lu, dtype = lu
        s = slice(0, lu.shape[0])
        if u.dtype.char in 'fdg' or dtype in 'FDG':
            u[s] = lu.solve(u[s])

        else:
            u.real[s] = lu.solve(u.real[s])
            u.imag[s] = lu.solve(u.imag[s])

    def __call__(self, b, u=None, axis=0, constraints=()):
        """Solve matrix problem Au = b along axis

        This routine also applies boundary conditions and constraints,
        and performes LU-decomposition on the fully assembled matrix.

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
        b = self.apply_bcs(b, u, axis=axis)
        b = self.apply_constraints(b, constraints, axis=axis)
        lu = self.perform_lu() # LU must be performed after constraints, because constraints modify the matrix
        if not lu is None:
            u = self.solve(b, u, axis, lu)
        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()
        return u

class BandedMatrixSolver(SparseMatrixSolver):
    def __init__(self, mats):
        SparseMatrixSolver.__init__(self, mats)
        self._lu = self.mat.diags('dia')

    def solve(self, b, u, axis, lu):
        if u is not b:
            sl = u.function_space().slice() if hasattr(u, 'function_space') else slice(None)
            u[sl] = b[sl]
        self.Solve(u, lu.data, axis)
        return u

    @staticmethod
    def LU(data):
        """LU-decomposition using either Cython or Numba

        Parameters
        ----------
        data : 2D-array
            Storage for dia-matrix on entry and L and U matrices
            on exit.
        """
        raise NotImplementedError

    @staticmethod
    def Solve(u, data, axis):
        """Fast solve using either Cython or Numba

        Parameters
        ----------
        u : array
            rhs on entry, solution on exit
        data : 2D-array
            Storage for dia-matrix containing L and U matrices
        axis : int
            The axis we are solving over
        """
        raise NotImplementedError


class DiagMA(BandedMatrixSolver):
    """Diagonal matrix solver

    Parameters
    ----------
    mats : Diagonal :class:`.SparseMatrix` or list of diagonal :class:`.SparseMatrix` instances

    """
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)
        self.issymmetric = True
        self._inner_arg = self._lu.data

    def perform_lu(self):
        return self._lu

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row'
            self._lu.diagonal(0)[0] = 1
            s = [slice(None)]*len(b.shape)
            s[axis] = 0
            b[tuple(s)] = constraints[0][1]
        return b

    @staticmethod
    def inner_solve(u, lu):
        d = lu[0]
        u[:d.shape[0]] /= d

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        DiagMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)

class TDMA(BandedMatrixSolver):
    """Tridiagonal matrix solver

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        Tridiagonal matrix with diagonals in offsets -2, 0, 2

    """
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)
        self.issymmetric = self.mat.issymmetric

    @staticmethod
    @runtimeoptimizer
    def LU(data):
        ld = data[0, :-2]
        d = data[1, :]
        ud = data[2, 2:]
        n = d.shape[0]
        for i in range(2, n):
            ld[i-2] = ld[i-2]/d[i-2]
            d[i] = d[i] - ld[i-2]*ud[i-2]

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of TDMA'
            self._lu.diagonal(0)[0] = 1
            self._lu.diagonal(2)[0] = 0
            s = [slice(None)]*len(b.shape)
            s[axis] = 0
            b[tuple(s)] = constraints[0][1]
        return b

    def perform_lu(self):
        if self._inner_arg is None:
            self.LU(self._lu.data)
            self._inner_arg = self._lu.data
        return self._lu

    @staticmethod
    def inner_solve(u, data):
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

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        TDMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)


class TDMA_O(BandedMatrixSolver):
    """Tridiagonal matrix solver

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        Symmetric tridiagonal matrix with diagonals in offsets -1, 0, 1

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)

    def perform_lu(self):
        if self._inner_arg is None:
            self.LU(self._lu.data)
            self._inner_arg = self._lu.data
        return self._lu

    @staticmethod
    @runtimeoptimizer
    def LU(data):
        ld = data[0, :-1]
        d = data[1, :]
        ud = data[2, 1:]
        n = d.shape[0]
        for i in range(1, n):
            ld[i-1] = ld[i-1]/d[i-1]
            d[i] -= ld[i-1]*ud[i-1]

    @staticmethod
    def inner_solve(u, data):
        ld = data[0, :-1]
        d = data[1, :]
        ud = data[2, 1:]
        n = d.shape[0]
        for i in range(1, n):
            u[i] -= ld[i-1]*u[i-1]

        u[n-1] = u[n-1]/d[n-1]
        for i in range(n-2, -1, -1):
            u[i] = (u[i] - ud[i]*u[i+1])/d[i]

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        TDMA_O.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)


class PDMA(BandedMatrixSolver):
    """Pentadiagonal matrix solver

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        Pentadiagonal matrix with diagonals in offsets
        -4, -2, 0, 2, 4

    """
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)
        assert len(self.mat) == 5

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of PDMA'
            self._lu.diagonal(0)[0] = 1
            self._lu.diagonal(2)[0] = 0
            self._lu.diagonal(4)[0] = 0
            if b.ndim > 1:
                s = [slice(None)]*len(b.shape)
                s[axis] = 0
                b[tuple(s)] = constraints[0][1]
            else:
                b[0] = constraints[0][1]
            self._inner_arg = self._lu.data
        return b

    @staticmethod
    @runtimeoptimizer
    def LU(data): # pragma: no cover
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

    def perform_lu(self):
        if self._inner_arg is None:
            self.LU(self._lu.data)
            self._inner_arg = self._lu.data
        return self._lu

    @staticmethod
    def inner_solve(u, data):
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

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        PDMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)

class FDMA(BandedMatrixSolver):
    """4-diagonal matrix solver

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        4-diagonal matrix with diagonals in offsets -2, 0, 2, 4

    """
    # pylint: disable=too-few-public-methods
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)

    def perform_lu(self):
        if self._inner_arg is None:
            self.LU(self._lu.data)
            self._inner_arg = self._lu.data
        return self._lu

    @staticmethod
    @runtimeoptimizer
    def LU(data):
        ld = data[0, :-2]
        d = data[1, :]
        u1 = data[2, 2:]
        u2 = data[3, 4:]
        n = d.shape[0]
        for i in range(2, n):
            ld[i-2] = ld[i-2]/d[i-2]
            d[i] = d[i] - ld[i-2]*u1[i-2]
            if i < n-2:
                u1[i] = u1[i] - ld[i-2]*u2[i-2]

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of FDMA'
            self._lu.diagonal(0)[0] = 1
            self._lu.diagonal(2)[0] = 0
            self._lu.diagonal(4)[0] = 0
            s = [slice(None)]*len(b.shape)
            s[axis] = 0
            b[tuple(s)] = constraints[0][1]
        return b

    @staticmethod
    def inner_solve(u, data):
        ld = data[0, :-2]
        d = data[1, :]
        u1 = data[2, 2:]
        u2 = data[3, 4:]
        n = d.shape[0]
        for i in range(2, n):
            u[i] -= ld[i-2]*u[i-2]
        u[n-1] = u[n-1]/d[n-1]
        u[n-2] = u[n-2]/d[n-2]
        u[n-3] = (u[n-3] - u1[n-3]*u[n-1])/d[n-3]
        u[n-4] = (u[n-4] - u1[n-4]*u[n-2])/d[n-4]
        for i in range(n - 5, -1, -1):
            u[i] = (u[i] - u1[i]*u[i+2] - u2[i]*u[i+4])/d[i]

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        FDMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)

class TwoDMA(BandedMatrixSolver):
    """2-diagonal matrix solver

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        2-diagonal matrix with diagonals in offsets 0, 2

    """
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)
        self._inner_arg = self._lu.data

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of TwoDMA'
            self._lu.diagonal(0)[0] = 1
            self._lu.diagonal(2)[0] = 0
            s = [slice(None)]*len(b.shape)
            s[axis] = 0
            b[tuple(s)] = constraints[0][1]
        return b

    def perform_lu(self):
        return self._lu

    @staticmethod
    def inner_solve(u, data):
        d = data[0, :]
        u1 = data[1, 2:]
        n = d.shape[0]
        u[n-1] = u[n-1]/d[n-1]
        u[n-2] = u[n-2]/d[n-2]
        for i in range(n - 3, -1, -1):
            u[i] = (u[i] - u1[i]*u[i+2])/d[i]

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        TwoDMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)

class ThreeDMA(BandedMatrixSolver):
    """3-diagonal matrix solver - all diagonals upper

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        3-diagonal matrix with diagonals in offsets 0, 2, 4

    """
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)
        self._inner_arg = self._lu.data

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of TwoDMA'
            self._lu.diagonal(0)[0] = 1
            self._lu.diagonal(2)[0] = 0
            self._lu.diagonal(4)[0] = 0
            s = [slice(None)]*len(b.shape)
            s[axis] = 0
            b[tuple(s)] = constraints[0][1]
        return b

    def perform_lu(self):
        return self._lu

    @staticmethod
    def inner_solve(u, data):
        d = data[0, :]
        u1 = data[1, 2:]
        u2 = data[1, 4:]
        n = d.shape[0]
        u[n-1] = u[n-1]/d[n-1]
        u[n-2] = u[n-2]/d[n-2]
        u[n-3] = (u[n-3]-u1[n-3]*u[n-1])/d[n-3]
        u[n-4] = (u[n-4]-u1[n-4]*u[n-2])/d[n-4]
        for i in range(n - 5, -1, -1):
            u[i] = (u[i] - u1[i]*u[i+2] - u2[i]*u[i+4])/d[i]

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        ThreeDMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)

class HeptaDMA(BandedMatrixSolver):
    """Heptadiagonal matrix solver

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
        Heptadiagonal matrix with diagonals in offsets
        -4, -2, 0, 2, 4, 6, 8

    """
    def __init__(self, mats):
        BandedMatrixSolver.__init__(self, mats)
        assert len(self.mat) == 7

    def apply_constraints(self, b, constraints, axis=0):
        if len(constraints) > 0:
            assert len(constraints) == 1
            assert constraints[0][0] == 0, 'Can only fix first row of HeptaDMA'
            self._lu.diagonal(0)[0] = 1
            self._lu.diagonal(2)[0] = 0
            self._lu.diagonal(4)[0] = 0
            self._lu.diagonal(6)[0] = 0
            self._lu.diagonal(8)[0] = 0
            if b.ndim > 1:
                s = [slice(None)]*len(b.shape)
                s[axis] = 0
                b[tuple(s)] = constraints[0][1]
            else:
                b[0] = constraints[0][1]
            self._inner_arg = self._lu.data
        return b

    @staticmethod
    @runtimeoptimizer
    def LU(data): # pragma: no cover
        a = data[0, :-4]
        b = data[1, :-2]
        d = data[2, :]
        e = data[3, 2:]
        f = data[4, 4:]
        g = data[5, 6:]
        h = data[6, 8:]
        n = d.shape[0]
        m = e.shape[0]
        k = n - m
        for i in range(n-2*k):
            lam = b[i]/d[i]
            d[i+k] -= lam*e[i]
            e[i+k] -= lam*f[i]
            if i < n-6:
                f[i+k] -= lam*g[i]
            if i < n-8:
                g[i+k] -= lam*h[i]
            b[i] = lam
            lam = a[i]/d[i]
            b[i+k] -= lam*e[i]
            d[i+2*k] -= lam*f[i]
            if i < n-6:
                e[i+2*k] -= lam*g[i]
            if i < n-8:
                f[i+2*k] -= lam*h[i]
            a[i] = lam
        i = n-4
        lam = b[i]/d[i]
        d[i+k] -= lam*e[i]
        b[i] = lam
        i = n-3
        lam = b[i]/d[i]
        d[i+k] -= lam*e[i]
        b[i] = lam

    def perform_lu(self):
        if self._inner_arg is None:
            self.LU(self._lu.data)
            self._inner_arg = self._lu.data
        return self._lu

    @staticmethod
    def inner_solve(u, data):
        a = data[0, :-4]
        b = data[1, :-2]
        d = data[2, :]
        e = data[3, 2:]
        f = data[4, 4:]
        g = data[5, 6:]
        h = data[6, 8:]
        n = d.shape[0]
        u[2] -= b[0]*u[0]
        u[3] -= b[1]*u[1]
        for k in range(4, n):
            u[k] -= (b[k-2]*u[k-2] + a[k-4]*u[k-4])
        u[n-1] /= d[n-1]
        u[n-2] /= d[n-2]
        u[n-3] = (u[n-3]-e[n-3]*u[n-1])/d[n-3]
        u[n-4] = (u[n-4]-e[n-4]*u[n-2])/d[n-4]
        u[n-5] = (u[n-5]-e[n-5]*u[n-3]-f[n-5]*u[n-1])/d[n-5]
        u[n-6] = (u[n-6]-e[n-6]*u[n-4]-f[n-6]*u[n-2])/d[n-6]
        u[n-7] = (u[n-7]-e[n-7]*u[n-5]-f[n-7]*u[n-3]-g[n-7]*u[n-1])/d[n-7]
        u[n-8] = (u[n-8]-e[n-8]*u[n-6]-f[n-8]*u[n-4]-g[n-8]*u[n-2])/d[n-8]
        for k in range(n-9, -1, -1):
            u[k] = (u[k]-e[k]*u[k+2]-f[k]*u[k+4]-g[k]*u[k+6]-h[k]*u[k+8])/d[k]

    @staticmethod
    @runtimeoptimizer
    def Solve(u, data, axis):
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
        HeptaDMA.inner_solve(u, data)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)


class Solve(SparseMatrixSolver):
    """Generic solver class for ::class:`.SparseMatrix`

    Possibly with inhomogeneous boundary values

    Parameters
    ----------
    mats : :class:`.SparseMatrix` or list of :class:`.SparseMatrix` instances
    format : str, optional
        The format of the scipy.sparse.spmatrix to convert into
        before solving. Default is Compressed Sparse Column `csc`.

    Note
    ----
    This solver converts the matrix to a Scipy sparse matrix of choice and
    uses `scipy.sparse` methods `splu` and `spsolve`.

    """
    def __init__(self, mats, format=None):
        format = config['matrix']['sparse']['solve'] if format is None else format
        SparseMatrixSolver.__init__(self, mats)
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
        self.mats2D = {}
        self._lu = None

    def matvec(self, u, c):
        c.fill(0)
        if u.ndim == 2:
            s0 = tuple(base.slice() for base in self.T)
            c[s0] = self.mats2D.dot(u[s0].flatten()).reshape(self.T.dims())
        else:
            raise NotImplementedError
        return c

    def get_diagonal_axis(self):
        naxes = self.T.get_nondiagonal_axes()
        diagonal_axis = np.setxor1d([0, 1, 2], naxes)
        assert len(diagonal_axis) == 1
        return diagonal_axis[0]

    def diags(self, i):
        """Return matrix for given index `i` in diagonal direction

        Parameters
        ----------
        i : int
            Fourier wavenumber
        """
        if i in self.mats2D:
            return self.mats2D[i]

        if self.T.dimensions == 2:
            # In 2D there's just 1 matrix, store and reuse
            m = self.tpmats[0]
            M0 = m.diags('csc')
            for m in self.tpmats[1:]:
                M0 = M0 + m.diags('csc')

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
        self.mats2D[i] = M0
        return M0

    def apply_constraints(self, b, constraints):
        if len(constraints) > 0:
            if self._lu is None:
                A = self.mats2D[0]
                A = A.tolil()
                for (row, val) in constraints:
                    _, zerorow = A[row].nonzero()
                    A[(row, zerorow)] = 0
                    A[row, row] = 1
                    b[row] = val
                self.mats2D[0] = A.tocsc()
            else:
                for (row, val) in constraints:
                    b[row] = val
        return b

    def assemble(self):
        if len(self.mats2D) == 0:
            ndim = self.tpmats[0].dimensions
            if ndim == 2:
                mat = self.diags(0)
                self.mats2D[0] = mat

            elif ndim == 3:
                diagonal_axis = self.get_diagonal_axis()
                for i in range(self.T.shape(True)[diagonal_axis]):
                    M0 = self.diags(i)
                    self.mats2D[i] = M0
        return self.mats2D

    def perform_lu(self):
        if self._lu is not None:
            return self._lu
        ndim = self.tpmats[0].dimensions
        self._lu = {}
        if ndim == 2:
            self._lu[0] = splu(self.mats2D[0], permc_spec=config['matrix']['sparse']['permc_spec'])
        else:
            diagonal_axis = self.get_diagonal_axis()
            for i in range(self.T.shape(True)[diagonal_axis]):
                self._lu[i] = splu(self.mats2D[i], permc_spec=config['matrix']['sparse']['permc_spec'])
        return self._lu

    def __call__(self, b, u=None, constraints=()):
        """Solve generic problem

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

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = Function(self.T).v
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)

        mats = self.assemble()

        b = self.apply_constraints(b, constraints)

        lu = self.perform_lu()

        if u.ndim == 2:
            s0 = self.T.slice()
            bs = b[s0].flatten()
            if b.dtype.char in 'fdg' or self.mats2D[0].dtype.char in 'FDG':
                u[s0] = lu[0].solve(bs).reshape(self.T.dims())
            else:
                u.real[s0] = lu[0].solve(bs.real).reshape(self.T.dims())
                u.imag[s0] = lu[0].solve(bs.imag).reshape(self.T.dims())

        elif u.ndim == 3:
            naxes = self.T.get_nondiagonal_axes()
            diagonal_axis = self.get_diagonal_axis()
            s0 = list(self.T.slice())
            for i in range(self.T.shape(True)[diagonal_axis]):
                s0[diagonal_axis] = i
                bs = b[tuple(s0)].flatten()
                shape = np.take(self.T.dims(), naxes)
                if b.dtype.char in 'fdg' or self.mats2D[0].dtype.char in 'FDG':
                    u[tuple(s0)] = lu[i].solve(bs).reshape(shape)
                else:
                    u.real[tuple(s0)] = lu[i].solve(bs.real).reshape(shape)
                    u.imag[tuple(s0)] = lu[i].solve(bs.imag).reshape(shape)

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
        """Solve problem with :class:`.TPMatrix` consisting only of diagonal
        matrices

        Parameters
        ----------
        b : array, right hand side
        u : array, solution
        constraints : tuple of 2-tuples
            Each 2-tuple (row, value) is a constraint set for the non-periodic
            direction, for Fourier index 0 in 2D and (0, 0) in 3D

        """
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
        self.T = m.space
        assert m._issimplified is False, "Cannot use simplified matrices with this solver"
        mat = m.diags(format='csc')
        for m in tpmats[1:]:
            mat = mat + m.diags('csc')
        self.mat = mat

    def matvec(self, u, c):
        c.fill(0)
        s0 = tuple(base.slice() for base in self.T)
        c[s0] = self.mat.dot(u[s0].flatten()).reshape(self.T.dims())
        return c

    @staticmethod
    def apply_constraints(A, b, constraints):
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
        """Solve generic problem for sum of :class:`TPMatrix` instances

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
            pass
            #assert u.shape == b.shape

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = Function(self.T).v
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)

        s0 = tuple(base.slice() for base in self.T)
        assert b.dtype.char == u.dtype.char

        bs = b[s0].flatten()
        self.mat, bs = self.apply_constraints(self.mat, bs, constraints)
        if self._lu is None:
            self._lu = splu(self.mat, permc_spec=config['matrix']['sparse']['permc_spec'])

        if b.dtype.char in 'fdg' or self.mat.dtype.char in 'FDG':
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
    non-diagonal matrices along only one axis and Fourier along
    the others.

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
        self.naxes = mats[0].naxes[0]
        bc_mats = extract_bc_matrices([mats])
        self.mats = mats
        self.testspace = mats[0].space
        self.trialspace = mats[0].trialspace
        self.bc_mats = bc_mats
        self.solvers1D = None
        self.assemble()
        self._lu = False
        self._data = None

    def matvec(self, u, c):
        c.fill(0)
        w0 = np.zeros_like(c)
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
        self.solvers1D = []
        if ndim == 2:
            zi = np.ndindex((1, shape[1])) if self.naxes == 0 else np.ndindex((shape[0], 1))
            other_axis = (self.naxes+1) % 2
            for i in zi:
                sol = None
                for mat in self.mats:
                    sc = mat.scale[i] if mat.scale.shape[other_axis] > 1 else mat.scale[0, 0]
                    if sol:
                        sol += mat.mats[self.naxes]*sc
                    else:
                        sol = mat.mats[self.naxes]*sc
                self.solvers1D.append(Solver(sol))

        elif ndim == 3:
            s = [0, 0, 0]
            n0, n1 = np.setxor1d((0, 1, 2), self.naxes)
            for i in range(shape[n0]):
                self.solvers1D.append([])
                s[n0] = i
                for j in range(shape[n1]):
                    sol = None
                    s[n1] = j
                    for mat in self.mats:
                        sc = np.broadcast_to(mat.scale, shape)[tuple(s)]
                        if sol:
                            sol += mat.mats[self.naxes]*sc
                        else:
                            sol = mat.mats[self.naxes]*sc
                    self.solvers1D[-1].append(Solver(sol))

    def apply_constraints(self, b, constraints=()):
        #The SolverGeneric1ND solver can only constrain the first dofs of
        #the diagonal axes. For Fourier this is the zero dof with the
        #constant basis function exp(0).
        if constraints == ():
            return b
        ndim = self.mats[0].dimensions
        space = self.mats[0].space
        z0 = space.local_slice()
        paxes = np.setxor1d(range(ndim), self.naxes)
        s = [0]*ndim
        s[self.naxes] = slice(None)
        s = tuple(s)
        is_rank_zero = np.array([z0[i].start for i in paxes]).prod()
        sol = self.solvers1D[0] if ndim == 2 else self.solvers1D[0][0]
        if is_rank_zero != 0:
            return b
        sol.apply_constraints(b[s], constraints)
        return b

    def perform_lu(self):
        if self._lu is True:
            return

        if isinstance(self.solvers1D[0], SparseMatrixSolver):
            for m in self.solvers1D:
                lu = m.perform_lu()

        else:
            for mi in self.solvers1D:
                for mij in mi:
                    lu = mij.perform_lu()
        self._lu = True

    def get_data(self, is_rank_zero):
        if not self._data is None:
            return self._data

        if self.mats[0].dimensions == 2:
            data = np.zeros((len(self.solvers1D),)+self.solvers1D[-1]._inner_arg.shape)
            for i, sol in enumerate(self.solvers1D):
                if i == 0 and is_rank_zero:
                    continue
                else:
                    data[i] = sol._inner_arg
        elif self.mats[0].dimensions == 3:
            data = np.zeros((len(self.solvers1D), len(self.solvers1D[0]))+self.solvers1D[-1][-1]._inner_arg.shape)
            for i, m in enumerate(self.solvers1D):
                for j, sol in enumerate(m):
                    if i == 0 and j == 0 and is_rank_zero:
                        continue
                    else:
                        data[i, j] = sol._inner_arg
        self._data = data
        return data

    @staticmethod
    @runtimeoptimizer
    def solve_data(u, data, sol, naxes, is_rank_zero):
        s = [0]*u.ndim
        s[naxes] = slice(None)
        paxes = np.setxor1d(range(u.ndim), naxes)
        if u.ndim == 2:
            for i in range(u.shape[paxes[0]]):
                if i == 0 and is_rank_zero:
                    continue
                s[paxes[0]] = i
                s0 = tuple(s)
                sol(u[s0], data[i])

        elif u.ndim == 3:
            for i in range(u.shape[paxes[0]]):
                s[paxes[0]] = i
                for j in range(u.shape[paxes[1]]):
                    if i == 0 and j == 0 and is_rank_zero:
                        continue
                    s[paxes[1]] = j
                    s0 = tuple(s)
                    sol(u[s0], data[i, j])
        return u

    def fast_solve(self, u, b, solvers1D, naxes):
        if u is not b:
            s = tuple([slice(0, i) for i in u.shape])
            u[s] = b[s]
        # Solve first for the possibly different Fourier wavenumber 0, or (0, 0) in 3D
        # All other wavenumbers we assume have the same solver
        sol0 = solvers1D[0] if u.ndim == 2 else solvers1D[0][0]
        sol1 = solvers1D[-1] if u.ndim == 2 else solvers1D[-1][-1]
        is_rank_zero = comm.Get_rank() == 0

        if is_rank_zero:
            s = [0]*u.ndim
            s[naxes] = slice(None)
            s = tuple(s)
            sol0.inner_solve(u[s], sol0._inner_arg)

        data = self.get_data(is_rank_zero)
        sol = optimizer(sol1.inner_solve, False)
        u = self.solve_data(u, data, sol, naxes, is_rank_zero)

    def solve(self, u, b, solvers1D, naxes):
        if u is not b:
            s = tuple([slice(0, i) for i in u.shape])
            u[s] = b[s]

        s = [0]*u.ndim
        s[naxes] = slice(None)
        paxes = np.setxor1d(range(u.ndim), naxes)
        if u.ndim == 2:
            for i, sol in enumerate(solvers1D):
                s[paxes[0]] = i
                s0 = tuple(s)
                if sol._inner_arg is not None:
                    sol.inner_solve(u[s0], sol._inner_arg)

        elif u.ndim == 3:
            for i, m in enumerate(solvers1D):
                s[paxes[0]] = i
                for j, sol in enumerate(m):
                    s[paxes[1]] = j
                    s0 = tuple(s)
                    if sol._inner_arg is not None:
                        sol.inner_solve(u[s0], sol._inner_arg)

    def __call__(self, b, u=None, constraints=(), fast=True):
        """Solve problem with one non-diagonal direction

        Parameters
        ----------
        b : array, right hand side
        u : array, solution
        constraints : tuple of 2-tuples
            Each 2-tuple (row, value) is a constraint set for the non-periodic
            direction, for Fourier index 0 in 2D and (0, 0) in 3D
        fast : bool
            Use fast routine if possible. A fast routine is possible
            for any system of matrices with a tailored solver, like the
            TDMA, PDMA, FDMA and TwoDMA.

        """
        if u is None:
            u = b
        else:
            pass
            #assert u.shape == b.shape

        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(b)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)

        b = self.apply_constraints(b, constraints)

        if not self._lu:
            self.perform_lu()

        sol1 = self.solvers1D[-1] if u.ndim == 2 else self.solvers1D[-1][-1]
        if isinstance(sol1._inner_arg, tuple):
            fast = False

        if not fast:
            self.solve(u, b, self.solvers1D, self.naxes)
        else:
            self.fast_solve(u, b, self.solvers1D, self.naxes)

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
        
        if self.bc_mat: # Add contribution to right hand side due to inhomogeneous boundary conditions
            u.set_boundary_dofs()
            w0 = np.zeros_like(b)
            b -= self.bc_mat.matvec(u, w0)

        nvars = b.shape[0] if len(b.shape) > space.dimensions else 1
        u = np.expand_dims(u, 0) if nvars == 1 else u
        b = np.expand_dims(b, 0) if nvars == 1 else b
        for con in constraints:
            assert len(con) == 3
            assert isinstance(con[0], Integral)
            assert isinstance(con[1], Integral)
            assert isinstance(con[2], Number)
        self.mat.assemble()
        if self._lu is None:
            self._lu = {}

        daxes = space.get_diagonal_axes()
        if len(daxes) == space.dimensions:
            # Only Fourier spaces, all diagonal
            assert len(daxes) == space.dimensions
            Ai = self.mat._Ai[0]
            gi = b.flatten()
            if isinstance(self._lu, dict):
                for con in constraints:
                    Ai, gi = self.apply_constraint(Ai, gi, int(np.sum(np.array(space.dims()[:con[0]]))), 0, con)
            else:
                for con in constraints:
                    _, gi = self.apply_constraint(None, gi, int(np.sum(np.array(space.dims()[:con[0]]))), 0, con)
            lu = sp.linalg.splu(Ai, permc_spec=config['matrix']['block']['permc_spec'])
            self._lu = lu
            u[:] = lu.solve(gi).reshape(u.shape)

        else:
            sl, dims = space._get_ndiag_slices_and_dims()
            gi = np.zeros(dims[-1], dtype=b.dtype)
            for key, Ai in self.mat._Ai.items():
                if len(daxes) > 0:
                    sl.T[daxes+1] = key if isinstance(key, int) else np.array(key)[:, None]
                gi = b.copy_to_flattened(gi, key, dims, sl)
                if key in self._lu:
                    lu = self._lu[key]
                    for con in constraints:
                        _, gi = self.apply_constraint(None, gi, dims[con[0]], key, con)
                else:
                    for con in constraints:
                        Ai, gi = self.apply_constraint(Ai, gi, dims[con[0]], key, con)

                    lu = sp.linalg.splu(Ai, permc_spec=config['matrix']['block']['permc_spec'])
                    self._lu[key] = lu

                if b.dtype.char in 'fdg' or lu.U.dtype.char in 'FDG':
                    u = u.copy_from_flattened(lu.solve(gi), key, dims, sl)
                else:
                    u.real = u.real.copy_from_flattened(lu.solve(gi.real), key, dims, sl)
                    u.imag = u.imag.copy_from_flattened(lu.solve(gi.imag), key, dims, sl)

        u = u.reshape(u.shape[1:]) if nvars == 1 else u
        b = b.reshape(b.shape[1:]) if nvars == 1 else b
        return u
