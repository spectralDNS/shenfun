r"""
This module contains linear algebra solvers for SparseMatrixes
"""
import numpy as np
import scipy.sparse as scp
from scipy.sparse.linalg import spsolve, splu
from shenfun.optimization import optimizer
from shenfun.matrixbase import SparseMatrix, extract_bc_matrices, SpectralMatrix


class TDMA:
    """Tridiagonal matrix solver

    Parameters
    ----------
    mat : SparseMatrix
        Tridiagonal matrix with diagonals in offsets -2, 0, 2

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, mat, neumann=False):
        assert isinstance(mat, SparseMatrix)
        self.mat = mat
        self.dd = np.zeros(0)
        self.neumann = neumann
        if isinstance(mat, SpectralMatrix):
            self.neumann = mat.testfunction[0].boundary_condition().lower() == 'neumann'

    def init(self):
        """Initialize and allocate solver"""
        N = self.mat.shape[0]
        self.symmetric = self.mat.issymmetric
        self.dd = self.mat[0]*np.ones(N)*self.mat.scale
        self.ud = self.mat[2]*np.ones(N-2)*self.mat.scale
        if self.neumann:
            self.dd[0] = 1
            self.ud[0] = 0
        self.ld = np.zeros(N-2) if self.symmetric else self.mat[-2]*np.ones(N-2)*self.mat.scale
        if self.symmetric:
            self.TDMA_SymLU(self.dd, self.ud, self.ld)
        else:
            self.TDMA_LU(self.ld, self.dd, self.ud)

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

    def __call__(self, b, u=None, axis=0, **kw):
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
            N = self.dd.shape[0]
            u[:] = b[:]

        if not self.dd.shape[0] == self.mat.shape[0]:
            self.init()

        if self.neumann:
            T = self.mat.testfunction[0]
            u[T.si[0]] = 0
            if isinstance(self, SpectralMatrix):
                u[T.si[0]] = self.mat.testfunction[0].mean

        self.TDMA_SymSolve(self.dd, self.ud, self.ld, u, axis=axis)

        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()

        if self.neumann:
            u[T.si[0]] = 0
            if isinstance(self, SpectralMatrix):
                u[T.si[0]] = self.mat.testfunction[0].mean

        return u

class PDMA:
    """Pentadiagonal matrix solver

    Parameters
    ----------
        mat : SparseMatrix
            Pentadiagonal matrix with diagonals in offsets
            -4, -2, 0, 2, 4
        neumann : bool, optional
            Whether matrix represents a Neumann problem, where
            the first index is known as the mean value and we
            solve for slice(1, N-3).
            If `mat` is a :class:`.SpectralMatrix`, then the
            `neumann` keyword is ignored and the information
            extracted from the matrix.

    """

    def __init__(self, mat, neumann=False):
        assert isinstance(mat, SparseMatrix)
        self.mat = mat
        self.N = 0
        self.d0 = np.zeros(0)
        self.d1 = None
        self.d2 = None
        self.A = None
        self.L = None
        self.neumann = neumann
        if isinstance(mat, SpectralMatrix):
            self.neumann = mat.testfunction[0].boundary_condition().lower() == 'neumann'

    def init(self):
        """Initialize and allocate solver"""
        B = self.mat
        shape = self.mat.shape[1]
        # Broadcast in case diagonal is simply a constant.
        self.d0 = np.broadcast_to(np.atleast_1d(B[0]), shape).copy()*B.scale
        self.d1 = np.broadcast_to(np.atleast_1d(B[2]), shape-2).copy()*B.scale
        self.d2 = np.broadcast_to(np.atleast_1d(B[4]), shape-4).copy()*B.scale
        if self.neumann:
            self.d0[0] = 1
            self.d1[0] = 0
            self.d2[0] = 0
        if B.issymmetric:
            self.PDMA_SymLU(self.d0, self.d1, self.d2)
        else:
            self.l1 = np.broadcast_to(np.atleast_1d(B[-2]), shape-2).copy()*B.scale
            self.l2 = np.broadcast_to(np.atleast_1d(B[-4]), shape-4).copy()*B.scale
            self.PDMA_LU(self.l2, self.l1, self.d0, self.d1, self.d2)

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
    def PDMA_SymSolve(d, e, f, b, axis=0): # pragma: no cover
        """Symmetric solve (for testing only)"""
        n = d.shape[0]
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

    @staticmethod
    @optimizer
    def PDMA_Solve(a, b, d, e, f, h, axis=0): # pragma: no cover
        """Symmetric solve (for testing only)"""
        n = d.shape[0]
        bc = h

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

    def __call__(self, b, u=None, axis=0, **kw):
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

        if not self.d0.shape[0] == self.mat[0].shape[0]:
            self.init()

        if self.neumann:
            T = self.mat.testfunction[0]
            u[T.si[0]] = 0
            if isinstance(self, SpectralMatrix):
                u[T.si[0]] = T.mean

        if self.mat.issymmetric:
            self.PDMA_SymSolve(self.d0, self.d1, self.d2, u, axis)
        else:
            self.PDMA_Solve(self.l2, self.l1, self.d0, self.d1, self.d2, u, axis)

        if hasattr(u, 'set_boundary_dofs'):
            u.set_boundary_dofs()

        return u

class Solve:
    """Generic solver class for SparseMatrix

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

    def __call__(self, b, u=None, axis=0, use_lu=False):
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

        if self.test.use_fixed_gauge:
            A = self.A.diags('lil')
            _, zerorow = A[0].nonzero()
            A[(0, zerorow)] = 0
            A[0, 0] = 1
            b[0] = self.test.mean
            A = A.tocsc()
        else:
            A = self.A.diags('csc')

        if b.ndim == 1:
            if use_lu:
                if b.dtype.char in 'fdg' or self._lu.U.dtype.char in 'FDG':
                    u[s] = self._lu.solve(b[s])
                else: # complex b and real matrix
                    u.real[s] = self._lu.solve(b[s].real)
                    u.imag[s] = self._lu.solve(b[s].imag)

            else:
                u[s] = spsolve(A, b[s])

        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))
            if use_lu:
                if b.dtype.char in 'fdg' or self._lu.U.dtype.char in 'FDG':
                    u[s] = self._lu.solve(br).reshape(u[s].shape)
                else: # complex b and real matrix
                    u.real[s] = self._lu.solve(br.real).reshape(u[s].shape)
                    u.imag[s] = self._lu.solve(br.imag).reshape(u[s].shape)

            else:
                u[s] = spsolve(A, br).reshape(u[s].shape)

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

    def __call__(self, b, u=None, axis=0, use_lu=False):
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
            if use_lu:
                if b.dtype.char in 'fdg' or self._lu.U.dtype.char in 'FDG':
                    u[s] = self._lu.solve(b[s])
                else:
                    u.real[s] = self._lu.solve(b[s].real)
                    u.imag[s] = self._lu.solve(b[s].imag)

            else:
                u[s] = spsolve(A, b[s])

        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))

            if use_lu:
                if b.dtype.char in 'fdg' or self._lu.U.dtype.char in 'FDG':
                    u[s] = self._lu.solve(br).reshape(u[s].shape)
                else:
                    u.real[s] = self._lu.solve(br.real).reshape(u[s].shape)
                    u.imag[s] = self._lu.solve(br.imag).reshape(u[s].shape)
            else:
                u[s] = spsolve(A, br).reshape(u[s].shape)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)
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

    def get_gauged_matrices(self, m0, m1):
        if self.T.use_fixed_gauge:
            A0 = m0.diags('lil')
            if m0.testfunction[1]+m0.trialfunction[1] == 2:
                zerorow = A0[0].nonzero()[1]
                A0[(0, zerorow)] = 0
                A0[0, 0] = 1
            A0 = A0.tocsc()

            A1 = m1.diags('lil')
            if m1.testfunction[1]+m1.trialfunction[1] == 2:
                zerorow = A1[0].nonzero()[1]
                A1[(0, zerorow)] = 0
                A1[0, 0] = 1
            A1 = A1.tocsc()
        else:
            A0 = m0.diags('csc')
            A1 = m1.diags('csc')
        return A0, A1

    def diags(self, i):
        """Return matrix for given index `i` in diagonal direction"""
        if self.T.dimensions == 2:
            # In 2D there's just 1 matrix, store and reuse
            if self.M is not None:
                return self.M
            m = self.tpmats[0]
            A0, A1 = self.get_gauged_matrices(m.mats[0], m.mats[1])
            M0 = scp.kron(A0, A1, 'csc')
            M0 *= np.atleast_1d(m.scale).item()
            for m in self.tpmats[1:]:
                A0, A1 = self.get_gauged_matrices(m.mats[0], m.mats[1])
                M1 = scp.kron(A0, A1, 'csc')
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
            A0, A1 = self.get_gauged_matrices(m.mats[naxes[0]], m.mats[naxes[1]])
            M0 = scp.kron(A0, A1, 'csc')
            M0 *= m.scale[tuple(sc)]
            for m in self.tpmats[1:]:
                A0, A1 = self.get_gauged_matrices(m.mats[naxes[0]], m.mats[naxes[1]])
                M1 = scp.kron(A0, A1, 'csc')
                sc[diagonal_axis] = i if m.scale.shape[diagonal_axis] > 1 else 0
                M1 *= m.scale[tuple(sc)]
                M0 = M0 + M1
            return M0

    def __call__(self, b, u=None):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
        if u.ndim == 2:
            s0 = self.T.slice()
            M = self.diags(0)
            u[s0] = scp.linalg.spsolve(M, b[s0].flatten()).reshape(self.T.dims())

        elif u.ndim == 3:
            naxes = self.T.get_nondiagonal_axes()
            diagonal_axis = self.get_diagonal_axis()
            s0 = list(self.T.slice())
            for i in range(self.T.shape(True)[diagonal_axis]):
                M0 = self.diags(i)
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

    Note
    ----
    If there are boundary matrices in the list of mats, then
    these matrices are used to modify the right hand side before
    solving. If this is not the desired behaviour, then use
    :func:`.extract_bc_matrices` on mats before using this class.

    """

    def __init__(self, tpmats):
        bc_mats = extract_bc_matrices([tpmats])
        self.tpmats = tpmats
        self.bc_mats = bc_mats
        m = tpmats[0]
        self.T = T = m.space
        ndim = T.dimensions
        assert ndim == 2
        assert np.atleast_1d(m.scale).size == 1, "Use level = 2 with :func:`.inner`"

        A0, A1 = self.get_gauged_matrices(m.mats[0], m.mats[1])
        M0 = scp.kron(A0, A1, 'csc')
        M0 *= np.atleast_1d(m.scale).item()
        for m in tpmats[1:]:
            A0, A1 = self.get_gauged_matrices(m.mats[0], m.mats[1])
            M1 = scp.kron(A0, A1, 'csc')
            assert np.atleast_1d(m.scale).size == 1, "Use level = 2 with :func:`.inner`"
            M1 *= np.atleast_1d(m.scale).item()
            M0 = M0 + M1
        self.M = M0

    def matvec(self, u, c):
        c.fill(0)
        s0 = tuple(base.slice() for base in self.T)
        c[s0] = self.M.dot(u[s0].flatten()).reshape(self.T.dims())
        return c

    def get_gauged_matrices(self, m0, m1):
        if self.T.use_fixed_gauge:
            A0 = m0.diags('lil')
            if m0.testfunction[1]+m0.trialfunction[1] == 2:
                zerorow = A0[0].nonzero()[1]
                A0[(0, zerorow)] = 0
                A0[0, 0] = 1
            A0 = A0.tocsc()

            A1 = m1.diags('lil')
            if m1.testfunction[1]+m1.trialfunction[1] == 2:
                zerorow = A1[0].nonzero()[1]
                A1[(0, zerorow)] = 0
                A1[0, 0] = 1
            A1 = A1.tocsc()
        else:
            A0 = m0.diags('csc')
            A1 = m1.diags('csc')
        return A0, A1

    def __call__(self, b, u=None):
        try:
            from codetiming import Timer
            has_timer = True
        except:
            has_timer = False

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
        if b.dtype.char in 'fdg':
            u[s0] = scp.linalg.spsolve(self.M, b[s0].flatten()).reshape(self.T.dims())
        else:
            if self.M.dtype.char in 'FDG':
                lu = splu(self.M)
                u[s0] = lu.solve(b[s0].flatten()).reshape(self.T.dims())

            else:
                lu = splu(self.M)
                u.imag[s0] = lu.solve(b.imag[s0].flatten()).reshape(self.T.dims())
                u.real[s0] = lu.solve(b.real[s0].flatten()).reshape(self.T.dims())

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
    up to two diagonal (Fourier) directions.
    Also note that if there are boundary matrices in the list of mats, then
    these matrices are used to modify the right hand side before
    solving. If this is not the desired behaviour, then use
    :func:`.extract_bc_matrices` on mats before using this class.

    """

    def __init__(self, mats):
        assert isinstance(mats, list)
        naxes = set()
        for tpmat in mats:
            if not tpmat._issimplified:
                tpmat.simplify_diagonal_matrices()
            naxes.update(tpmat.naxes)
        assert len(naxes) == 1
        self.naxes = naxes.pop()
        bc_mats = extract_bc_matrices([mats])
        self.mats = mats
        self.bc_mats = bc_mats
        # For time-dependent solver, store all generated matrices and reuse
        # This takes a lot of memory, so for now it's only implemented for 2D
        self.MM = None

    @staticmethod
    def apply_constraint(A, b, i, constraint):
        assert isinstance(constraint, tuple)
        assert len(constraint) == 4

        if constraint is None:
            return A, b

        if not i == constraint[0]:
            return A, b

        row = constraint[1]
        col = constraint[2]
        val = constraint[3]
        b[row] = val
        r = A.getrow(row).nonzero()
        A[(row, r[1])] = 0
        A[row, col] = 1
        return A, b

    def __call__(self, b, u=None, constraints=()):
        """Solve problem with one non-diagonal direction

        Parameters
        ----------
        b : array, right hand side
        u : array, solution
        constraints : tuple of 4-tuples
            Each 4-tuple is a constraint, with each item representing
              - 0 : The diagonal index, or indices for 3D
              - 1 : row
              - 2 : column
              - 3 : value
            Matrix row is zeroed and then indented by setting A[row, col] = 0

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

        if u.ndim == 2:

            if self.naxes == 0:
                # non-diagonal in axis=0

                if self.MM is None:
                    self.MM = []
                    for i in range(b.shape[1]):
                        MM = None
                        for mat in self.mats:
                            sc = mat.scale[0, i] if mat.scale.shape[1] > 1 else mat.scale[0, 0]
                            if MM:
                                MM += mat.mats[0]*sc
                            else:
                                MM = mat.mats[0]*sc
                        sl = m.space.bases[0].slice()
                        #u[sl, i] = MM.solve(b[sl, i], u[sl, i])
                        Mc = MM.diags('csc')
                        for constraint in constraints:
                            Mc, b = self.apply_constraint(Mc, b, i, constraint)
                        try:
                            MM._lu = splu(Mc)
                            u[sl, i] = MM.solve(b[sl, i], u[sl, i], use_lu=True)
                        except:
                            print('Singular matrix for j=', i)
                            u[sl, i] = 0
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
                                MM += mat.mats[1]*sc
                            else:
                                MM = mat.mats[1]*sc
                        sl = m.space.bases[1].slice()
                        Mc = MM.diags('csc')
                        for constraint in constraints:
                            Mc, b = self.apply_constraint(Mc, b, i, constraint)
                        MM._lu = splu(Mc)
                        MM.solve(b[i, sl], u[i, sl], use_lu=True)
                        self.MM.append(MM)

                else:
                    for i in range(b.shape[0]):
                        sl = m.space.bases[1].slice()
                        u[i, sl] = self.MM[i].solve(b[i, sl], u[i, sl], use_lu=True)

        elif u.ndim == 3:
            if self.naxes == 0:
                # non-diagonal in axis=0
                for i in range(b.shape[1]):
                    for j in range(b.shape[2]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, u.shape)[0, i, j]
                            if MM:
                                MM += mat.mats[0]*sc
                            else:
                                MM = mat.mats[0]*sc
                        Mc = MM.diags('csc')
                        for constraint in constraints:
                            Mc, b = self.apply_constraint(Mc, b, (i, j), constraint)
                        MM._lu = splu(Mc)
                        sl = mat.space.bases[0].slice()
                        u[sl, i, j] = MM.solve(b[sl, i, j], u[sl, i, j], use_lu=True)

            elif self.naxes == 1:
                # non-diagonal in axis=1
                for i in range(b.shape[0]):
                    for j in range(b.shape[2]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, u.shape)[i, 0, j]
                            if MM:
                                MM += mat.mats[1]*sc
                            else:
                                MM = mat.mats[1]*sc
                        Mc = MM.diags('csc')
                        for constraint in constraints:
                            Mc, b = self.apply_constraint(Mc, b, (i, j), constraint)
                        MM._lu = splu(Mc)
                        sl = mat.space.bases[1].slice()
                        u[i, sl, j] = MM.solve(b[i, sl, j], u[i, sl, j], use_lu=True)

            elif self.naxes == 2:
                # non-diagonal in axis=2
                for i in range(b.shape[0]):
                    for j in range(b.shape[1]):
                        MM = None
                        for mat in self.mats:
                            sc = np.broadcast_to(mat.scale, u.shape)[i, j, 0]
                            if MM:
                                MM += mat.mats[2]*sc
                            else:
                                MM = mat.mats[2]*sc
                        Mc = MM.diags('csc')
                        for constraint in constraints:
                            Mc, b = self.apply_constraint(Mc, b, (i, j), constraint)
                        MM._lu = splu(Mc)
                        sl = mat.space.bases[2].slice()
                        u[i, j, sl] = MM.solve(b[i, j, sl], u[i, j, sl], use_lu=True)

        return u
