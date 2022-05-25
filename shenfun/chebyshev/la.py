#pylint: disable=line-too-long, missing-docstring

from copy import copy
import numpy as np
from shenfun.optimization import runtimeoptimizer
from shenfun.la import SparseMatrixSolver
from shenfun.matrixbase import TPMatrix, SpectralMatrix, extract_bc_matrices,\
    get_simplified_tpmatrices


class ADDSolver(SparseMatrixSolver):
    def __init__(self, mats):
        SparseMatrixSolver.__init__(self, mats)
        assert self.mat.__class__.__name__ == 'ASDSDmat'
        #assert self.bc_mats == []

    def solve(self, b, u, axis, lu):
        if u is not b:
            u[:] = b
        self.Poisson_Solve_ADD(self.mat, b, u, axis=axis)
        return u

    def perform_lu(self):
        return 1

    @staticmethod
    @runtimeoptimizer
    def Poisson_Solve_ADD(A, b, u, axis=0):
        s = A.trialfunction[0].slice()

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        bs = b[s]
        us = u[s]
        if len(b.shape) == 1:
            se = 0.0
            so = 0.0
        else:
            se = np.zeros(us.shape[1:])
            so = np.zeros(us.shape[1:])

        d = A[0]
        d1 = A[2]
        M = us.shape
        us[-1] = bs[-1] / d[-1]
        us[-2] = bs[-2] / d[-2]
        for k in range(M[0]-3, -1, -1):
            if k%2 == 0:
                se += us[k+2]
                us[k] = bs[k] - d1[k]*se
            else:
                so += us[k+2]
                us[k] = bs[k] - d1[k]*so
            us[k] /= d[k]

        u /= A.scale
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)
        u.set_boundary_dofs()
        return u


class ANNSolver(SparseMatrixSolver):

    def __call__(self, b, u=None, axis=0, constraints=((0, 0),)):
        A = self.mat
        assert A.shape[0] + 2 == b.shape[0]
        s = A.trialfunction[0].slice()

        if u is None:
            u = b

        # Boundary conditions
        b = self.apply_bcs(b, u, axis=axis)

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if u is not b:
                b = np.moveaxis(b, axis, 0)
        bs = b[s]
        us = u[s]
        j2 = np.arange(A.shape[0])**2
        j2[0] = 1
        j2 = 1./j2
        if len(b.shape) == 1:
            se = 0.0
            so = 0.0
        else:
            se = np.zeros(u.shape[1:])
            so = np.zeros(u.shape[1:])
            j2.repeat(np.prod(bs.shape[1:])).reshape(bs.shape)
        d = A[0]*j2
        d1 = A[2]*j2[2:]
        M = us.shape
        us[-1] = bs[-1] / d[-1]
        us[-2] = bs[-2] / d[-2]
        for k in range(M[0]-3, 0, -1):
            if k%2 == 0:
                se += us[k+2]
                us[k] = bs[k] - d1[k]*se
            else:
                so += us[k+2]
                us[k] = bs[k] - d1[k]*so
            us[k] /= d[k]
        sl = [np.newaxis]*b.ndim
        sl[0] = slice(None)
        us *= j2[tuple(sl)]
        u /= A.scale
        for con in constraints:
            u[con[0]] = con[1]
        A.testfunction[0].bc.set_boundary_dofs(u, True)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)
        return u


class Helmholtz:
    r"""Helmholtz solver

    .. math::

        \alpha u'' + \beta u = b

    where :math:`u` is the solution, :math:`b` is the right hand side and
    :math:`\alpha` and :math:`\beta` are scalars, or arrays of scalars for
    a multidimensional problem.

    The user must provide mass and stiffness matrices with scale arrays
    :math:`(\alpha/\beta)` to each matrix. The matrices and scales can be
    provided as instances of :class:`.TPMatrix`, or :class:`.SpectralMatrix`.

    Parameters
    ----------
        A : :class:`.SpectralMatrix` or :class:`.TPMatrix`
            mass or stiffness matrix
        B : :class:`.SpectralMatrix` or :class:`.TPMatrix`
            mass or stiffness matrix

        scale_A : array, optional
            Scale array to stiffness matrix
        scale_B : array, optional
            Scale array to mass matrix

    The two matrices must be one stiffness and one mass matrix. Which is which
    will be found by inspection if only two arguments are provided. The scales
    :math:`\alpha` and :math:`\beta` must then be available as A.scale and
    B.scale.
    If four arguments are provided they must be in the order

        - stiffness matrix, mass matrix, scale stiffness, scale mass

    The solver can be used along any axis of a multidimensional problem. For
    example, if the Chebyshev basis (Dirichlet or Neumann) is the last in a
    3-dimensional TensorProductSpace, where the first two dimensions use Fourier,
    then the 1D Helmholtz equation arises when one is solving the 3D Poisson
    equation

    .. math::

        \nabla^2 u = b

    With the spectral Galerkin method we multiply this equation with a test
    function (:math:`v`) and integrate (weighted inner product :math:`(\cdot, \cdot)_w`)
    over the domain

    .. math::

        (v, \nabla^2 u)_w = (v, b)_w


    See :ref:`demo:poisson3d`
    for details, since it is actually quite involved. But basically, one
    obtains a linear algebra system to be solved along the :math:`z`-axis for
    all combinations of the two Fourier indices :math:`k` and :math:`l`

    .. math::

        (A_{mj} - (k^2 + l^2) B_{mj}) \hat{u}[k, l, j] = (v, b)_w[k, l, m]

    Note that :math:`k` only varies along :math:`x`-direction, whereas :math:`l`
    varies along :math:`y`. To allow for Numpy broadcasting these two variables
    are stored as arrays of shape

    .. math::

        k : (N, 1, 1)

        l : (1, M, 1)

    Here it is assumed that the solution array :math:`\hat{u}` has shape
    (N, M, P). Now, multiplying k array with :math:`\hat{u}` is achieved as an
    elementwise multiplication

    .. math::

        k \cdot \hat{u}

    Numpy will then take care of broadcasting :math:`k` to an array of shape
    (N, M, P) before performing the elementwise multiplication. Likewise, the
    constant scale :math:`1` in front of the :math:`A_{mj}` matrix is
    stored with shape (1, 1, 1), and multiplying with :math:`\hat{u}` is
    performed as if it was a scalar (as it here happens to be).

    This is where the scale arrays come from. :math:`\alpha` is here
    :math:`1`, whereas :math:`\beta` is :math:`(k^2+l^2)`. Note that
    :math:`k+l` is an array of shape (N, M, 1).

    """
    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
        self.bc_mats = []
        if isinstance(args[-1], TPMatrix):
            args = get_simplified_tpmatrices(args)
        if isinstance(args[-1], (TPMatrix, SpectralMatrix)):
            bc_mats = extract_bc_matrices([args])
            self.tpmats = args
            self.bc_mats = bc_mats

        assert len(args) in (2, 4)
        A, B = args[:2]
        M = {d.get_key(): d for d in (A, B)}
        self.A = A = M.get('ASDSDmat', M.get('ASNSNmat'))
        self.B = B = M.get('BSDSDmat', M.get('BSNSNmat'))

        if len(args) == 2:
            self.alfa = self.A.scale
            self.beta = self.B.scale
            if isinstance(self.A, TPMatrix):
                self.A = self.A.pmat
                self.B = self.B.pmat
                self.alfa *= self.A.scale
                self.beta *= self.B.scale
        elif len(args) == 4:
            self.alfa = args[2]
            self.beta = args[3]

        A, B = self.A, self.B
        B[2] = np.broadcast_to(B[2], A[2].shape)
        B[-2] = np.broadcast_to(B[-2], A[2].shape)
        v = A.testfunction[0]
        neumann = self.neumann = v.boundary_condition() == 'Neumann'
        self.axis = A.axis
        shape = [1]
        T = self.T = A.tensorproductspace
        if T is not None:
            shape = list(T.shape(True))
            shape[A.axis] = 1
        self.alfa = np.atleast_1d(self.alfa).astype(float)
        self.beta = np.atleast_1d(self.beta).astype(float)
        if not self.alfa.shape == shape:
            self.alfa = np.broadcast_to(self.alfa, shape).copy()
        if not self.beta.shape == shape:
            self.beta = np.broadcast_to(self.beta, shape).copy()

        shape[self.axis] = A.shape[0] + 2
        self.u0 = np.zeros(shape)     # Diagonal entries of U
        self.u1 = np.zeros(shape)     # Diagonal+2 entries of U
        self.u2 = np.zeros(shape)     # Diagonal+4 entries of U
        self.L = np.zeros(shape)      # The single nonzero row of L
        self.LU_Helmholtz(A, B, self.alfa, self.beta, neumann, self.u0,
                          self.u1, self.u2, self.L, self.axis)

    @staticmethod
    @runtimeoptimizer
    def LU_Helmholtz(A, B, As, Bs, neumann, u0, u1, u2, L, axis=0):
        raise NotImplementedError

    @staticmethod
    @runtimeoptimizer
    def Solve_Helmholtz(b, u, neumann, u0, u1, u2, L, axis=0):
        raise NotImplementedError

    def __call__(self, b, u=None, constraints=()):
        """Solve matrix problem

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array
                Output array

        If b and u are multidimensional, then the axis over which to solve for is
        determined on creation of the class.

        """
        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)
        self.Solve_Helmholtz(b, u, self.neumann, self.u0, self.u1, self.u2, self.L, self.axis)
        if constraints != ():
            z0 = self.T.local_slice()
            paxes = np.setxor1d(range(u.ndim), self.axis)
            is_rank_zero = np.array([z0[i].start for i in paxes]).prod()
            if is_rank_zero == 0:
                assert len(constraints) == 1
                s = [0]*u.ndim
                u[tuple(s)] = constraints[0][1]
        return u

    def matvec(self, v, c):
        """Matrix vector product c = dot(self, v)

        Parameters
        ----------
            v : array
            c : array

        Returns
        -------
            c : array
        """
        c[:] = 0
        if not self.neumann:
            self.Helmholtz_matvec(v, c, self.alfa, self.beta, self.A, self.B, self.axis)
        else:
            self.Helmholtz_Neumann_matvec(v, c, self.alfa, self.beta, self.A, self.B, self.axis)
        if len(self.bc_mats) > 0:
            v.set_boundary_dofs()
            w0 = np.zeros_like(v)
            for bc_mat in self.bc_mats:
                c += bc_mat.matvec(v, w0)
        return c

    @staticmethod
    @runtimeoptimizer
    def Helmholtz_matvec(v, b, alfa, beta, A, B, axis=0):
        raise NotImplementedError("Use Cython or Numba")

    @staticmethod
    @runtimeoptimizer
    def Helmholtz_Neumann_matvec(v, b, alfa, beta, A, B, axis=0):
        raise NotImplementedError("Use Cython or Numba")


class Biharmonic:
    r"""Multidimensional Biharmonic solver for

    .. math::

        a_0 u'''' + \alpha u'' + \beta u = b

    where :math:`u` is the solution, :math:`b` is the right hand side and
    :math:`a_0, \alpha` and :math:`\beta` are scalars, or arrays of scalars for
    a multidimensional problem.

    The user must provide mass, stiffness and biharmonic matrices with
    associated scale arrays :math:`(a_0/\alpha/\beta)`. The matrices and scales
    can be provided in any order

    Parameters
    ----------
        S : :class:`.TPMatrix` or :class:`.SpectralMatrix`
        A : :class:`.TPMatrix` or :class:`.SpectralMatrix`
        B : :class:`.TPMatrix` or :class:`.SpectralMatrix`

        scale_S : array, optional
        scale_A : array, optional
        scale_B : array, optional

    If only three arguments are passed, then we decide which matrix is which
    through inspection. The three scale arrays must then be available as
    S.scale, A.scale, B.scale.
    If six arguments are provided they must be in order S, A, B, scale S,
    scale A, scale B.

    Variables are extracted from the matrices

    The solver can be used along any axis of a multidimensional problem. For
    example, if the Chebyshev basis (Biharmonic) is the last in a
    3-dimensional TensorProductSpace, where the first two dimensions use Fourier,
    then the 1D equation listed above arises when one is solving the 3D biharmonic
    equation

    .. math::

        \nabla^4 u = b

    With the spectral Galerkin method we multiply this equation with a test
    function (:math:`v`) and integrate (weighted inner product :math:`(\cdot, \cdot)_w`)
    over the domain

    .. math::

        (v, \nabla^4 u)_w = (v, b)_w

    See `the Poisson problem <https://rawgit.com/spectralDNS/shenfun/master/docs/src/mekit17/pub/._shenfun_bootstrap004.html#sec:tensorproductspaces>`_
    for details, since it is actually quite involved. But basically, one obtains
    a linear algebra system to be solved along the :math:`z`-axis for all combinations
    of the two Fourier indices :math:`k` and :math:`l`

    .. math::

        (S_{mj} - 2(k^2 + l^2) A_{mj}) + (k^2 + l^2)^2 B_{mj}) \hat{u}[k, l, j] = (v, b)_w[k, l, m]

    Note that :math:`k` only varies along :math:`x`-direction, whereas :math:`l` varies along
    :math:`y`. To allow for Numpy broadcasting these two variables are stored as arrays of
    shape

    .. math::

        k : (N, 1, 1)

        l : (1, M, 1)

    Here it is assumed that the solution array :math:`\hat{u}` has shape
    (N, M, P). Now, multiplying :math:`k` array with :math:`\hat{u}` is achieved as

    .. math::

        k \cdot \hat{u}

    Numpy will then take care of broadcasting :math:`k` to an array of shape (N, M, P)
    before performing the elementwise multiplication. Likewise, the constant
    scale :math:`1` in front of the :math:`A_{mj}` matrix is stored with
    shape (1, 1, 1), and multiplying with :math:`\hat{u}` is performed as if it
    was a scalar (as it here happens to be).

    This is where the scale arrays in the signature to the Helmholt solver comes
    from. :math:`a_0` is here :math:`1`, whereas :math:`\alpha` and
    :math:`\beta` are :math:`-2(k^2+l^2)` and :math:`(k^2+l^2)^2`, respectively.
    Note that :math:`k+l` is an array of shape (N, M, 1).
    """
    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]

        args = list(args)
        self.bc_mats = []
        if isinstance(args[-1], TPMatrix):
            args = get_simplified_tpmatrices(args)
        if isinstance(args[-1], (TPMatrix, SpectralMatrix)):
            bc_mats = extract_bc_matrices([args])
            self.tpmats = args
            self.bc_mats = bc_mats

        assert len(args) in (3, 6)

        S, A, B = args[:3]
        M = {d.get_key(): d for d in (S, A, B)}
        self.S = M['SSBSBmat']
        self.A = M['ASBSBmat']
        self.B = M['BSBSBmat']

        if len(args) == 3:
            self.a0 = a0 = np.atleast_1d(self.S.scale).item()
            self.alfa = alfa = self.A.scale
            self.beta = beta = self.B.scale
            if isinstance(self.S, TPMatrix):
                self.S = self.S.pmat
                self.A = self.A.pmat
                self.B = self.B.pmat
                self.alfa *= self.A.scale
                self.beta *= self.B.scale
        elif len(args) == 6:
            self.a0 = a0 = np.atleast_1d(args[3]).item()
            self.alfa = alfa = args[4]
            self.beta = beta = args[5]

        S, A, B = self.S, self.A, self.B
        self.axis = S.axis
        T = S.tensorproductspace
        if T is None:
            shape = [S[0].shape]
        else:
            shape = list(T.shape(True))
        sii, siu, siuu = S[0], S[2], S[4]
        ail, aii, aiu = A[-2], A[0], A[2]
        bill, bil, bii, biu, biuu = B[-4], B[-2], B[0], B[2], B[4]
        M = sii[::2].shape[0]
        shape[S.axis] = M
        ss = copy(shape)
        ss.insert(0, 2)
        self.u0 = np.zeros(ss)
        self.u1 = np.zeros(ss)
        self.u2 = np.zeros(ss)
        self.l0 = np.zeros(ss)
        self.l1 = np.zeros(ss)
        self.ak = np.zeros(ss)
        self.bk = np.zeros(ss)
        self.LU_Biharmonic(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                           bill, bil, bii, biu, biuu, self.u0, self.u1,
                           self.u2, self.l0, self.l1, self.axis)
        self.Biharmonic_factor_pr(self.ak, self.bk, self.l0, self.l1, self.axis)

    @staticmethod
    @runtimeoptimizer
    def LU_Biharmonic(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                      bill, bil, bii, biu, biuu, u0, u1, u2, l0, l1, axis):
        raise NotImplementedError('Use Cython or Numba')

    @staticmethod
    @runtimeoptimizer
    def Biharmonic_factor_pr(ak, bk, l0, l1, axis):
        raise NotImplementedError('Use Cython or Numba')

    @staticmethod
    @runtimeoptimizer
    def Biharmonic_Solve(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0):
        raise NotImplementedError('Use Cython or Numba')

    @staticmethod
    @runtimeoptimizer
    def Biharmonic_matvec(v, b, a0, alfa, beta,
                          sii, siu, siuu, ail, aii, aiu,
                          bill, bil, bii, biu, biuu, axis=0):
        raise NotImplementedError('Use Cython or Numba')

    def __call__(self, b, u=None, **kw):
        """Solve matrix problem

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array
                Output array

        If b and u are multidimensional, then the axis over which to solve for is
        determined on creation of the class.

        """
        if len(self.bc_mats) > 0:
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            for bc_mat in self.bc_mats:
                b -= bc_mat.matvec(u, w0)
        self.Biharmonic_Solve(b, u, self.u0, self.u1, self.u2, self.l0,
                              self.l1, self.ak, self.bk, self.a0, self.axis)

        return u

    def matvec(self, v, c):
        c[:] = 0
        self.Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S[0],
                               self.S[2], self.S[4], self.A[-2], self.A[0],
                               self.A[2], self.B[-4], self.B[-2], self.B[0],
                               self.B[2], self.B[4], self.axis)
        return c
