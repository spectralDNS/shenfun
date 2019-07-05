#pylint: disable=line-too-long, missing-docstring

from copy import copy
import numpy as np
from shenfun.optimization import optimizer
from shenfun.optimization.cython import la
from shenfun.la import TDMA as la_TDMA
from shenfun.utilities import inheritdocstrings
from shenfun.matrixbase import TPMatrix

@inheritdocstrings
class TDMA(la_TDMA):

    def __call__(self, b, u=None, axis=0):

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        if not self.dd.shape[0] == self.mat.shape[0]:
            self.init()

        self.TDMA_SymSolve(self.dd, self.ud, self.L, u, axis=axis)

        if not self.mat.scale in (1, 1.0):
            u /= self.mat.scale
        return u

class Helmholtz(object):
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

    Attributes
    ----------
        axis : int
            The axis over which to solve for
        neumann : bool
            Whether or not bases are Neumann
        bc : BoundaryValues
            For Dirichlet problem with inhomogeneous boundary values

    Variables are extracted from the matrices

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

        args = list(args)
        for i, arg in enumerate(args):
            if hasattr(arg, 'is_bc_matrix'):
                if arg.is_bc_matrix():
                    # For this particular case the boundary dofs contribution
                    # to the right hand side is only nonzero for Fourier wavenumber
                    # 0, so the contribution is in effect zero
                    args.pop(i)
                    break

        assert len(args) in (2, 4)
        A, B = args[0], args[1]
        M = {d.get_key(): d for d in (A, B)}
        self.A = A = M.get('ADDmat', M.get('ANNmat'))
        self.B = B = M.get('BDDmat', M.get('BNNmat'))

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
        if not self.neumann:
            self.bc = v.bc
        self.axis = A.axis
        shape = [1]
        T = A.tensorproductspace
        if T is not None:
            shape = list(T.shape(True))
            shape[A.axis] = 1
        self.alfa = np.atleast_1d(self.alfa)
        self.beta = np.atleast_1d(self.beta)
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
    @optimizer
    def LU_Helmholtz(A, B, As, Bs, neumann, u0, u1, u2, L, axis=0):
        raise NotImplementedError

    @staticmethod
    @optimizer
    def Solve_Helmholtz(b, u, neumann, u0, u1, u2, L, axis=0):
        raise NotImplementedError

    def __call__(self, u, b):
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
        self.Solve_Helmholtz(b, u, self.neumann, self.u0, self.u1, self.u2, self.L, self.axis)

        if not self.neumann:
            self.bc.apply_after(u, True)

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
        assert self.neumann is False
        c[:] = 0
        self.Helmholtz_matvec(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0], self.axis)
        return c

    @staticmethod
    @optimizer
    def Helmholtz_matvec(v, b, alfa, beta, dd, ud, bd, axis=0):
        raise NotImplementedError("Use Cython or Numba")

class Biharmonic(object):
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

        assert len(args) in (3, 6)
        S, A, B = args[0], args[1], args[2]
        M = {d.get_key(): d for d in (S, A, B)}
        self.S = M['SBBmat']
        self.A = M['ABBmat']
        self.B = M['BBBmat']

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
            self.a0 = a0 = args[3]
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
    @optimizer
    def LU_Biharmonic(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                      bill, bil, bii, biu, biuu, u0, u1, u2, l0, l1, axis):
        raise NotImplementedError('Use Cython or Numba')

    @staticmethod
    @optimizer
    def Biharmonic_factor_pr(ak, bk, l0, l1, axis):
        raise NotImplementedError('Use Cython or Numba')

    @staticmethod
    @optimizer
    def Biharmonic_Solve(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0):
        raise NotImplementedError('Use Cython or Numba')

    @staticmethod
    @optimizer
    def Biharmonic_matvec(v, b, a0, alfa, beta,
                          sii, siu, siuu, ail, aii, aiu,
                          bill, bil, bii, biu, biuu, axis=0):
        raise NotImplementedError('Use Cython or Numba')

    def __call__(self, u, b):
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


class PDMA(object):
    r"""Pentadiagonal matrix solver

    Pentadiagonal matrix with diagonals in offsets -4, -2, 0, 2, 4

    Arising with Poisson equation and biharmonic basis u

    .. math::

        \alpha u'' + \beta u = f

    As 4 arguments

    Parameters
    ----------
        A : SpectralMatrix
            Stiffness matrix
        B : SpectralMatrix
            Mass matrix
        alfa : array
        beta : array

    or as dict with key/vals

    Parameters
    ----------
        solver : str
            Choose between implementations ('cython', 'python')
        ABBmat : A
            Stiffness matrix
        BBBmat : B
            Mass matrix

    where alfa and beta must be avalable as A.scale, B.scale.

    Attributes
    ----------
        axis : int
            The axis over which to solve for

    Variables are extracted from the matrices

    """

    def __init__(self, *args, **kwargs):
        if 'ABBmat' in kwargs:
            assert 'BBBmat' in kwargs
            A = self.A = kwargs['ABBmat']
            B = self.B = kwargs['BBBmat']
            self.alfa = A.scale
            self.beta = B.scale

        elif len(args) == 4:
            A = self.A = args[0]
            B = self.B = args[1]
            self.alfa = args[2]
            self.beta = args[3]
        else:
            raise RuntimeError('Wrong input to PDMA solver')

        self.solver = kwargs.get('solver', 'cython')
        self.d, self.u1, self.u2 = (np.zeros_like(B[0]), np.zeros_like(B[2]),
                                    np.zeros_like(B[4]))
        self.l1, self.l2 = np.zeros_like(B[2]), np.zeros_like(B[4])
        self.alfa = np.atleast_1d(self.alfa)
        self.beta = np.atleast_1d(self.beta)
        shape = list(self.beta.shape)

        if len(shape) == 1:
            if self.solver == 'python':
                H = self.alfa[0]*self.A + self.beta[0]*self.B
                self.d[:] = H[0]
                self.u1[:] = H[2]
                self.u2[:] = H[4]
                self.l1[:] = H[-2]
                self.l2[:] = H[-4]
                self.PDMA_LU(self.l2, self.l1, self.d, self.u1, self.u2)

            elif self.solver == 'cython':
                la.LU_Helmholtz_Biharmonic_1D(self.A, self.B, self.alfa[0],
                                              self.beta[0], self.l2, self.l1,
                                              self.d, self.u1, self.u2)
        else:

            self.axis = A.axis
            assert self.alfa.shape[A.axis] == 1
            assert self.beta.shape[A.axis] == 1
            N = A.shape[0]+4
            self.alfa = np.broadcast_to(self.alfa, shape).copy()
            shape[A.axis] = N
            self.d = np.zeros(shape, float)      # Diagonal entries of U
            self.u1 = np.zeros(shape, float)     # Diagonal+2 entries of U
            self.l1 = np.zeros(shape, float)     # Diagonal-2 entries of U
            self.u2 = np.zeros(shape, float)     # Diagonal+4 entries of U
            self.l2 = np.zeros(shape, float)     # Diagonal-4 entries of U

            if len(shape) == 2:
                raise NotImplementedError

            elif len(shape) == 3:
                la.LU_Helmholtz_Biharmonic_3D(A, B, A.axis, self.alfa, self.beta, self.l2,
                                              self.l1, self.d, self.u1, self.u2)

    @staticmethod
    def PDMA_LU(l2, l1, d, u1, u2): # pragma: no cover
        """LU decomposition of PDM (for testing only)"""
        n = d.shape[0]
        m = u1.shape[0]
        k = n - m

        for i in range(n-2*k):
            lam = l1[i]/d[i]
            d[i+k] -= lam*u1[i]
            u1[i+k] -= lam*u2[i]
            l1[i] = lam
            lam = l2[i]/d[i]
            l1[i+k] -= lam*u1[i]
            d[i+2*k] -= lam*u2[i]
            l2[i] = lam

        i = n-4
        lam = l1[i]/d[i]
        d[i+k] -= lam*u1[i]
        l1[i] = lam
        i = n-3
        lam = l1[i]/d[i]
        d[i+k] -= lam*u1[i]
        l1[i] = lam


    @staticmethod
    def PDMA_Solve(l2, l1, d, u1, u2, b): # pragma: no cover
        """Solve method for PDM (for testing only)"""
        n = d.shape[0]
        bc = np.full_like(b, b)

        bc[2] -= l1[0]*bc[0]
        bc[3] -= l1[1]*bc[1]
        for k in range(4, n):
            bc[k] -= (l1[k-2]*bc[k-2] + l2[k-4]*bc[k-4])

        bc[n-1] /= d[n-1]
        bc[n-2] /= d[n-2]
        bc[n-3] /= d[n-3]
        bc[n-3] -= u1[n-3]*bc[n-1]/d[n-3]
        bc[n-4] /= d[n-4]
        bc[n-4] -= u1[n-4]*bc[n-2]/d[n-4]
        for k in range(n-5, -1, -1):
            bc[k] /= d[k]
            bc[k] -= (u1[k]*bc[k+2]/d[k] + u2[k]*bc[k+4]/d[k])
        b[:] = bc

    def __call__(self, u, b):
        """Solve matrix problem

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array
                Output array

        If b and u are multidimensional, then the axis over which to solve for
        is determined on creation of the class.

        """
        if np.ndim(u) == 3:
            la.Solve_Helmholtz_Biharmonic_3D_ptr(self.A.axis, b, u, self.l2,
                                                 self.l1, self.d, self.u1,
                                                 self.u2)
        elif np.ndim(u) == 2:
            la.Solve_Helmholtz_Biharmonic_2D_ptr(self.A.axis, b, u, self.l2,
                                                 self.l1, self.d, self.u1,
                                                 self.u2)
        else:
            if self.solver == 'python': # pragma: no cover
                u[:] = b
                self.PDMA_Solve(self.l2, self.l1, self.d, self.u1, self.u2, u)

            elif self.solver == 'cython':
                if u is b:
                    u = np.zeros_like(b)
                #la.Solve_Helmholtz_Biharmonic_1D(b, u, self.l2, self.l1,
                #                                 self.d, self.u1, self.u2)
                la.Solve_Helmholtz_Biharmonic_1D_p(b, u, self.l2, self.l1,
                                                   self.d, self.u1, self.u2)

        #u /= self.mat.scale
        return u
