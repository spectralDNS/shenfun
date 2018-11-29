#pylint: disable=line-too-long, len-as-condition, missing-docstring, too-many-instance-attributes

from copy import copy
import numpy as np
import scipy.linalg as scipy_la
from shenfun.optimization import optimizer
from shenfun.optimization.cython import la
from shenfun.utilities import inheritdocstrings
from shenfun.la import TDMA as la_TDMA

@inheritdocstrings
class TDMA(la_TDMA):
    """Tridiagonal matrix solver

    Parameters
    ----------
        mat : SparseMatrix
              Symmetric tridiagonal matrix with diagonals in offsets -2, 0, 2

    """
    def __call__(self, b, u=None, axis=0):

        v = self.mat.testfunction[0]
        bc = v.bc

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        if v.is_scaled():
            bc.apply_before(u, False, scales=(-1./np.sqrt(6.), -1./3./np.sqrt(10.)))
        else:
            bc.apply_before(u, False, scales=(-1., -1./3.))

        if not self.dd.shape[0] == self.mat.shape[0]:
            self.init()

        self.TDMA_SymSolve(self.dd, self.ud, self.L, u, axis=axis)

        bc.apply_after(u, False)

        u /= self.mat.scale
        return u


class Helmholtz(object):
    r"""Helmholtz solver

    .. math::

        \alpha u'' + \beta u = b

    where :math:`u` is the solution, :math:`b` is the right hand side and
    :math:`\alpha` and :math:`\beta` are scalars, or arrays of scalars for
    a multidimensional problem.

    The user must provide mass and stiffness matrices and scale arrays
    :math:`(\alpha/\beta)` to each matrix. The matrices and scales can be
    provided as either kwargs or args

    As 4 arguments

    Parameters
    ----------
        A : SpectralMatrix
            Stiffness matrix (Dirichlet or Neumann)
        B : SpectralMatrix
            Mass matrix (Dirichlet or Neumann)
        alfa : Numpy array
        beta : Numpy array

    or as a dict with keys

    Parameters
    ----------
        ADDmat : A
                 Stiffness matrix (Dirichlet basis)
        BDDmat : B
                 Mass matrix (Dirichlet basis)
        ANNmat : A
                  Stiffness matrix (Neumann basis)
        BNNmat : B
                 Mass matrix (Neumann basis)

    where :math:`\alpha` and :math:`\beta` are avalable as A.scale and B.scale.

    The solver can be used along any axis of a multidimensional problem. For
    example, if the Legendre basis (Dirichlet or Neumann) is the last in a
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


    See https://rawgit.com/spectralDNS/shenfun/master/docs/src/Poisson3D/poisson3d_bootstrap.html
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

    This is where the scale arrays in the signature to the Helmholt solver comes
    from. :math:`\alpha` is here :math:`1`, whereas :math:`\beta` is
    :math:`(k^2+l^2)`. Note that :math:`k+l` is an array of shape (N, M, 1).

    """

    def __init__(self, *args, **kwargs):

        if 'ADDmat' in kwargs or 'ANNmat' in kwargs:
            if 'ADDmat' in kwargs:
                assert 'BDDmat' in kwargs
                A = self.A = kwargs['ADDmat']
                B = self.B = kwargs['BDDmat']

            if 'ANNmat' in kwargs:
                assert 'BNNmat' in kwargs
                A = self.A = kwargs['ANNmat']
                B = self.B = kwargs['BNNmat']
            A_scale = self.A_scale = A.scale
            B_scale = self.B_scale = B.scale

        elif len(args) == 4:
            A = self.A = args[0]
            B = self.B = args[1]
            A_scale = self.A_scale = args[2]
            B_scale = self.B_scale = args[3]

        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        v = A.testfunction[0]
        neumann = self.neumann = v.boundary_condition() == 'Neumann'
        if not neumann:
            self.bc = v.bc
            self.scaled = v.is_scaled()

        self.axis = A.axis
        shape = [1]
        T = A.tensorproductspace
        if T is not None:
            shape = list(T.local_shape(True))
            shape[A.axis] = 1

        if np.ndim(B_scale) > 1:
            if len(shape) == 2:
                if neumann and B_scale[0, 0] == 0:
                    B_scale[0, 0] = 1.

            elif len(shape) == 3:
                if neumann and B_scale[0, 0, 0] == 0:
                    B_scale[0, 0, 0] = 1.

            A[0] = np.atleast_1d(A[0])
            if A[0].shape[0] == 1:
                A[0] = np.ones(A.shape[0])*A[0]
            A0 = v.broadcast_to_ndims(A[0])
            B0 = v.broadcast_to_ndims(B[0])
            B2 = v.broadcast_to_ndims(B[2])
            shape[A.axis] = v.N
            self.d0 = np.zeros(shape)
            self.d1 = np.zeros(shape)
            ss = [slice(None)]*self.d0.ndim
            ss[self.axis] = slice(0, A.shape[0])
            self.d0[tuple(ss)] = A0*A_scale + B0*B_scale
            ss[self.axis] = slice(0, A.shape[0]-2)
            self.d1[tuple(ss)] = B2*B_scale
            self.L = np.zeros_like(self.d0)
            self.TDMA_SymLU_VC(self.d0, self.d1, self.L, self.axis)

        else:
            self.d0 = A[0]*A_scale + B[0]*B_scale
            self.d1 = B[2]*B_scale
            self.L = np.zeros_like(self.d1)
            self.bc = A.testfunction[0].bc
            self.axis = 0
            self.TDMA_SymLU(self.d0, self.d1, self.L)

    @staticmethod
    @optimizer
    def TDMA_SymLU_VC(d0, d1, L, axis=0):
        pass

    @staticmethod
    @optimizer
    def TDMA_SymSolve_VC(d, a, l, x, axis=0):
        pass

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

    def __call__(self, u, b):
        u[:] = b

        self.TDMA_SymSolve_VC(self.d0, self.d1, self.L, u, self.axis)

        if not self.neumann:
            self.bc.apply_after(u, True)

        return u

    def matvec(self, v, c):
        c[:] = 0
        c1 = np.zeros_like(c)
        c1 = self.A.matvec(v, c1, axis=self.axis)
        c = self.B.matvec(v, c, axis=self.axis)
        c += c1
        return c

class Biharmonic(object):
    r"""Multidimensional Biharmonic solver for

    .. math::

        a_0 u'''' + \alpha u'' + \beta u = b

    where :math:`u` is the solution, :math:`b` is the right hand side and
    :math:`a_0, \alpha` and :math:`\beta` are scalars, or arrays of scalars for
    a multidimensional problem.

    The user must provide mass, stiffness and biharmonic matrices and scale
    arrays :math:`(a_0/\alpha/\beta)`. The matrices and scales can be provided
    as either kwargs or args

    As 6 arguments

    Parameters
    ----------
        S : SpectralMatrix
            Biharmonic matrix
        A : SpectralMatrix
            Stiffness matrix
        B : SpectralMatrix
            Mass matrix
        a0 : array
        alfa : array
        beta : array

    or as dict with key/values

    Parameters
    ----------
        SBBmat : S
                 Biharmonic matrix
        ABBmat : A
                 Stiffness matrix
        BBBmat : B
                 Mass matrix

    where a0, alfa and beta must be avalable as S.scale, A.scale, B.scale.

    The solver can be used along any axis of a multidimensional problem. For
    example, if the Chebyshev basis (Biharmonic) is the last in a
    3-dimensional TensorProductSpace, where the first two dimensions use
    Fourier, then the 1D equation listed above arises when one is solving the
    3D biharmonic equation

    .. math::

        \nabla^4 u = b

    With the spectral Galerkin method we multiply this equation with a test
    function (:math:`v`) and integrate (weighted inner product
    :math:`(\cdot, \cdot)_w`) over the domain

    .. math::

        (v, \nabla^4 u)_w = (v, b)_w

    See https://rawgit.com/spectralDNS/shenfun/master/docs/._shenfun_bootstrap004.html#sec:tensorproductspaces
    for details, since it is actually quite involved. But basically, one obtains
    a linear algebra system to be solved along the z-axis for all combinations
    of the two Fourier indices k and l

    .. math::

        ((2\pi)^2 S_{mj} - 2(k^2 + l^2) A_{mj}) + (k^2 + l^2)^2 B_{mj}) \hat{u}[k, l, j] = (v, b)_w[k, l, m]

    Note that :math:`k` only varies along :math:`x`-direction, whereas :math:`l`
    varies along :math:`y`. To allow for Numpy broadcasting these two variables
    are stored as arrays of shape

    .. math::

        k : (N, 1, 1)

        l : (1, M, 1)

    Here it is assumed that the solution array :math:`\hat{u}` has shape
    (N, M, P). Now, multiplying :math:`k` array with :math:`\hat{u}` is
    achieved as

    .. math::

        k \cdot \hat{u}

    Numpy will then take care of broadcasting :math:`k` to an array of shape
    (N, M, P) before performing the elementwise multiplication. Likewise, the
    constant scale :math:`(2\pi)^2` in front of the :math:`A_{mj}` matrix is
    stored with shape (1, 1, 1), and multiplying with :math:`\hat{u}` is
    performed as if it was a scalar (as it here happens to be).

    This is where the scale arrays in the signature to the Helmholt solver comes
    from. :math:`a_0` is here :math:`(2\pi)^2`, whereas :math:`\alpha` and
    :math:`\beta` are :math:`-2(k^2+l^2)` and :math:`(k^2+l^2)^2`, respectively.
    Note that :math:`k+l` is an array of shape (N, M, 1).
    """

    def __init__(self, *args, **kwargs):

        if 'SBBmat' in kwargs:
            assert 'PBBmat' in kwargs and 'BBBmat' in kwargs
            S = self.S = kwargs['SBBmat']
            A = self.A = kwargs['PBBmat']
            B = self.B = kwargs['BBBmat']
            S_scale = S.scale
            A_scale = A.scale
            B_scale = B.scale

        elif len(args) == 6:
            S = self.S = args[0]
            A = self.A = args[1]
            B = self.B = args[2]
            S_scale = args[3]
            A_scale = args[4]
            B_scale = args[5]
        else:
            raise RuntimeError('Wrong input to Biharmonic solver')

        if np.ndim(B_scale) > 1:
            shape = list(B_scale.shape)
            self.axis = S.axis
            v = S.testfunction[0]
            shape[S.axis] = v.N
            self.d0 = np.zeros(shape)
            self.d1 = np.zeros(shape)
            self.d2 = np.zeros(shape)
            S0 = v.broadcast_to_ndims(S[0])
            A0 = v.broadcast_to_ndims(A[0])
            B0 = v.broadcast_to_ndims(B[0])
            A2 = v.broadcast_to_ndims(A[2])
            B2 = v.broadcast_to_ndims(B[2])
            B4 = v.broadcast_to_ndims(B[4])
            ss = [slice(None)]*self.d0.ndim
            ss[S.axis] = slice(0, A[0].shape[0])
            self.d0[tuple(ss)] = S0*S_scale + A0*A_scale + B0*B_scale
            ss[S.axis] = slice(0, A[2].shape[0])
            self.d1[tuple(ss)] = A2*A_scale + B2*B_scale
            ss[S.axis] = slice(0, B[4].shape[0])
            self.d2[tuple(ss)] = B4*B_scale
            self.PDMA_SymLU_VC(self.d0, self.d1, self.d2, S.axis)

        else:
            self.d0 = S[0]*S_scale + A[0]*A_scale + B[0]*B_scale
            self.d1 = A[2]*A_scale + B[2]*B_scale
            self.d2 = B[4]*B_scale
            self.axis = 0
            la.PDMA_SymLU(self.d0, self.d1, self.d2)

    @staticmethod
    @optimizer
    def PDMA_SymLU_VC(d0, d1, d2, axis=0):
        raise NotImplementedError("Use Cython or Numba")

    @staticmethod
    @optimizer
    def PDMA_SymSolve_VC(d0, d1, d2, u, axis=0):
        raise NotImplementedError("Use Cython or Numba")

    def __call__(self, u, b):
        u[:] = b
        self.PDMA_SymSolve_VC(self.d0, self.d1, self.d2, u, self.axis)
        return u

    def matvec(self, v, c):
        c[:] = 0
        c1 = np.zeros_like(c)
        c1 = self.S.matvec(v, c1, axis=self.axis)
        c += c1
        c1[:] = 0
        c1 = self.A.matvec(v, c1, axis=self.axis)
        c += c1
        c1[:] = 0
        c1 = self.B.matvec(v, c1, axis=self.axis)
        c += c1
        return c

class Helmholtz_2dirichlet(object):
    """Helmholtz solver for 2-dimensional problems with 2 Dirichlet bases.

    .. math::

        a_0 BUB + a_1 AUB + a_2 BUA^T = F

    Somewhat experimental.

    """

    def __init__(self, T, kwargs):

        self.T = T
        self.V = np.zeros(0)
        assert len(kwargs) == 3

        # There are three terms, BUB, AUB and BUA
        # Extract A and B
        scale = {}
        for tmp in kwargs:
            pmat = tmp.pmat
            if pmat[0].get_key() == 'BDDmat' and pmat[1].get_key() == 'BDDmat':
                B = pmat[0]
                B1 = pmat[1]
                scale['BUB'] = tmp.scale

            elif pmat[0].get_key() == 'ADDmat' and pmat[1].get_key() == 'BDDmat':
                A = pmat[0]
                scale['AUB'] = tmp.scale

            else:
                A1 = pmat[1]
                scale['BUA'] = tmp.scale

        # Create transfer object to realign data in second direction
        pencilA = T.forward.output_pencil
        pencilB = pencilA.pencil(1)
        self.pencilB = pencilB
        self.transAB = pencilA.transfer(pencilB, 'd')
        self.u_B = np.zeros(self.transAB.subshapeB)
        self.rhs_A = np.zeros(self.transAB.subshapeA)
        self.rhs_B = np.zeros(self.transAB.subshapeB)

        self.A = A
        self.B = B
        self.A1 = A1
        self.B1 = B1
        self.scale = scale

        self.lmbda = None
        self.lmbdax = None
        self.lmbday = None
        self.Vx = None
        self.Vy = None

    def solve_eigen_problem(self, A, B, solver):
        """Solve the eigen problem"""
        N = A.testfunction[0].N
        s = A.testfunction[0].slice()
        self.V = np.zeros((N, N))
        self.lmbda = np.ones(N)
        if solver == 0:
            self.lmbda[s], self.V[s, s] = scipy_la.eigh(A.diags().toarray(),
                                                        B.diags().toarray())

        elif solver == 1:
            #self.lmbda[s], self.V[s, s] = scipy_la.eigh(B.diags().toarray())
            a = np.zeros((3, N-2))
            a[0, :] = B[0]
            a[2, :-2] = B[2]
            self.lmbda[s], self.V[s, s] = scipy_la.eig_banded(a, lower=True)

    def __call__(self, u, b, solver=1):

        if solver == 0: # pragma: no cover

            if len(self.V) == 0:
                self.solve_eigen_problem(self.A, self.B, solver)
                self.Vx = self.V
                self.lmbdax = self.lmbda
                if not self.A.testfunction[0].N == self.A1.testfunction[0].N:
                    self.Vx = self.V.copy()
                    self.lmbdax = self.lmbda.copy()
                    self.solve_eigen_problem(self.A1, self.B1, solver)
                    self.Vy = self.V
                    self.lmbday = self.lmbda
                else:
                    self.Vy = self.Vx
                    self.lmbday = self.lmbdax

            # Map the right hand side to eigen space
            u[:] = (self.Vx.T).dot(b)
            self.transAB.forward(u, self.u_B)
            self.u_B[:] = self.u_B.dot(self.Vy)
            self.transAB.backward(self.u_B, u)

            # Apply the inverse in eigen space
            ls = self.T.local_slice()
            u /= (self.scale['BUB'] + self.lmbdax[:, np.newaxis] + self.lmbday[np.newaxis, :])[ls]

            # Map back to physical space
            u[:] = self.Vx.dot(u)
            self.transAB.forward(u, self.u_B)
            self.u_B[:] = self.u_B.dot(self.Vy.T)
            self.transAB.backward(self.u_B, u)

        if solver == 1:

            assert self.A.testfunction[0].is_scaled()

            if len(self.V) == 0:
                self.solve_eigen_problem(self.A, self.B, solver)

            ls = [slice(start, start+shape) for start, shape in zip(self.pencilB.substart,
                                                                    self.pencilB.subshape)]

            self.B1.scale = np.zeros((ls[0].stop-ls[0].start, 1))
            self.B1.scale[:, 0] = self.scale['BUB'] + 1./self.lmbda[ls[0]]
            self.A1.scale = np.ones((1, 1))
            # Create Helmholtz solver along axis=1
            Helmy = Helmholtz(**{'ADDmat': self.A1, 'BDDmat': self.B1})

            # Map the right hand side to eigen space
            self.rhs_A = (self.V.T).dot(b)
            self.rhs_A /= self.lmbda[:, np.newaxis]
            self.transAB.forward(self.rhs_A, self.rhs_B)
            self.u_B = Helmy(self.u_B, self.rhs_B)
            self.transAB.backward(self.u_B, u)
            u[:] = self.V.dot(u)

        elif solver == 2: # pragma: no cover
            N = self.A.testfunction[0].N
            s = self.A.testfunction[0].slice()
            AA = np.zeros((N, N))
            BB = np.zeros((N, N))
            G = np.zeros((N, N))
            H = np.zeros((N, N))

            BB[s, s] = self.B.diags().toarray()
            AA[s, s] = self.A.diags().toarray()
            G[:] = BB.dot(u)
            H[:] = u.dot(BB)
            bc = b.copy()
            B_scale = copy(self.B.scale)
            self.B.scale = np.broadcast_to(self.B.scale, (1, u.shape[1])).copy()
            self.B.scale *= self.scale['BUB']
            self.A.scale = np.ones((1, 1))
            Helmx = Helmholtz(**{'ADDmat': self.A, 'BDDmat': self.B})
            converged = False
            G_old = G.copy()
            Hc = H.copy()
            num_iter = 0
            # Solve with successive overrelaxation
            Gc = G.T.copy()
            omega = 1.6
            om = 1.
            while not converged and num_iter < 1000:
                bc[:] = b - G.dot(AA.T)
                Hc = Helmx(Hc, bc)
                H[:] = om*Hc + (1-om)*H[:]
                bc[:] = b.T - (H.T).dot(AA.T)
                Gc = Helmx(Gc, bc)
                G[:] = om*Gc.T + (1-om)*G[:]
                err = np.linalg.norm(G_old-G)
                print('Error ', num_iter, err)
                num_iter += 1
                G_old[:] = G
                converged = err < 1e-8
                om = omega

            self.B.scale = B_scale
            u = self.B.solve(G, u)

        return u
