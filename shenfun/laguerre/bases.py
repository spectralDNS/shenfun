import functools
import sympy as sp
import numpy as np
from numpy.polynomial import laguerre as lag
from scipy.special import eval_laguerre
from mpi4py_fft import fftw
from shenfun.matrixbase import SparseMatrix
from shenfun.spectralbase import SpectralBase, Transform, islicedict, \
    slicedict, getCompositeBase, BoundaryConditions

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

__all__ = ['Orthogonal',
           'CompactDirichlet',
           'CompactNeumann',
           'Generic',
           'CompositeBase']

xp = sp.Symbol('x', real=True)

class Orthogonal(SpectralBase):
    r"""Function space for a regular Laguerre series

    The orthogonal basis is the Laguerre function

    .. math::

        \phi_k = La_k \exp(-x/2), \quad k = 0, 1, \ldots, N-1,

    where :math:`La_k` is the :math:`k`'th Laguerre polynomial.

    Parameters
    ----------
    N : int
        Number of quadrature points
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    Note
    ----
    We are using Laguerre functions and not the regular Laguerre polynomials
    as basis functions.
    """
    def __init__(self, N, dtype=float, padding_factor=1, dealias_direct=False,
                 coordinates=None, **kw):
        SpectralBase.__init__(self, N, quad="LG", domain=(0., np.inf), dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def family():
        return 'laguerre'

    def reference_domain(self):
        return (0., np.inf)

    def domain_factor(self):
        return 1

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        if self.quad == "LG":
            points, weights = lag.laggauss(N)
            if weighted:
                weights *= np.exp(points)
        else:
            raise NotImplementedError

        return points, weights

    def vandermonde(self, x):
        V = lag.lagvander(x, int(self.N*self.padding_factor)-1)
        V *= np.exp(-x/2)[:, None]
        return V

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_laguerre(i, x, out=output_array)
        output_array *= np.exp(-x/2)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[i] = 1
        basis = lag.Laguerre(basis)
        if k == 1:
            basis = basis.deriv(k)-0.5*basis
        elif k == 2:
            basis = basis.deriv(2)-basis.deriv(1)+0.25*basis
        elif k == 3:
            basis = basis.deriv(3)-1.5*basis.deriv(2)+0.75*basis.deriv(1)-basis/8
        elif k == 4:
            basis = basis.deriv(4)-2*basis.deriv(3)+1.5*basis.deriv(2)-0.5*basis.deriv(1)+basis/16
        output_array[:] = basis(x)*np.exp(-x/2)
        return output_array

    @staticmethod
    def bnd_values(k=0, **kw):
        if k == 0:
            return (lambda i: 1, None)
        elif k == 1:
            return (lambda i: -i-sp.S.Half, None)
        raise NotImplementedError

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = lag.lagvander(x, int(self.N*self.padding_factor)-1)
        M = V.shape[1]
        if k == 1:
            D = np.zeros((M, M))
            D[:-1, :] = lag.lagder(np.eye(M), 1)
            W = np.dot(V, D)
            W -= 0.5*V
            V = W*np.exp(-x/2)[:, np.newaxis]

        elif k == 2:
            D = np.zeros((M, M))
            D[:-2, :] = lag.lagder(np.eye(M), 2)
            D[:-1, :] -= lag.lagder(np.eye(M), 1)
            W = np.dot(V, D)
            W += 0.25*V
            V = W*np.exp(-x/2)[:, np.newaxis]

        elif k == 3:
            D = np.zeros((M, M))
            D[:-3, :] = lag.lagder(np.eye(M), 3)
            D[:-2, :] -= lag.lagder(np.eye(M), 2)*3/2
            D[:-1, :] += lag.lagder(np.eye(M), 1)*3/4
            W = np.dot(V, D)
            W += V/8
            V = W*np.exp(-x/2)[:, np.newaxis]

        elif k == 4:
            D = np.zeros((M, M))
            D[:-4, :] = lag.lagder(np.eye(M), 4)
            D[:-3, :] -= lag.lagder(np.eye(M), 3)*2
            D[:-2, :] += lag.lagder(np.eye(M), 2)*3/2
            D[:-1, :] -= lag.lagder(np.eye(M), 1)/2
            W = np.dot(V, D)
            W += V/16
            V = W*np.exp(-x/2)[:, np.newaxis]

        elif k == 0:
            V *= np.exp(-x/2)[:, np.newaxis]

        else:
            raise NotImplementedError

        return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        return V

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        output_array[:] = lag.lagval(x, u)*np.exp(-x/2)
        return output_array

    def sympy_basis(self, i=0, x=sp.symbols('x')):
        return sp.laguerre(i, x)*sp.exp(-x/2)

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self):
        return self

    @staticmethod
    def short_name():
        return 'La'

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCGeneric(self.N, bc=self.bcs)
        return self._bc_basis

    def plan(self, shape, axis, dtype, options):
        if shape in (0, (0,)):
            return

        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        U = fftw.aligned(shape, dtype=dtype)
        V = fftw.aligned(shape, dtype=dtype)
        U.fill(0)
        V.fill(0)
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.scalar_product = Transform(self.scalar_product, None, U, V, trunc_array)
            self.forward = Transform(self.forward, None, U, V, trunc_array)
            self.backward = Transform(self.backward, None, trunc_array, V, U)
        else:
            self.scalar_product = Transform(self.scalar_product, None, U, V, V)
            self.forward = Transform(self.forward, None, U, V, V)
            self.backward = Transform(self.backward, None, V, V, U)

        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

CompositeBase = getCompositeBase(Orthogonal)

class CompactDirichlet(CompositeBase):
    r"""Laguerre function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= (La_k - La_{k+1})\exp(-x/2), \, k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= L_0\exp(-x/2),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(0) &= a

    The last basis function is for boundary condition and only used if a is
    different from 0. In one dimension :math:`\hat{u}_{N-1}=a`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - LG - Laguerre-Gauss
    bc : 1-tuple of number (a,)
        Boundary value at x=0
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """
    def __init__(self, N, bc=(0,), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad='LG', dtype=dtype, padding_factor=padding_factor,
                               bc=bc, dealias_direct=dealias_direct, domain=(0, np.inf),
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'SD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N)
        d[-1:] = 0
        return SparseMatrix({0: d, 1: -d[:-1]}, (N, N))

    def slice(self):
        return slice(0, self.N-1)

class CompactNeumann(CompositeBase):
    r"""Laguerre function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= (La_k - \frac{2k+1}{2k+3}La_{k+1})\exp(-x/2), \, k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= La_0\exp(-x/2),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u'(0) &= a

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - LG - Laguerre-Gauss
    bc : 1-tuple of number (a,)
        Boundary value a = u'(0)
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """
    def __init__(self, N, bc=(0,), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            assert len(bc) == 1
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {}}, domain=(0, np.inf))
        CompositeBase.__init__(self, N, dtype=dtype, quad='LG', padding_factor=padding_factor,
                               bc=bc, dealias_direct=dealias_direct, domain=(0, np.inf),
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'CN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N)
        d[-1:] = 0
        k = np.arange(N)
        return SparseMatrix({0: d, 1: -(2*k[:-1]+1)/(2*k[:-1]+3)}, (N, N))

    def slice(self):
        return slice(0, self.N-1)

class Generic(CompositeBase):
    r"""Function space for Laguerre space with any boundary conditions

    Any combination of Dirichlet and Neumann is possible.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - LG - Laguerre-Gauss

    bc : dict, optional
        The dictionary must have key 'left' (not 'right'), to describe boundary
        conditions on the left boundary. Specify Dirichlet with

            {'left': {'D': a}}

        for some value `a`, that will be neglected in the current
        function. Specify mixed Neumann and Dirichlet as

            {'left': {'D': a, 'N': b}}

        See :class:`~shenfun.spectralbase.BoundaryConditions`.
    domain : 2-tuple of floats, optional
        The computational domain
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """
    def __init__(self, N, quad="LG", bc={}, domain=(0, np.inf), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        from shenfun.utilities.findbasis import get_stencil_matrix
        self._stencil = get_stencil_matrix(bc, 'laguerre')
        if not isinstance(bc, BoundaryConditions):
            bc = BoundaryConditions(bc, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Generic'

    @staticmethod
    def short_name():
        return 'GL'

    def slice(self):
        return slice(0, self.N-self.bcs.num_bcs())

    def stencil_matrix(self, N=None):
        from shenfun.utilities.findbasis import n
        N = self.N if N is None else N
        d0 = np.ones(N, dtype=int)
        d0[-self.bcs.num_bcs():] = 0
        d = {0: d0}
        k = np.arange(N)
        for i, s in enumerate(self._stencil):
            di = sp.lambdify(n, s)(k[:-(i+1)])
            if not np.allclose(di, 0):
                if isinstance(di, np.ndarray):
                    di[(N-self.bcs.num_bcs()):] = 0
                d[i+1] = di
        return SparseMatrix(d, (N, N))


class BCBase(CompositeBase):
    """Function space for inhomogeneous boundary conditions

    Parameters
    ----------
    N : int
        Number of quadrature points in the homogeneous space.
    bc : dict
        The boundary conditions in dictionary form, see
        :class:`.BoundaryConditions`.

    """

    def __init__(self, N, bc=None, **kw):
        CompositeBase.__init__(self, N, bc=bc)
        self._stencil_matrix = None

    def stencil_matrix(self, N=None):
        raise NotImplementedError

    @staticmethod
    def short_name():
        raise NotImplementedError

    @staticmethod
    def boundary_condition():
        return 'Apply'

    def shape(self, forward_output=True):
        if forward_output:
            return self.stencil_matrix().shape[0]
        else:
            return self.N

    @property
    def num_T(self):
        return self.stencil_matrix().shape[1]

    def slice(self):
        return slice(self.N-self.shape(), self.N)

    def vandermonde(self, x):
        V = lag.lagvander(x, self.num_T-1)
        V *= np.exp(-x/2)[:, None]
        return V

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([sp.laguerre(j, x)*sp.exp(-x/2) for j in range(self.num_T)]))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        V = self.vandermonde(x)
        output_array[:] = np.dot(V, self.stencil_matrix()[i])
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        output_array = SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)
        return output_array

    def to_ortho(self, input_array, output_array=None):
        from shenfun import Function
        T = self.get_orthogonal()
        if output_array is None:
            output_array = Function(T)
        else:
            output_array.fill(0)
        M = self.stencil_matrix().T
        for k, row in enumerate(M):
            output_array[k] = np.dot(row, input_array)
        return output_array

    def eval(self, x, u, output_array=None):
        v = self.to_ortho(u)
        output_array = v.eval(x, output_array=output_array)
        return output_array

class BCGeneric(BCBase):

    @staticmethod
    def short_name():
        return 'BG'

    def stencil_matrix(self, N=None):
        if self._stencil_matrix is None:
            from shenfun.utilities import get_bc_basis
            self._stencil_matrix = np.array(get_bc_basis(self.bcs, 'laguerre'))
        return self._stencil_matrix
