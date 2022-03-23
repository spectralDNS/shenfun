r"""
Module for function spaces of ultraspherical type.

The ultraspherical polynomial :math:`Q^{(\alpha)}_k` is here defined as

.. math::

        Q^{(\alpha)}_k = \frac{1}{P^{(\alpha,\alpha)}_k(1)} P^{(\alpha,\alpha)}_k

where :math:`P^{(\alpha,\alpha)}_k` is the regular Jacobi polynomial with two
equal parameters. The scaling with :math:`(P^{(\alpha,\alpha)}_k(1))^{-1}` is
not standard, but it leads to the boundary values

.. math::

    {Q}^{(\alpha)}_k(\pm 1) = (\pm 1)^{k}

The Chebyshev (first and second kind) and Legendre polynomials can be defined as

.. math::

    T_k(x) &= Q^{(-1/2)}_k(x) \\
    U_k(x) &= (k+1)Q^{(1/2)}_k(x) \\
    L_k(x) &= Q^{(0)}_k(x)

"""

import functools
import numpy as np
import sympy as sp
from scipy.special import eval_jacobi, roots_jacobi #, gamma
from mpi4py_fft import fftw
from shenfun.config import config
from shenfun.spectralbase import SpectralBase, Transform, islicedict, \
    slicedict, getCompositeBase, BoundaryConditions
from shenfun.matrixbase import SparseMatrix
from shenfun.jacobi.recursions import cn

xp = sp.Symbol('x', real=True)
m, n, k = sp.symbols('m,n,k', real=True, integer=True)

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

__all__ = ['Orthogonal',
           'Phi1',
           'Phi2',
           'Phi3',
           'Phi4',
           'CompactDirichlet',
           'CompactNeumann',
           'CompositeBase',
           'Generic',
           'BCBase',
           'BCGeneric']


class Orthogonal(SpectralBase):
    r"""Function space for regular (orthogonal) ultraspherical polynomials

    The orthogonal basis is

    .. math::

        Q^{(\alpha)}_k = \frac{1}{P^{(\alpha,\alpha)}_k(1)} P^{(\alpha,\alpha)}_k, \quad k = 0, 1, \ldots, N-1,

    where :math:`P^{(\alpha,\beta)}_k` is the `Jacobi polynomial <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_.
    The basis :math:`\{Q^{(\alpha)}_k\}` is orthogonal with weight :math:`(1-x^2)^{\alpha}`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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

    def __init__(self, N, quad="QG", alpha=0, domain=(-1, 1),
                 dtype=float, padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        SpectralBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.alpha = alpha
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def family():
        return 'ultraspherical'

    def reference_domain(self):
        return (-1, 1)

    def get_orthogonal(self):
        return self

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "QG"
        points, weights = roots_jacobi(N, float(self.alpha), float(self.alpha))
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    @staticmethod
    def jacobiQ(x, alpha, N):
        V = np.zeros((x.shape[0], N))
        for i in range(N):
            V[:, i] = eval_jacobi(i, float(alpha), float(alpha), x)
        return V

    def derivative_jacobiQ(self, x, alpha, k=1):
        V = self.jacobiQ(x, alpha+k, self.N)
        if k > 0:
            Vc = np.zeros_like(V)
            for j in range(k, self.N):
                dj = np.prod(np.array([j+2*alpha+1+i for i in range(k)]))
                Vc[:, j] = (dj/2**k)*V[:, j-k]
            V = Vc
        return V

    def vandermonde(self, x):
        V = self.jacobiQ(x, self.alpha, self.shape(False))
        if self.alpha != 0:
            #V *= sp.lambdify(n, cn(self.alpha, self.alpha, n))(np.arange(self.N))[None, :]
            V *= np.array([cn(self.alpha, self.alpha, n).subs(n, i) for i in np.arange(self.N)], dtype=V.dtype)[None, :]
        return V

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

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'Q'

    def sympy_basis(self, i=0, x=xp):
        return cn(self.alpha, self.alpha, i)*sp.jacobi(i, self.alpha, self.alpha, x)

    @staticmethod
    def bnd_values(k=0, alpha=0, beta=0):
        from shenfun.jacobi.recursions import bnd_values
        return bnd_values(alpha, alpha, k=k, gn=cn)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_jacobi(i, float(self.alpha), float(self.alpha), x, out=output_array)
        if self.alpha != 0:
            output_array *= cn(self.alpha, self.alpha, i).n()
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights()[0]
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)

        dj = np.prod(np.array([i+2*self.alpha+1+j for j in range(k)]))
        output_array[:] = dj/2**k*eval_jacobi(i-k, float(self.alpha+k), float(self.alpha+k), x)
        if self.alpha != 0:
            output_array[:] = output_array*cn(self.alpha, self.alpha, i).n()
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        V = self.derivative_jacobiQ(x, self.alpha, k)
        if self.alpha != 0:
            #V *= sp.lambdify(n, cn(self.alpha, self.alpha, n))(np.arange(self.N))[None, :]
            V *= np.array([cn(self.alpha, self.alpha, n).subs(n, i) for i in np.arange(self.N)], dtype=V.dtype)[None, :]
        return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        return self.vandermonde(x)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.output_array.dtype)
        x = self.map_reference_domain(x)
        P = self.vandermonde(x)
        output_array = np.dot(P, u, out=output_array)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCGeneric(self.N, bc=self.bcs, domain=self.domain, alpha=self.alpha, beta=self.alpha)
        return self._bc_basis

CompositeBase = getCompositeBase(Orthogonal)

class Phi1(CompositeBase):
    r"""Function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= \frac{(1-x^2)}{h^{(1,\alpha)}_{k+1}} \frac{dQ^{(\alpha)}_{k+1}}{dx}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \tfrac{1}{2}(Q^{(\alpha)}_0 - Q^{(\alpha)}_1)
        \phi_{N-1} &= \tfrac{1}{2}(Q^{(\alpha)}_0 + Q^{(\alpha)}_1)

    where

    .. math::

        h^{(1,\alpha)}_k&=\int_{-1}^1 \left(\frac{dQ^{(\alpha)}_{k}}{dx}\right)^2 (1-x^2)^{\alpha+1}dx, \\

    and the expansion is

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a \text{ and } u(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    bc : 2-tuple of numbers, optional
        Boundary conditions at, respectively, x=(-1, 1).
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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
    def __init__(self, N, quad="QG", bc=(0., 0.), domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)
        self._stencil_matrix = {}
        #self.b0n = sp.simplify(b(alpha, alpha, n+1, n, cn) / h(alpha, alpha, n, 0, cn))
        #self.b2n = sp.simplify(b(alpha, alpha, n+1, n+2, cn) / h(alpha, alpha, n+2, 0, cn))
        a = alpha
        self.b0n = sp.gamma(2*a + n + 2)/(2*2**(2*a)*sp.gamma(a + 1)**2*sp.gamma(n + 2))
        self.b2n = -sp.gamma(2*a + n + 2)/(2*2**(2*a)*sp.gamma(a + 1)**2*sp.gamma(n + 2))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        if N in self._stencil_matrix:
            return self._stencil_matrix[N]
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = sp.lambdify(n, self.b0n)(k[:N-2])
        d2[:] = sp.lambdify(n, self.b2n)(k[:N-2])
        self._stencil_matrix[N] = SparseMatrix({0: d0, 2: d2}, (N, N))
        return self._stencil_matrix[N]

    def slice(self):
        return slice(0, self.N-2)


class Phi2(CompositeBase):
    r"""Function space for biharmonic equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-5` are

    .. math::
        \phi_k &= \frac{(1-x^2)^2}{h^{(2,\alpha)}_{k+2}} \frac{d^2Q^{(\alpha)}_{k+2}}{dx^2},

    where

    .. math::

        h^{(2,\alpha)}_k&=\int_{-1}^1 \left(\frac{d^2Q^{(\alpha)}_{k}}{dx^2}\right)^2 (1-x^2)^{\alpha+2}dx, \\
            &= \frac{2^{2 \alpha + 1} \cdot \left(2 \alpha + n + 1\right) \left(2 \alpha + n + 2\right) \Gamma^{2}\left(\alpha + n + 1\right)}{\left(2 \alpha + 2 n + 1\right) \Gamma\left(n - 1\right) \Gamma\left(2 \alpha + n + 1\right)},

    The 4 boundary basis functions are computed using :func:`.jacobi.findbasis.get_bc_basis`,
    but they are too messy to print here. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u(1)=c \text{ and } u'(1) = d.

    The last 4 basis functions are for boundary conditions and only used if
    a, b, c or d are different from 0. In one dimension :math:`\hat{u}_{N-4}=a`,
    :math:`\hat{u}_{N-3}=b`, :math:`\hat{u}_{N-2}=c` and :math:`\hat{u}_{N-1}=d`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    bc : 4-tuple of numbers, optional
        Boundary conditions.
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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

    def __init__(self, N, quad="QG", bc=(0, 0, 0, 0), domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)
        self._stencil_matrix = {}
        a = alpha
        #self.b0n = sp.simplify(matpow(b, 2, alpha, alpha, n+2, n, cn) / h(alpha, alpha, n, 0, cn))
        #self.b2n = sp.simplify(matpow(b, 2, alpha, alpha, n+2, n+2, cn) / h(alpha, alpha, n+2, 0, cn))
        #self.b4n = sp.simplify(matpow(b, 2, alpha, alpha, n+2, n+4, cn) / h(alpha, alpha, n+4, 0, cn))
        self.b0n = sp.gamma(2*a + n + 3)/(2*2**(2*a)*(2*a + 2*n + 3)*sp.gamma(a + 1)**2*sp.gamma(n + 3))
        self.b2n = -(2*a + 2*n + 5)*sp.gamma(2*a + n + 3)/(4**a*(2*a + 2*n + 3)*(2*a + 2*n + 7)*sp.gamma(a + 1)**2*sp.gamma(n + 3))
        self.b4n = sp.gamma(2*a + n + 3)/(2*2**(2*a)*(2*a + 2*n + 7)*sp.gamma(a + 1)**2*sp.gamma(n + 3))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        if N in self._stencil_matrix:
            return self._stencil_matrix[N]
        k = np.arange(N)
        d0, d2, d4 = np.zeros(N), np.zeros(N-2), np.zeros(N-4)
        d0[:-4] = sp.lambdify(n, sp.simplify(self.b0n))(k[:N-4])
        d2[:-2] = sp.lambdify(n, sp.simplify(self.b2n))(k[:N-4])
        d4[:] = sp.lambdify(n, sp.simplify(self.b4n))(k[:N-4])
        self._stencil_matrix[N] = SparseMatrix({0: d0, 2: d2, 4: d4}, (N, N))
        return self._stencil_matrix[N]

    def slice(self):
        return slice(0, self.N-4)

class Phi3(CompositeBase):
    r"""Function space for 6th order equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-7` are

    .. math::
        \phi_k &= \frac{(1-x^2)^3}{h^{(3,\alpha)}_{k+3}} \frac{d^3Q^{(\alpha)}_{k+3}}{dx^3}, \\

    where

    .. math::

        h^{(3,\alpha)}_k=\int_{-1}^1 \left(\frac{d^3Q^{(\alpha)}_{k}}{dx^3}\right)^2 (1-x^2)^{\alpha+3}dx, \\

    The 6 boundary basis functions are computed using :func:`.jacobi.findbasis.get_bc_basis`,
    but they are too messy to print here. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u(1)=ed u'(1)=e, u''(1)=f.

    The last 6 basis functions are for boundary conditions and only used if there
    are nonzero boundary conditions.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    bc : 6-tuple of numbers, optional
        Boundary conditions.
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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
    def __init__(self, N, quad="QG", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)
        self._stencil_matrix = {}
        #self.b0n = sp.simplify(matpow(b, 3, alpha, alpha, n+3, n, cn) / h(alpha, alpha, n, 0, cn))
        #self.b2n = sp.simplify(matpow(b, 3, alpha, alpha, n+3, n+2, cn) / h(alpha, alpha, n+2, 0, cn))
        #self.b4n = sp.simplify(matpow(b, 3, alpha, alpha, n+3, n+4, cn) / h(alpha, alpha, n+4, 0, cn))
        #self.b6n = sp.simplify(matpow(b, 3, alpha, alpha, n+3, n+6, cn) / h(alpha, alpha, n+6, 0, cn))
        a = alpha
        self.b0n = sp.gamma(2*a + n + 4)/(2*2**(2*a)*(2*a + 2*n + 3)*(2*a + 2*n + 5)*sp.gamma(a + 1)**2*sp.gamma(n + 4))
        self.b2n = -3*sp.gamma(2*a + n + 4)/(2*2**(2*a)*(2*a + 2*n + 3)*(2*a + 2*n + 9)*sp.gamma(a + 1)**2*sp.gamma(n + 4))
        self.b4n = 3*sp.gamma(2*a + n + 4)/(2*2**(2*a)*(2*a + 2*n + 5)*(2*a + 2*n + 11)*sp.gamma(a + 1)**2*sp.gamma(n + 4))
        self.b6n = -sp.gamma(2*a + n + 4)/(2*2**(2*a)*(2*a + 2*n + 9)*(2*a + 2*n + 11)*sp.gamma(a + 1)**2*sp.gamma(n + 4))

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'P3'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        if N in self._stencil_matrix:
            return self._stencil_matrix[N]
        k = np.arange(N)
        d0, d2, d4, d6 = np.zeros(N), np.zeros(N-2), np.zeros(N-4), np.zeros(N-6)
        d0[:-6] = sp.lambdify(n, self.b0n)(k[:N-6])
        d2[:-4] = sp.lambdify(n, self.b2n)(k[:N-6])
        d4[:-2] = sp.lambdify(n, self.b4n)(k[:N-6])
        d6[:] = sp.lambdify(n, self.b6n)(k[:N-6])
        self._stencil_matrix[N] = SparseMatrix({0: d0, 2: d2, 4: d4, 6: d6}, (N, N))
        return self._stencil_matrix[N]

    def slice(self):
        return slice(0, self.N-6)


class Phi4(CompositeBase):
    r"""Function space for 8th order equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-9` are

    .. math::
        \phi_k &= \frac{(1-x^2)^4}{h^{(4,\alpha)}_{k+4}} \frac{d^4Q^{(\alpha)}_{k+4}}{dx^4}, \\

    where

    .. math::

        h^{(4,\alpha)}_k&=\int_{-1}^1 \left(\frac{d^4Q^{(\alpha)}_{k}}{dx^4}\right)^2 (1-x^2)^{\alpha+4}dx, \\
            &=\frac{2^{2 \alpha + 1} \cdot \left(2 \alpha + n + 1\right) \left(2 \alpha + n + 2\right) \left(2 \alpha + n + 3\right) \left(2 \alpha + n + 4\right) \Gamma^{2}\left(\alpha + n + 1\right)}{\left(2 \alpha + 2 n + 1\right) \Gamma\left(n - 3\right) \Gamma\left(2 \alpha + n + 1\right)},

    The 8 boundary basis functions are computed using :func:`.jacobi.findbasis.get_bc_basis`,
    but they are too messy to print here. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u''''(-1)=d, u(1)=e, u'(1)=f, u''(1)=g, u''''(1)=h.

    The last 8 basis functions are for boundary conditions and only used if there
    are nonzero boundary conditions.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    bc : 8-tuple of numbers, optional
        Boundary conditions.
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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
    def __init__(self, N, quad="QG", bc=(0,)*8, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)
        self._stencil_matrix = {}
        #self.b0n = sp.simplify(matpow(b, 4, alpha, alpha, n+4, n, cn) / h(alpha, alpha, n, 0, cn))
        #self.b2n = sp.simplify(matpow(b, 4, alpha, alpha, n+4, n+2, cn) / h(alpha, alpha, n+2, 0, cn))
        #self.b4n = sp.simplify(matpow(b, 4, alpha, alpha, n+4, n+4, cn) / h(alpha, alpha, n+4, 0, cn))
        #self.b6n = sp.simplify(matpow(b, 4, alpha, alpha, n+4, n+6, cn) / h(alpha, alpha, n+6, 0, cn))
        #self.b8n = sp.simplify(matpow(b, 4, alpha, alpha, n+4, n+8, cn) / h(alpha, alpha, n+8, 0, cn))
        a = alpha
        self.b0n = sp.gamma(2*a + n + 5)/(2*2**(2*a)*(2*a + 2*n + 3)*(2*a + 2*n + 5)*(2*a + 2*n + 7)*sp.gamma(a + 1)**2*sp.gamma(n + 5))
        self.b2n = -2**(1 - 2*a)*sp.gamma(2*a + n + 5)/((2*a + 2*n + 3)*(2*a + 2*n + 7)*(2*a + 2*n + 11)*sp.gamma(a + 1)**2*sp.gamma(n + 5))
        self.b4n = 3*(2*a + 2*n + 9)*sp.gamma(2*a + n + 5)/(4**a*(2*a + 2*n + 5)*(2*a + 2*n + 7)*(2*a + 2*n + 11)*(2*a + 2*n + 13)*sp.gamma(a + 1)**2*sp.gamma(n + 5))
        self.b6n = -2**(1 - 2*a)*sp.gamma(2*a + n + 5)/((2*a + 2*n + 7)*(2*a + 2*n + 11)*(2*a + 2*n + 15)*sp.gamma(a + 1)**2*sp.gamma(n + 5))
        self.b8n = sp.gamma(2*a + n + 5)/(2*2**(2*a)*(2*a + 2*n + 11)*(2*a + 2*n + 13)*(2*a + 2*n + 15)*sp.gamma(a + 1)**2*sp.gamma(n + 5))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic*2'

    @staticmethod
    def short_name():
        return 'P4'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        if N in self._stencil_matrix:
            return self._stencil_matrix[N]
        k = np.arange(N)
        d0, d2, d4, d6, d8 = np.zeros(N), np.zeros(N-2), np.zeros(N-4), np.zeros(N-6), np.zeros(N-8)
        d0[:-8] = sp.lambdify(n, self.b0n)(k[:N-8])
        d2[:-6] = sp.lambdify(n, self.b2n)(k[:N-8])
        d4[:-4] = sp.lambdify(n, self.b4n)(k[:N-8])
        d6[:-2] = sp.lambdify(n, self.b6n)(k[:N-8])
        d8[:] = sp.lambdify(n, self.b8n)(k[:N-8])
        self._stencil_matrix[N] = SparseMatrix({0: d0, 2: d2, 4: d4, 6: d6, 8: d8}, (N, N))
        return self._stencil_matrix[N]

    def slice(self):
        return slice(0, self.N-8)


class CompactDirichlet(CompositeBase):
    r"""Function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= Q^{(\alpha)}_k - Q^{(\alpha)}_{k+2} \quad k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \tfrac{1}{2}(Q^{(\alpha)}_0 - Q^{(\alpha)}_1)
        \phi_{N-1} &= \tfrac{1}{2}(Q^{(\alpha)}_0 + Q^{(\alpha)}_1)

    and the expansion is

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a \text{ and } u(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    bc : 2-tuple of numbers, optional
        Boundary conditions at, respectively, x=(-1, 1).
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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
    def __init__(self, N, quad="QG", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'QD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-2:] = 0
        return SparseMatrix({0: d, 2: -d[:-2]}, (N, N))

    def slice(self):
        return slice(0, self.N-2)


class CompactNeumann(CompositeBase):
    r"""Function space for Neumann boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
    n*(-2*a - n - 1)/(2*a*n + 4*a + n**2 + 5*n + 6)
        \phi_k &= Q^{(\alpha)}_k - \frac{k(2\alpha+k+1)}{2 \alpha k + 4k + k^2 + 5k + 6}Q^{(\alpha)}_{k+2}  \quad k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \tfrac{1}{2}Q^{(\alpha)}_1 - \frac{(\alpha+1)}{2(2\alpha+3) Q^{(\alpha)}_2}
        \phi_{N-1} &= \tfrac{1}{2}Q^{(\alpha)}_1 + \frac{(\alpha+1)}{2(2\alpha+3) Q^{(\alpha)}_2}

    and the expansion is

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u'(-1) &= a \text{ and } u'(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    bc : 2-tuple of numbers, optional
        Boundary conditions at, respectively, x=(-1, 1).
    alpha : number, optional
        Parameter of the ultraspherical polynomial
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
    def __init__(self, N, quad="QG", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0,
                 coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'QN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        a = self.alpha
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = 1
        d2[:] = sp.lambdify(n, n*(-2*a-n-1)/(2*a*n+4*a+n**2+5*n+6))(k[:N-2])
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)


class Generic(CompositeBase):
    r"""Function space for space with any boundary conditions

    Any combination of Dirichlet and Neumann is possible.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature
        - QG - Jacobi-Gauss
    bc : dict, optional
        The dictionary must have keys 'left' and 'right', to describe boundary
        conditions on the left and right boundaries, and a list of 2-tuples to
        specify the condition. Specify Dirichlet on both ends with

            {'left': {'D': a}, 'right': {'D': b}}

        for some values `a` and `b`, that will be neglected in the current
        function. Specify mixed Neumann and Dirichlet as

            {'left': {'N': a}, 'right': {'N': b}}

        For both conditions on the right do

            {'right': {'N': a, 'D': b}}

        Any combination should be possible, and it should also be possible to
        use second derivatives `N2`. See :class:`~shenfun.spectralbase.BoundaryConditions`.
    domain : 2-tuple of floats, optional
        The computational domain
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    alpha : number, optional
        Parameter of the ultraspherical polynomial.
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """
    def __init__(self, N, quad="QG", bc={}, domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None,
                 alpha=0, **kw):
        from shenfun.utilities.findbasis import get_stencil_matrix
        self._stencil = get_stencil_matrix(bc, 'ultraspherical', alpha)
        if not isinstance(bc, BoundaryConditions):
            bc = BoundaryConditions(bc, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Generic'

    @staticmethod
    def short_name():
        return 'GQ'

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
    domain : 2-tuple, optional
        The domain of the homogeneous space.
    alpha : number, optional
        Parameter of the Ultraspherical polynomial

    """

    def __init__(self, N, bc=(0, 0), domain=(-1, 1), alpha=0, **kw):
        CompositeBase.__init__(self, N, bc=bc, domain=domain, alpha=alpha)
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
        return self.jacobiQ(x, self.alpha, self.num_T)

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([cn(self.alpha, self.alpha, i)*sp.jacobi(j, self.alpha, self.alpha, x) for j in range(self.num_T)]))

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
            from shenfun.utilities.findbasis import get_bc_basis
            self._stencil_matrix = np.array(get_bc_basis(self.bcs, 'ultraspherical', self.alpha))
        return self._stencil_matrix
