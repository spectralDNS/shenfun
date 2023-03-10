"""
Module for function spaces of generalized Jacobi type

"""

import numpy as np
import sympy as sp
from scipy.special import eval_jacobi, roots_jacobi #, gamma
from shenfun.config import config
from shenfun.spectralbase import SpectralBase, getCompositeBase, getBCGeneric, \
    BoundaryConditions
from shenfun.matrixbase import SparseMatrix
from .recursions import h, n

xp = sp.Symbol('x', real=True)
m, n, k = sp.symbols('m,n,k', real=True, integer=True)

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

bases = ['Orthogonal',
         'CompactDirichlet',
         'CompactNeumann',
         'UpperDirichlet',
         'LowerDirichlet',
         'Generic']
bcbases = ['BCGeneric']
testbases = ['Phi1', 'Phi2', 'Phi3', 'Phi4']
__all__ = bases + bcbases + testbases + ['JacobiBase']


class JacobiBase(SpectralBase):
    r"""Abstract base class for Jacobi function spaces
    """
    def __init__(self, N, quad='', alpha=0, beta=0, padding_factor=1, domain=(-1., 1.), dtype=None,
                 dealias_direct=False, coordinates=None):
        SpectralBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.alpha = alpha  # Jacobi parameter
        self.beta = beta    # Jacobi parameter
        self.gn = 1         # Jacobi scaling function

    @property
    def is_jacobi(self):
        return True

    def L2_norm_sq(self, i):
        return h(self.alpha, self.beta, i, 0, gn=self.gn)

    def l2_norm_sq(self, i=None):
        if i is None:
            return sp.lambdify(n, h(self.alpha, self.beta, n, 0, gn=self.gn))(np.arange(self.N))
        return h(self.alpha, self.beta, i, 0, gn=self.gn)

    @staticmethod
    def bnd_values(k=0, alpha=None, beta=None, gn=None):
        from shenfun.jacobi.recursions import bnd_values
        return bnd_values(alpha, beta, k=k, gn=gn)

    def reference_domain(self):
        return (-1, 1)

    def unweighted_quadrature_weights(self):
        r"""Return quadrature weights for unweighted integrals

        .. math::

            \int_{-1}^{1} f dx \approx \sum_{j=0}^{N} f(x_j) w_j,

        where `w_j` are the quadrature weights.

        """
        xj, wj = self.points_and_weights()
        h = self.l2_norm_sq()
        uj = np.zeros(len(xj))
        b = self.vandermonde(xj)
        for j in range(len(xj)):
            for i in range(self.N):
                if i == 0:
                    fi = 2
                else:
                    fl = self.orthogonal_basis_function(i=i).as_poly().integrate()
                    fi = fl(1)-fl(-1)
                uj[j] += (b[j, i] / h[i]) * fi
            uj[j] *= wj[j]
        return uj

    def lagrange_poly(self, i):
        """Return i'th Lagrange polynomial of self

        Parameters
        ----------
        i : int
            For the i'th Lagrange polynomial

        """
        xj, wj = self.points_and_weights()
        h = self.l2_norm_sq()
        b = self.vandermonde(xj)
        l = sp.S(0)
        for j in range(self.N):
            if j == 0:
                fl = 1
            else:
                fl = self.orthogonal_basis_function(i=j).as_poly()
            l += (b[i, j] / h[j]) * fl
        l *= wj[i]
        return l

    def get_bc_space(self):
        if self._bc_space:
            return self._bc_space
        self._bc_space = BCGeneric(self.N, bc=self.bcs, domain=self.domain, alpha=self.alpha, beta=self.alpha)
        return self._bc_space

class Orthogonal(JacobiBase):
    r"""Function space for regular (orthogonal) Jacobi functions

    The orthogonal basis is

    .. math::

        \phi_k = P^{(\alpha,\beta)}_k, \quad k = 0, 1, \ldots, N-1,

    where :math:`P^{(\alpha,\beta)}_k` is the `Jacobi polynomial <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_.
    The basis is orthogonal with weight :math:`(1-x)^{\alpha}(1+x)^{\beta}`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - JG - Jacobi-Gauss
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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

    def __init__(self, N, quad="JG", alpha=0, beta=0, domain=(-1, 1),
                 dtype=float, padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        JacobiBase.__init__(self, N, quad=quad, alpha=alpha, beta=beta, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def family():
        return 'jacobi'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        return SparseMatrix({0: 1}, (N, N))

    def get_orthogonal(self, **kwargs):
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 alpha=self.alpha,
                 beta=self.beta,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return Orthogonal(self.N, **d)

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, self.alpha, self.beta)
        if weighted is False:
            weights = self.unweighted_quadrature_weights()

        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def jacobi(self, x, alpha, beta, N):
        mode = config['bases']['jacobi']['mode']
        V = np.zeros((x.shape[0], N))
        if mode == 'numpy':
            for n in range(N):
                V[:, n] = eval_jacobi(n, alpha, beta, x)
        else:
            for n in range(N):
                V[:, n] = sp.lambdify(xp, sp.jacobi(n, alpha, beta, xp), 'mpmath')(x)
        return V

    def derivative_jacobi(self, x, alpha, beta, k=1):
        V = self.jacobi(x, alpha+k, beta+k, self.N)
        if k > 0:
            Vc = np.zeros_like(V)
            for j in range(k, self.N):
                dj = np.prod(np.array([j+alpha+beta+1+i for i in range(k)]))
                #dj = gamma(j+alpha+beta+1+k) / gamma(j+alpha+beta+1)
                Vc[:, j] = (dj/2**k)*V[:, j-k]
            V = Vc
        return V

    def vandermonde(self, x):
        return self.jacobi(x, self.alpha, self.beta, self.shape(False))

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'J'

    def orthogonal_basis_function(self, i=0, x=xp):
        return sp.jacobi(i, self.alpha, self.beta, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        mode = config['bases']['jacobi']['mode']
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array = eval_jacobi(i, self.alpha, self.beta, x, out=output_array)
        else:
            f = self.orthogonal_basis_function(i, xp)
            output_array[:] = sp.lambdify(xp, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        mode = config['bases']['jacobi']['mode']
        if x is None:
            x = self.points_and_weights(mode=mode)[0]
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)

        if mode == 'numpy':
            dj = np.prod(np.array([i+self.alpha+self.beta+1+j for j in range(k)]))
            output_array[:] = dj/2**k*eval_jacobi(i-k, self.alpha+k, self.beta+k, x)
        else:
            f = sp.jacobi(i, self.alpha, self.beta, xp)
            output_array[:] = sp.lambdify(xp, f.diff(xp, k), 'mpmath')(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        mode = config['bases']['jacobi']['mode']
        if x is None:
            x = self.points_and_weights()[0]
        if mode == 'numpy':
            return self.derivative_jacobi(x, self.alpha, self.beta, k)
        else:
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N):
                f = sp.jacobi(i, self.alpha, self.beta, xp)
                V[:, i] = sp.lambdify(xp, f.diff(xp, k), 'mpmath')(x)
        return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.points_and_weights()[0]
        return self.vandermonde(x)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.output_array.dtype)
        x = self.map_reference_domain(x)
        P = self.vandermonde(x)
        output_array = np.dot(P, u, out=output_array)
        return output_array

    #def get_bc_space(self):
    #    if self._bc_space:
    #        return self._bc_space
    #    self._bc_space = BCGeneric(self.N, bc=self.bcs, domain=self.domain, alpha=self.alpha, beta=self.beta)
    #    return self._bc_space

CompositeBase = getCompositeBase(Orthogonal)
BCGeneric = getBCGeneric(CompositeBase)

class Phi1(CompositeBase):
    r"""Function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= \frac{(1-x^2)}{h^{(1,\alpha,\beta)}_{k+1}} \frac{dP^{(\alpha,\beta)}_{k+1}}{dx}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{\alpha + 1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_0  - \frac{1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_1, \\
        \phi_{N-1} &= \frac{\beta + 1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_0+ \frac{1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_1,

    where

    .. math::

        h^{(1,\alpha,\beta)}_k&=\int_{-1}^1 \left(\frac{dP^{(\alpha,\beta)}_{k}}{dx}\right)^2 (1-x)^{\alpha+1}(1+x)^{\beta+1}dx, \\
            &= \frac{2^{\alpha+\beta+1}(\alpha+\beta+k+1) \Gamma{(\alpha+k+1)} \Gamma{(\beta+k+1)}}{(\alpha+\beta+2k+1) \Gamma{(k)} \Gamma{(\alpha+\beta+k+1)}},

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

        - JG - Jacobi-Gauss
    bc : 2-tuple of numbers, optional
        Boundary conditions at, respectively, x=(-1, 1).
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(b(alpha, beta, n+1, n) / h(alpha, beta, n, 0)),
        #   2: sp.simplify(b(alpha, beta, n+1, n+1) / h(alpha, beta, n+1, 0)),
        #   4: sp.simplify(b(alpha, beta, n+1, n+2) / h(alpha, beta, n+2, 0))}
        a, b = alpha, beta
        self._stencil = {
            0: 2**(-a-b)*sp.gamma(n+1)*sp.gamma(a+b+n+2)/((a+b+2*n+2)*sp.gamma(a+n+1)*sp.gamma(b+n+1)),
            2: -2**(-a-b)*sp.gamma(n+3)*sp.gamma(a+b+n+2)/((a+b+2*n+4)*sp.gamma(a+n+2)*sp.gamma(b+n+2))
        }
        if alpha != beta:
            self._stencil[1] = 2**(-a-b)*(a-b)*(a+b+2*n+3)*sp.gamma(n+2)*sp.gamma(a+b+n+2)/((a+b+2*n+2)*(a+b+2*n+4)*sp.gamma(a+n+2)*sp.gamma(b+n+2))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'


class Phi2(CompositeBase):
    r"""Function space for biharmonic equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-5` are

    .. math::
        \phi_k &= \frac{(1-x^2)^2}{h^{(2,\alpha,\beta)}_{k+2}} \frac{d^2P^{(\alpha,\beta)}_{k+2}}{dx^2},

    where

    .. math::

        h^{(2,\alpha,\beta)}_k&=\int_{-1}^1 \left(\frac{d^2P^{(\alpha,\beta)}_{k}}{dx^2}\right)^2 (1-x)^{\alpha+2}(1+x)^{\beta+2}dx, \\
            &= \frac{2^{\alpha+\beta+1}(\alpha+\beta+k+1) \Gamma{(\alpha+k+1)} \Gamma{(\beta+k+1)}}{(\alpha+\beta+2k+1) \Gamma{(k)} \Gamma{(\alpha+\beta+k+1)}},

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

        - JG - Jacobi-Gauss
    bc : 4-tuple of numbers, optional
        Boundary conditions.
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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

    def __init__(self, N, quad="JG", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        a, b = alpha, beta
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 2, alpha, beta, n+2, n) / h(alpha, beta, n, 0)),
        #   2: sp.simplify(matpow(b, 2, alpha, beta, n+2, n+1) / h(alpha, beta, n+1, 0)),
        #   4: sp.simplify(matpow(b, 2, alpha, beta, n+2, n+2) / h(alpha, beta, n+2, 0)),
        #   6: sp.simplify(matpow(b, 2, alpha, beta, n+2, n+3) / h(alpha, beta, n+3, 0)),
        #   8: sp.simplify(matpow(b, 2, alpha, beta, n+2, n+4) / h(alpha, beta, n+4, 0))}
        self._stencil = {
            0: 2**(-a - b + 1)*sp.gamma(n + 1)*sp.gamma(a + b + n + 3)/((a + b + 2*n + 2)*(a + b + 2*n + 3)*(a + b + 2*n + 4)*sp.gamma(a + n + 1)*sp.gamma(b + n + 1)),
            2: 2**(-a - b + 1)*(a + b + 2*n + 5)*(a**2 - 4*a*b - 2*a*n - 5*a + b**2 - 2*b*n - 5*b - 2*n**2 - 10*n - 12)*sp.gamma(n + 3)*sp.gamma(a + b + n + 3)/((a + b + 2*n + 3)*(a + b + 2*n + 4)*(a + b + 2*n + 6)*(a + b + 2*n + 7)*sp.gamma(a + n + 3)*sp.gamma(b + n + 3)),
            4: 2**(-a - b + 1)*sp.gamma(n + 5)*sp.gamma(a + b + n + 3)/((a + b + 2*n + 6)*(a + b + 2*n + 7)*(a + b + 2*n + 8)*sp.gamma(a + n + 3)*sp.gamma(b + n + 3))
        }
        if self.alpha != self.beta:
            self._stencil[1] = 2**(-a - b + 2)*(a - b)*sp.gamma(n + 2)*sp.gamma(a + b + n + 3)/((a + b + 2*n + 2)*(a + b + 2*n + 4)*(a + b + 2*n + 6)*sp.gamma(a + n + 2)*sp.gamma(b + n + 2))
            self._stencil[3] = -2**(-a - b + 2)*(a - b)*sp.gamma(n + 4)*sp.gamma(a + b + n + 3)/((a + b + 2*n + 4)*(a + b + 2*n + 6)*(a + b + 2*n + 8)*sp.gamma(a + n + 3)*sp.gamma(b + n + 3))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'

class Phi3(CompositeBase):
    r"""Function space for 6th order equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-7` are

    .. math::
        \phi_k &= \frac{(1-x^2)^3}{h^{(3,\alpha,\beta)}_{k+3}} \frac{d^3P^{(\alpha,\beta)}_{k+3}}{dx^3}, \\

    where

    .. math::

        h^{(3,\alpha,\beta)}_k&=\int_{-1}^1 \left(\frac{d^3P^{(\alpha,\beta)}_{k}}{dx^3}\right)^2 (1-x)^{\alpha+3}(1+x)^{\beta+3}dx, \\

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

        - JG - Jacobi-Gauss
    bc : 6-tuple of numbers, optional
        Boundary conditions.
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n) / h(self.alpha, self.beta, n, 0)),
        #   2: sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n+2) / h(self.alpha, self.beta, n+2, 0)),
        #   4: sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n+4) / h(self.alpha, self.beta, n+4, 0)),
        #   6: sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n+6) / h(self.alpha, self.beta, n+6, 0))}
        a, b = alpha, beta
        self._stencil = {
            0: 2**(-a - b + 2)*sp.gamma(n + 1)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 2)*(a + b + 2*n + 3)*(a + b + 2*n + 4)*(a + b + 2*n + 5)*(a + b + 2*n + 6)*sp.gamma(a + n + 1)*sp.gamma(b + n + 1)),
            2: 3*2**(-a - b + 2)*(a**2 - 3*a*b - a*n - 3*a + b**2 - b*n - 3*b - n**2 - 6*n - 8)*sp.gamma(n + 3)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 3)*(a + b + 2*n + 4)*(a + b + 2*n + 6)*(a + b + 2*n + 8)*(a + b + 2*n + 9)*sp.gamma(a + n + 3)*sp.gamma(b + n + 3)),
            4: 3*2**(-a - b + 2)*(-a**2 + 3*a*b + a*n + 4*a - b**2 + b*n + 4*b + n**2 + 8*n + 15)*sp.gamma(n + 5)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*sp.gamma(a + n + 4)*sp.gamma(b + n + 4)),
            6: -2**(-a - b + 2)*sp.gamma(n + 7)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 8)*(a + b + 2*n + 9)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*(a + b + 2*n + 12)*sp.gamma(a + n + 4)*sp.gamma(b + n + 4))
        }
        if alpha != beta:
            #self._stencil[1] = sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n+1) / h(self.alpha, self.beta, n+1, 0))
            #self._stencil[3] = sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n+3) / h(self.alpha, self.beta, n+3, 0))
            #self._stencil[5] = sp.simplify(matpow(b, 3, self.alpha, self.beta, n+3, n+5) / h(self.alpha, self.beta, n+5, 0))
            self._stencil[1] = 3*2**(-a - b + 2)*(a - b)*sp.gamma(n + 2)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 2)*(a + b + 2*n + 4)*(a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 8)*sp.gamma(a + n + 2)*sp.gamma(b + n + 2))
            self._stencil[3] = 2**(-a - b + 2)*(a - b)*(a + b + 2*n + 7)*(a**2 - 8*a*b - 6*a*n - 21*a + b**2 - 6*b*n - 21*b - 6*n**2 - 42*n - 70)*sp.gamma(n + 4)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 4)*(a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 8)*(a + b + 2*n + 9)*(a + b + 2*n + 10)*sp.gamma(a + n + 4)*sp.gamma(b + n + 4))
            self._stencil[5] = 3*2**(-a - b + 2)*(a - b)*sp.gamma(n + 6)*sp.gamma(a + b + n + 4)/((a + b + 2*n + 6)*(a + b + 2*n + 8)*(a + b + 2*n + 9)*(a + b + 2*n + 10)*(a + b + 2*n + 12)*sp.gamma(a + n + 4)*sp.gamma(b + n + 4))

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'P3'


class Phi4(CompositeBase):
    r"""Function space for 8th order equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-9` are

    .. math::
        \phi_k &= \frac{(1-x^2)^4}{h^{(4,\alpha,\beta)}_{k+4}} \frac{d^4P^{(\alpha,\beta)}_{k+4}}{dx^4}, \\

    where

    .. math::

        h^{(4,\alpha,\beta)}_k&=\int_{-1}^1 \left(\frac{d^4P^{(\alpha,\beta)}_{k}}{dx^4}\right)^2 (1-x)^{\alpha+4}(1+x)^{\beta+4}dx, \\
            &=\frac{2^{\alpha + \beta + 1} \left(\alpha + \beta + n + 1\right) \left(\alpha + \beta + n + 2\right) \left(\alpha + \beta + n + 3\right) \left(\alpha + \beta + n + 4\right) \Gamma\left(\alpha + n + 1\right) \Gamma\left(\beta + n + 1\right)}{\left(\alpha + \beta + 2 n + 1\right) \Gamma\left(n - 3\right) \Gamma\left(\alpha + \beta + n + 1\right)},

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

        - JG - Jacobi-Gauss
    bc : 8-tuple of numbers, optional
        Boundary conditions.
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(0,)*8, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 4, alpha, beta, n+4, n) / h(alpha, beta, n, 0)),
        #   2: sp.simplify(matpow(b, 4, alpha, beta, n+4, n+2) / h(alpha, beta, n+2, 0)),
        #   4: sp.simplify(matpow(b, 4, alpha, beta, n+4, n+4) / h(alpha, beta, n+4, 0)),
        #   6: sp.simplify(matpow(b, 4, alpha, beta, n+4, n+6) / h(alpha, beta, n+6, 0)),
        #   8: sp.simplify(matpow(b, 4, alpha, beta, n+4, n+8) / h(alpha, beta, n+8, 0))}
        a, b = alpha, beta
        self._stencil = {
            0: 2**(-a - b + 3)*sp.gamma(n + 1)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 2)*(a + b + 2*n + 3)*(a + b + 2*n + 4)*(a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 7)*(a + b + 2*n + 8)*sp.gamma(a + n + 1)*sp.gamma(b + n + 1)),
            2: 2**(-a - b + 4)*(3*a**2 - 8*a*b - 2*a*n - 7*a + 3*b**2 - 2*b*n - 7*b - 2*n**2 - 14*n - 20)*sp.gamma(n + 3)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 3)*(a + b + 2*n + 4)*(a + b + 2*n + 6)*(a + b + 2*n + 7)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*sp.gamma(a + n + 3)*sp.gamma(b + n + 3)),
            4: 2**(-a - b + 3)*(a + b + 2*n + 9)*(a**4 - 16*a**3*b - 12*a**3*n - 54*a**3 + 36*a**2*b**2 + 24*a**2*b*n + 108*a**2*b - 6*a**2*n**2 - 54*a**2*n - 109*a**2 - 16*a*b**3 + 24*a*b**2*n + 108*a*b**2 + 48*a*b*n**2 + 432*a*b*n + 932*a*b + 12*a*n**3 + 162*a*n**2 + 714*a*n + 1026*a + b**4 - 12*b**3*n - 54*b**3 - 6*b**2*n**2 - 54*b**2*n - 109*b**2 + 12*b*n**3 + 162*b*n**2 + 714*b*n + 1026*b + 6*n**4 + 108*n**3 + 714*n**2 + 2052*n + 2160)*sp.gamma(n + 5)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 7)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*(a + b + 2*n + 12)*(a + b + 2*n + 13)*sp.gamma(a + n + 5)*sp.gamma(b + n + 5)),
            6: 2**(-a - b + 4)*(3*a**2 - 8*a*b - 2*a*n - 11*a + 3*b**2 - 2*b*n - 11*b - 2*n**2 - 22*n - 56)*sp.gamma(n + 7)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 7)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*(a + b + 2*n + 12)*(a + b + 2*n + 14)*(a + b + 2*n + 15)*sp.gamma(a + n + 5)*sp.gamma(b + n + 5)),
            8: 2**(-a - b + 3)*sp.gamma(n + 9)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 10)*(a + b + 2*n + 11)*(a + b + 2*n + 12)*(a + b + 2*n + 13)*(a + b + 2*n + 14)*(a + b + 2*n + 15)*(a + b + 2*n + 16)*sp.gamma(a + n + 5)*sp.gamma(b + n + 5))
        }

        if alpha != beta:
            #self._stencil[1] = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+1) / h(alpha, beta, n+1, 0))
            #self._stencil[3] = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+3) / h(alpha, beta, n+3, 0))
            #self._stencil[5] = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+5) / h(alpha, beta, n+5, 0))
            #self._stencil[7] = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+7) / h(alpha, beta, n+7, 0))
            self._stencil[1] = 2**(-a - b + 5)*(a - b)*sp.gamma(n + 2)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 2)*(a + b + 2*n + 4)*(a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 7)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*sp.gamma(a + n + 2)*sp.gamma(b + n + 2))
            self._stencil[3] = 2**(-a - b + 5)*(a - b)*(a**2 - 5*a*b - 3*a*n - 12*a + b**2 - 3*b*n - 12*b - 3*n**2 - 24*n - 43)*sp.gamma(n + 4)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 4)*(a + b + 2*n + 5)*(a + b + 2*n + 6)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*(a + b + 2*n + 12)*sp.gamma(a + n + 4)*sp.gamma(b + n + 4))
            self._stencil[5] = 2**(-a - b + 5)*(a - b)*(-a**2 + 5*a*b + 3*a*n + 15*a - b**2 + 3*b*n + 15*b + 3*n**2 + 30*n + 70)*sp.gamma(n + 6)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 6)*(a + b + 2*n + 7)*(a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 12)*(a + b + 2*n + 13)*(a + b + 2*n + 14)*sp.gamma(a + n + 5)*sp.gamma(b + n + 5))
            self._stencil[7] = -2**(-a - b + 5)*(a - b)*sp.gamma(n + 8)*sp.gamma(a + b + n + 5)/((a + b + 2*n + 8)*(a + b + 2*n + 10)*(a + b + 2*n + 11)*(a + b + 2*n + 12)*(a + b + 2*n + 13)*(a + b + 2*n + 14)*(a + b + 2*n + 16)*sp.gamma(a + n + 5)*sp.gamma(b + n + 5))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic*2'

    @staticmethod
    def short_name():
        return 'P4'


class CompactDirichlet(CompositeBase):
    r"""Function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= P^{(\alpha,\beta)}_k + a_k P^{(\alpha,\beta)}_{k+1} + b_k P^{(\alpha,\beta)}_{k+2}  \quad k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{\alpha + 1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_0  - \frac{1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_1, \\
        \phi_{N-1} &= \frac{\beta + 1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_0+ \frac{1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_1,

    where

    .. math::
        a_k &= \frac{\left(\alpha - \beta\right) \left(n + 1\right) \left(\alpha + \beta + 2 n + 3\right)}{\left(\alpha + n + 1\right) \left(\beta + n + 1\right) \left(\alpha + \beta + 2 n + 4\right)}, \\
        b_k &= - \frac{\left(n + 1\right) \left(n + 2\right) \left(\alpha + \beta + 2 n + 2\right)}{\left(\alpha + n + 1\right) \left(\beta + n + 1\right) \left(\alpha + \beta + 2 n + 4\right)}, \\

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

        - JG - Jacobi-Gauss
    bc : 2-tuple of numbers, optional
        Boundary conditions at, respectively, x=(-1, 1).
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        a, b = alpha, beta
        self._stencil = {
            0: 1,
            2: -(n + 1)*(n + 2)*(a + b + 2*n + 2)/((a + n + 1)*(b + n + 1)*(a + b + 2*n + 4))
        }
        if self.alpha != self.beta:
            self._stencil[1] = (a - b)*(n + 1)*(a + b + 2*n + 3)/((a + n + 1)*(b + n + 1)*(a + b + 2*n + 4))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'CD'


class CompactNeumann(CompositeBase):
    r"""Function space for Neumann boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= P^{(\alpha,\beta)}_k + a_k P^{(\alpha,\beta)}_{k+1} + b_k P^{(\alpha,\beta)}_{k+2}  \quad k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{\alpha + 1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_0  - \frac{1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_1, \\
        \phi_{N-1} &= \frac{\beta + 1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_0+ \frac{1}{\alpha + \beta + 2}P^{(\alpha,\beta)}_1,

    where

    .. math::
        a_k &= \frac{n \left(\alpha - \beta\right) \left(\alpha + \beta + n + 1\right) \left(\alpha + \beta + 2 n + 3\right)}{\left(\alpha + n + 1\right) \left(\beta + n + 1\right) \left(\alpha + \beta + n + 2\right) \left(\alpha + \beta + 2 n + 4\right)}, \\
        b_k &= - \frac{n \left(n + 1\right) \left(\alpha + \beta + n + 1\right) \left(\alpha + \beta + 2 n + 2\right)}{\left(\alpha + n + 1\right) \left(\beta + n + 1\right) \left(\alpha + \beta + n + 3\right) \left(\alpha + \beta + 2 n + 4\right)}, \\

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

        - JG - Jacobi-Gauss
    bc : 2-tuple of numbers, optional
        Boundary conditions at, respectively, x=(-1, 1).
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        a, b = self.alpha, self.beta
        self._stencil = {
            0: 1,
            2: -n*(n+1)*(a+b+n+1)*(a+b+2*n+2)/((a+n+1)*(b+n+1)*(a+b+n+3)*(a+b+2*n+4))
        }
        if self.alpha != self.beta:
            self._stencil[1] = n*(a-b)*(a+b+n+1)*(a+b+2*n+3)/((a+n+1)*(b+n+1)*(a+b+n+2)*(a+b+2*n+4))

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'CN'


class UpperDirichlet(CompositeBase):
    r"""Function space for Dirichlet boundary conditions at right edge

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= P^{(\alpha, \beta)}_k - \frac{(k+1)}{\alpha+k+1} P^{(\alpha,\beta)}_{k+1}  \quad k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= 1

    and the expansion is

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(1) = a.

    The last basis function is for boundary condition and only used if a is
    different from 0. In one dimension :math:`\hat{u}_{N-1}=a`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - JG - Jacobi-Gauss
    bc : 2-tuple of (None, number), optional
        Boundary condition at x=1.
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(None, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        self._stencil = {0: 1, 1: -(n+1)/(self.alpha+n+1)}

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'


class LowerDirichlet(CompositeBase):
    r"""Function space for Dirichlet boundary condition at left edge

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::
        \phi_k &= P^{(\alpha, \beta)}_k + \frac{(k+1)}{\beta+k+1} P^{(\alpha,\beta)}_{k+1}  \quad k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= 1

    and the expansion is

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) = a.

    The last basis function is for boundary condition and only used if a is
    different from 0. In one dimension :math:`\hat{u}_{N-1}=a`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - JG - Jacobi-Gauss
    bc : 2-tuple of (number, None), optional
        Boundary condition at x=-1.
    alpha : number, optional
        Parameter of the Jacobi polynomial
    beta : number, optional
        Parameter of the Jacobi polynomial
    domain : 2-tuple of numbers, optional
        The computational domain
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
    def __init__(self, N, quad="JG", bc=(0, None), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0,
                 coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)
        self._stencil = {0: 1, 1: (n+1)/(self.beta+n+1)}

    @staticmethod
    def boundary_condition():
        return 'LowerDirichlet'

    @staticmethod
    def short_name():
        return 'LD'


class Generic(CompositeBase):
    r"""Function space for space with any boundary conditions

    Any combination of Dirichlet and Neumann is possible.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature
        - JG - Jacobi-Gauss
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
    domain : 2-tuple of numbers, optional
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

    Note
    ----
    A test function is always using homogeneous boundary conditions.

    """
    def __init__(self, N, quad="JG", bc={}, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None,
                 alpha=0, beta=0, **kw):
        from shenfun.utilities.findbasis import get_stencil_matrix
        self._stencil = get_stencil_matrix(bc, 'jacobi', alpha, beta, 1)
        if not isinstance(bc, BoundaryConditions):
            bc = BoundaryConditions(bc, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               alpha=alpha, beta=beta, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Generic'

    @staticmethod
    def short_name():
        return 'GJ'
