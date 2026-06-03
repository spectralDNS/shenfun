r"""
Module for function spaces of ultraspherical type.

The ultraspherical polynomial :math:`Q^{(\alpha)}_k` is here defined as

.. math::

    C^{(\alpha)}_k = \frac{(2\alpha)_k}{(\alpha+1/2)_k} P^{(\alpha-1/2,\alpha-1/2)}_k

where :math:`P^{(\alpha-1/2,\alpha-1/2)}_k` is the regular Jacobi polynomial with two
equal parameters.

.. math::

    {Q}^{(\alpha)}_k(\pm 1) = \frac{\gamma(2\alpha +1)}{\gamma(2 \alpha) k!}

"""
import profile
import copy
import numpy as np
import sympy as sp
from scipy.special import eval_jacobi, roots_jacobi  # , gamma
from shenfun.matrixbase import SparseMatrix
from shenfun.spectralbase import (
    getCompositeBase,
    getBCGeneric,
    BoundaryConditions,
    Domain,
)
from shenfun.jacobi.recursions import wn, h, alfa
from shenfun.jacobi import JacobiBase

xp = sp.Symbol("x", real=True)
m, n, k = sp.symbols("m,n,k", real=True, integer=True, positive=True)

# pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

bases = [
    "Orthogonal",
    "CompactDirichlet",
    "CompactBiharmonic",
    "Generic",
]
bcbases = ["BCGeneric"]
testbases = []
__all__ = bases + bcbases + testbases


class Orthogonal(JacobiBase):
    r"""Function space for regular (orthogonal) Gegenbauer polynomials

    The orthogonal basis is

    .. math::

        C^{(\alpha)}_k = \frac{(2\alpha)_k}{(\alpha+1/2)_k} P^{(\alpha-1/2,\alpha-1/2)}_k, \quad k = 0, 1, \ldots, N-1,

    where :math:`P^{(\alpha-1/2,\alpha-1/2)}_k` is the `Jacobi polynomial <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_.
    The basis :math:`\{C^{(\alpha)}_k\}` is orthogonal with weight :math:`(1-x^2)^{\alpha-1/2}`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - QG - Jacobi-Gauss
    alpha : number, optional
        Parameter of the Gegenbauer polynomial
    domain : Domain, 2-tuple of numbers, optional
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

    def __init__(
        self,
        N,
        quad="CG",
        lambda_=0,
        domain=Domain(-1, 1),
        dtype=float,
        padding_factor=1,
        dealias_direct=False,
        coordinates=None,
        **kw,
    ):
        JacobiBase.__init__(
            self,
            N,
            quad=quad,
            alpha=sp.sympify(lambda_ - sp.S.Half),
            beta=sp.sympify(lambda_ - sp.S.Half),
            domain=domain,
            dtype=dtype,
            padding_factor=padding_factor,
            dealias_direct=dealias_direct,
            coordinates=coordinates,
        )
        self.gn = wn
        self.plan(int(N * padding_factor), 0, dtype, {})

    @property
    def lambda_(self):
        return self.alpha + sp.S.Half

    @staticmethod
    def family():
        return "gegenbauer"

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "CG"
        points, weights = roots_jacobi(N, float(self.alpha), float(self.alpha))
        if weighted is False:
            weights = self.unweighted_quadrature_weights()

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
        V = self.jacobiQ(x, alpha + k, self.N)
        if k > 0:
            Vc = np.zeros_like(V)
            for j in range(k, self.N):
                dj = np.prod(np.array([j + 2 * alpha + 1 + i for i in range(k)]))
                Vc[:, j] = (dj / 2**k) * V[:, j - k]
            V = Vc
        return V

    def vandermonde(self, x):
        V = self.jacobiQ(x, self.alpha, self.shape(False))
        if self.alpha != 0:
            # V *= sp.lambdify(n, cn(self.alpha, self.alpha, n))(np.arange(self.N))[None, :]
            V *= np.array(
                [wn(self.alpha, self.alpha, n).subs(n, i) for i in np.arange(self.N)],
                dtype=V.dtype,
            )[None, :]
        return V

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return "W"

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        return SparseMatrix({0: 1}, (N, N))

    def sympy_stencil(
        self, i=sp.Symbol("i", integer=True), j=sp.Symbol("j", integer=True)
    ):
        return sp.KroneckerDelta(i, j)

    def orthogonal_basis_function(self, i=0, x=xp):
        return wn(self.alpha, self.alpha, i) * sp.jacobi(i, self.alpha, self.alpha, x)

    def L2_norm_sq(self, i):
        if i == 0:
            return sp.simplify(h(alfa, alfa, i, 0, wn).subs(i, 0)).subs(
                alfa, self.alpha
            )
        return h(self.alpha, self.alpha, i, 0, wn)

    def weight(self, x=xp):
        return sp.Pow(1 - x**2, self.alpha)

    def l2_norm_sq(self, i=None):
        if i is None:
            hh = np.zeros(self.N)
            hh[:] = sp.lambdify(n, h(self.alpha, self.alpha, n, 0, wn))(
                np.arange(self.N)
            )
            hh[0] = self.L2_norm_sq(0)
            return hh
        return self.L2_norm_sq(i)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_jacobi(
            i, float(self.alpha), float(self.alpha), x, out=output_array
        )
        if self.alpha != 0:
            output_array *= wn(self.alpha, self.alpha, i).n()
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights()[0]
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)

        dj = np.prod(np.array([i + 2 * (self.alpha) + 1 + j for j in range(k)]))
        output_array[:] = (
            dj
            / 2**k
            * eval_jacobi(i - k, float(self.alpha + k), float(self.alpha + k), x)
        )
        if self.alpha != 0:
            output_array[:] = output_array * wn(self.alpha, self.alpha, i).n()
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.points_and_weights()[0]
        V = self.derivative_jacobiQ(x, self.alpha, k)
        if self.alpha != 0:
            # V *= sp.lambdify(n, cn(self.alpha, self.alpha, n))(np.arange(self.N))[None, :]
            V *= np.array(
                [wn(self.alpha, self.alpha, n).subs(n, i) for i in np.arange(self.N)],
                dtype=V.dtype,
            )[None, :]
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

    def get_orthogonal(self, **kwargs):
        d = dict(
            quad=self.quad,
            domain=self.domain,
            dtype=self.dtype,
            lambda_=self.lambda_,
            padding_factor=self.padding_factor,
            dealias_direct=self.dealias_direct,
            coordinates=self.coors.coordinates,
        )
        d.update(kwargs)
        return Orthogonal(self.N, **d)

    def get_bc_space(self):
        if self._bc_space:
            return self._bc_space
        self._bc_space = BCGeneric(
            self.N, bc=self.bcs, alpha=self.alpha, beta=self.alpha, domain=self.domain
        )
        return self._bc_space

    def get_refined(self, N, **kwargs):
        """Return space (otherwise as self) with N quadrature points

        Parameters
        ----------
        N : int
            The number of quadrature points for returned space
        kwargs : keyword arguments
            Any other keyword arguments used in the creation of the bases.

        Returns
        -------
        :class:`.SpectralBase`
            A new space with new number of quadrature points, otherwise as self.
        """
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, 'bcs'):
            d['bc'] = copy.deepcopy(self.bcs)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled
        d['lambda_'] = self.lambda_
        d.update(kwargs)
        return self.__class__(N, **d)

    def get_unplanned(self, **kwargs):
        """Return unplanned space (otherwise as self)

        Parameters
        ----------
        kwargs : keyword arguments, optional
            Any keyword argument used in the creation of the unplanned
            space. Could be any one of

            - quad
            - domain
            - dtype
            - padding_factor
            - dealias_direct
            - coordinates
            - bcs
            - scaled

            Not all will be applicable for all spaces.

        Returns
        -------
        :class:`.SpectralBase`
            Space not planned for a :class:`.TensorProductSpace`

        """
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, 'bcs'):
            d['bc'] = copy.deepcopy(self.bcs)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled
        d['lambda_'] = self.lambda_
        d.update(kwargs)
        return self.__class__(self.N, **d)

    def get_homogeneous(self, **kwargs):
        """Return space (otherwise as self) with homogeneous boundary conditions

        Parameters
        ----------
        kwargs : keyword arguments
            Any keyword arguments used in the creation of the bases.

        Returns
        -------
        :class:`.SpectralBase`
            A new space with homogeneous boundary conditions, otherwise as self.
        """
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled
        d['lambda_'] = self.lambda_
        d.update(kwargs)
        return self.__class__(self.N, **d)


CompositeBase = getCompositeBase(Orthogonal)
BCGeneric = getBCGeneric(CompositeBase)


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
    domain : Domain, 2-tuple of numbers, optional
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

    def __init__(
        self,
        N,
        quad="CG",
        bc=(0, 0),
        domain=Domain(-1, 1),
        dtype=float,
        padding_factor=1,
        dealias_direct=False,
        lambda_=2,
        coordinates=None,
        **kw,
    ):
        CompositeBase.__init__(
            self,
            N,
            quad=quad,
            domain=domain,
            dtype=dtype,
            bc=bc,
            padding_factor=padding_factor,
            dealias_direct=dealias_direct,
            lambda_=lambda_,
            coordinates=coordinates,
        )
        self._stencil = {0: -(n+4)*(n+5) / (2 * (n + 1) * (n + 2)**2), 2: 1 / (2*(n + 2))}

    @staticmethod
    def boundary_condition():
        return "Dirichlet"

    @staticmethod
    def short_name():
        return "WD"

    def get_orthogonal(self, **kwargs):
        d = dict(
            quad=self.quad,
            domain=self.domain,
            dtype=self.dtype,
            lambda_=self.lambda_,
            padding_factor=self.padding_factor,
            dealias_direct=self.dealias_direct,
            coordinates=self.coors.coordinates,
        )
        d.update(kwargs)
        return Orthogonal(self.N, **d)


class CompactBiharmonic(CompositeBase):
    r"""Function space for biharmonic equations

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-5` are

    .. math::
        \phi_k &= \frac{h_n}{b^{(2)}_{n+2,n}} \frac{(1-x^2)^2}{h^{(2,\alpha)}_{k+2}} \frac{d^2Q^{(\alpha)}_{k+2}}{dx^2},

    where

    .. math::

        h^{(2,\alpha)}_k&=\int_{-1}^1 \left(\frac{d^2Q^{(\alpha)}_{k}}{dx^2}\right)^2 (1-x^2)^{\alpha+2}dx, \\
            &= \frac{2^{2 \alpha + 1} \cdot \left(2 \alpha + n + 1\right) \left(2 \alpha + n + 2\right) \Gamma^{2}\left(\alpha + n + 1\right)}{\left(2 \alpha + 2 n + 1\right) \Gamma\left(n - 1\right) \Gamma\left(2 \alpha + n + 1\right)},

    This is :class:`.Phi2` scaled such that the main diagonal of the stencil matrix is unity.

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
    domain : Domain, 2-tuple of numbers, optional
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

    def __init__(
        self,
        N,
        quad="CG",
        bc=(0, 0, 0, 0),
        domain=Domain(-1, 1),
        dtype=float,
        padding_factor=1,
        dealias_direct=False,
        lambda_=2,
        coordinates=None,
        **kw,
    ):
        CompositeBase.__init__(
            self,
            N,
            quad=quad,
            domain=domain,
            dtype=dtype,
            bc=bc,
            padding_factor=padding_factor,
            dealias_direct=dealias_direct,
            lambda_=lambda_,
            coordinates=coordinates,
        )
        # self._stencil = {
        #    0: 1,
        #    2: -2 * (n + 1) * (n + 2) * (n + 6) / ((n + 7) * (n + 8) * (n + 9)),
        #    4: (n + 1)
        #    * (n + 2)
        #    * (n + 3)
        #    * (n + 4)
        #    * (n + 5)
        #    / ((n + 7) * (n + 8) * (n + 9) * (n + 10) * (n + 11)),
        # }
        self._stencil = {
            0: (n + 7)
            * (n + 8)
            * (n + 9)
            * (n + 10)
            * (n + 11)
            / ((n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)),
            2: -2 * (n + 6) * (n + 10) * (n + 11) / ((n + 3) * (n + 4) * (n + 5)),
            4: 1,
        }

    @staticmethod
    def boundary_condition():
        return "Biharmonic"

    @staticmethod
    def short_name():
        return "W2"

    def get_orthogonal(self, **kwargs):
        d = dict(
            quad=self.quad,
            domain=self.domain,
            dtype=self.dtype,
            lambda_=self.lambda_,
            padding_factor=self.padding_factor,
            dealias_direct=self.dealias_direct,
            coordinates=self.coors.coordinates,
        )
        d.update(kwargs)
        return Orthogonal(self.N, **d)


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
    domain : Domain, 2-tuple of numbers, optional
        The computational domain
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    lambda_ : number, optional
        Parameter of the ultraspherical polynomial.
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """

    def __init__(
        self,
        N,
        quad="QG",
        bc={},
        domain=Domain(-1, 1),
        dtype=float,
        padding_factor=1,
        dealias_direct=False,
        coordinates=None,
        lambda_=0,
        **kw,
    ):
        from shenfun.utilities.findbasis import get_stencil_matrix
        alpha = lambda_ - sp.S.Half
        self._stencil = get_stencil_matrix(bc, "gegenbauer", alpha, alpha, wn)
        if not isinstance(bc, BoundaryConditions):
            bc = BoundaryConditions(bc, domain=domain)
        CompositeBase.__init__(
            self,
            N,
            quad=quad,
            domain=domain,
            dtype=dtype,
            bc=bc,
            padding_factor=padding_factor,
            dealias_direct=dealias_direct,
            lambda_=lambda_,
            coordinates=coordinates,
        )

    @staticmethod
    def boundary_condition():
        return "Generic"

    @staticmethod
    def short_name():
        return "GW"
