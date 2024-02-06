r"""
Module for defining function spaces in the Chebyshev family.

A function is approximated in the Chebyshev basis as

..  math::

    u(x) = \sum_{i=0}^{N-1} \hat{u}_i T_i(x)

where :math:`T_i(x)` is the i'th Chebyshev polynomial of the first kind.
The Chebyshev polynomials are orthogonal with weight :math:`\omega=1/\sqrt{1-x^2}`

.. math::

    \int_{-1}^1 T_i T_k \omega dx = \frac{c_k \pi}{2} \delta_{ki},

where :math:`c_0=2` and :math:`c_i=1` for :math:`i>0`.

All other bases defined in this module are combinations of :math:`T_i`'s.
For example, a Dirichlet basis is

.. math::

    \phi_i = T_i - T_{i+2}

The basis is implemented using a stencil matrix :math:`K \in \mathbb{R}^{N-2 \times N}`,
such that

.. math::

    \boldsymbol{\phi} = K \boldsymbol{T},

where :math:`\boldsymbol{\phi}=(\phi_0, \phi_1, \ldots, \phi_{N-3})` and
:math:`\boldsymbol{T}=(T_0, T_1, \ldots, T_{N-1})`. For the Dirichlet basis
:math:`K = (\delta_{i, j} - \delta_{i+2, j})_{i,j=0}^{N-2, N}`.

All composite bases make use of the fast transforms that exists for
:math:`\boldsymbol{T}` through fast cosine transforms. The stencil matrix
is used to transfer any composite basis back and forth to the orthogonal basis.

"""
from __future__ import division
import functools
import numpy as np
from numpy.polynomial import chebyshev as n_cheb
import sympy as sp
from scipy.special import eval_chebyt
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, FuncWrap, \
    islicedict, slicedict, getCompositeBase, getBCGeneric, BoundaryConditions
from shenfun.matrixbase import SparseMatrix
from shenfun.optimization import optimizer
from shenfun.config import config
from shenfun.jacobi.recursions import half, cn
from shenfun.jacobi import JacobiBase
from shenfun.utilities import n

bases = ['Orthogonal',
         'ShenDirichlet',
         'Heinrichs',
         'ShenNeumann',
         'CombinedShenNeumann',
         'MikNeumann',
         'ShenBiharmonic',
         'UpperDirichlet',
         'LowerDirichlet',
         'UpperDirichletNeumann',
         'LowerDirichletNeumann',
         'ShenBiPolar',
         'PolarDirichlet',
         'DirichletNeumann',
         'NeumannDirichlet',
         'Compact3',
         'Compact4',
         'Generic']
bcbases = ['BCGeneric']
testbases = ['Phi1', 'Phi2', 'Phi3', 'Phi4', 'Phi6']

__all__ = bases + bcbases + testbases

#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

chebval = optimizer(n_cheb.chebval)

xp = sp.Symbol('x', real=True)

class DCTWrap(FuncWrap):
    """DCT for complex input"""

    @property
    def dct(self):
        return object.__getattribute__(self, '_func')

    def __call__(self, **kw):
        dct_obj = self.dct
        dct_obj.input_array[...] = self.input_array.real
        dct_obj(None, None, **kw)
        self.output_array.real[...] = dct_obj.output_array
        dct_obj.input_array[...] = self.input_array.imag
        dct_obj(None, None, **kw)
        self.output_array.imag[...] = dct_obj.output_array
        return self.output_array


class Orthogonal(JacobiBase):
    r"""Function space for regular Chebyshev series

    The orthogonal basis is

    .. math::

        T_k, \quad k = 0, 1, \ldots, N-1,

    where :math:`T_k` is the :math:`k`'th Chebyshev polynomial of the first
    kind.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
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
    """

    def __init__(self, N, quad='GC', domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        JacobiBase.__init__(self, N, quad=quad, alpha=-half, beta=-half, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        assert quad in ('GC', 'GL', 'GU')
        self.gn = cn
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dctn, type=2)
            self._xfftn_bck = functools.partial(fftw.dctn, type=3)
            self._xfftn_fwd.opts = self._xfftn_bck.opts = config['fftw']['dct']
        elif quad in ('GL', 'GU'):
            self._xfftn_fwd = functools.partial(fftw.dctn, type=1)
            self._xfftn_bck = functools.partial(fftw.dctn, type=1)
            self._xfftn_fwd.opts = self._xfftn_bck.opts = config['fftw']['dct']
        self.plan((int(padding_factor*N),), 0, dtype, {})

    # Comment due to curvilinear issues
    #def apply_inverse_mass(self, array):
    #    coors = self.tensorproductspace.coors if self.dimensions > 1 else self.coors
    #    if not coors.hi.prod() == 1:
    #        return JacobiBase.apply_inverse_mass(self, array)
    #    array *= (2/np.pi*self.domain_factor())
    #    array[self.si[0]] /= 2
    #    if self.quad == 'GL':
    #        array[self.si[-1]] /= 2
    #    return array

    @staticmethod
    def family():
        return 'chebyshev'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)

        if weighted:
            if self.quad == "GL":
                points = -(n_cheb.chebpts2(N)).astype(float)
                weights = np.full(N, np.pi/(N-1))
                weights[0] /= 2
                weights[-1] /= 2

            elif self.quad == "GC":
                points, weights = n_cheb.chebgauss(N)
                points = points.astype(float)
                weights = weights.astype(float)

            elif self.quad == "GU":
                points = np.cos((np.arange(N)+1)*np.pi/(N+1))
                weights = np.full(N, np.pi/(N+1))

        else:
            if self.quad == "GL":
                points = np.cos(np.arange(N)*np.pi/(N-1))
                d = fftw.aligned(N, fill=0)
                k = 2*(1 + np.arange((N-1)//2))
                d[::2] = (2./(N-1))/np.hstack((1., 1.-k*k))
                w = fftw.aligned_like(d)
                dct = fftw.dctn(w, axes=(0,), type=1)
                weights = dct(d, w)
                weights[0] *= 0.5
                weights[-1] *= 0.5

            elif self.quad == "GC":
                points = n_cheb.chebgauss(N)[0]
                d = fftw.aligned(N, fill=0)
                k = 2*(1 + np.arange((N-1)//2))
                d[::2] = (2./N)/np.hstack((1., 1.-k*k))
                w = fftw.aligned_like(d)
                dct = fftw.dctn(w, axes=(0,), type=3)
                weights = dct(d, w)

            elif self.quad == "GU":
                theta = (np.arange(N)+1)*np.pi/(N+1)
                points = np.cos(theta)
                d = fftw.aligned(N, fill=0)
                k = np.arange(N)
                d[::2] = 2/(k[::2]+1)
                w = fftw.aligned_like(d)
                dst = fftw.dstn(w, axes=(0,), type=1)
                weights = dst(d, w)
                weights *= (np.sin(theta))/(N+1)

        if map_true_domain is True:
            points = self.map_true_domain(points)

        return points, weights

    def vandermonde(self, x):
        return n_cheb.chebvander(x, self.shape(False)-1)

    def weight(self, x=xp):
        return 1/sp.sqrt(1-x**2)

    def orthogonal_basis_function(self, i=0, x=xp):
        return sp.chebyshevt(i, x)

    def L2_norm_sq(self, i):
        return (1+int(i==0))*sp.pi/2

    def l2_norm_sq(self, i=None):
        if i is None:
            f = np.full(self.N, np.pi/2)
            f[0] *= 2
            if self.quad == 'GL':
                f[-1] *= 2
            return f
        elif i == 0 or i == self.N-1 and self.quad == 'GL':
            return np.pi
        return np.pi/2

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        #output_array[:] = np.cos(i*np.arccos(x))
        output_array[:] = eval_chebyt(i, x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[i] = 1
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        M = V.shape[1]
        if k > 0:
            D = np.zeros((M, M))
            D[:-k] = n_cheb.chebder(np.eye(M, M), k)
            V = np.dot(V, D)
        return V

    def _evaluate_expansion_all(self, input_array, output_array, x=None, kind='fast'):
        if kind != 'fast':
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, kind=kind)
            return

        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        output_array = self.backward.xfftn()

        if self.quad == "GC":
            s0 = self.sl[slice(0, 1)]
            output_array *= 0.5
            output_array += input_array[s0]/2

        elif self.quad == "GL":
            output_array *= 0.5
            output_array += input_array[self.sl[slice(0, 1)]]/2
            s0 = self.sl[slice(-1, None)]
            s2 = self.sl[slice(0, None, 2)]
            output_array[s2] += input_array[s0]/2
            s2 = self.sl[slice(1, None, 2)]
            output_array[s2] -= input_array[s0]/2

    def _evaluate_scalar_product(self, kind='fast'):
        if kind != 'fast':
            SpectralBase._evaluate_scalar_product(self, kind=kind)
            return

        if self.quad == "GC":
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*self.domain_factor()*self.N*self.padding_factor))

        elif self.quad == "GL":
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*self.domain_factor()*(self.N*self.padding_factor-1)))
    #@profile
    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        x = self.map_reference_domain(x)
        #oa = chebval(x, u)
        oa = n_cheb.chebval(x, u, False)
        if output_array is not None:
            output_array[:] = oa
            return output_array
        return oa

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'T'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        return SparseMatrix({0: 1}, (N, N))

    def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True)):
        return sp.KroneckerDelta(i, j)

    def to_ortho(self, input_array, output_array=None):
        assert input_array.__class__.__name__ == 'Orthogonal'
        if output_array:
            output_array[:] = input_array
            return output_array
        return input_array

    def get_orthogonal(self, **kwargs):
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return Orthogonal(self.N, **d)

    def get_bc_space(self):
        if self._bc_space:
            return self._bc_space
        self._bc_space = BCGeneric(self.N, bc=self.bcs, domain=self.domain)
        return self._bc_space

    def plan(self, shape, axis, dtype, options):
        if shape in (0, (0,)):
            return

        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        opts = plan_fwd.opts
        opts['overwrite_input'] = 'FFTW_DESTROY_INPUT'
        opts.update(options)
        flags = (fftw.flag_dict[opts['planner_effort']],
                 fftw.flag_dict[opts['overwrite_input']])
        threads = opts['threads']

        U = fftw.aligned(shape, dtype=float)

        xfftn_fwd = plan_fwd(U, axes=(axis,), threads=threads, flags=flags)
        V = xfftn_fwd.output_array
        xfftn_bck = plan_bck(V, axes=(axis,), threads=threads, flags=flags, output_array=U)
        V.fill(0)
        U.fill(0)

        if np.dtype(dtype) is np.dtype('complex'):
            # dct only works on real data, so need to wrap it
            U = fftw.aligned(shape, dtype=complex)
            V = fftw.aligned(shape, dtype=complex)
            U.fill(0)
            V.fill(0)
            xfftn_fwd = DCTWrap(xfftn_fwd, U, V)
            xfftn_bck = DCTWrap(xfftn_bck, V, U)

        self.axis = axis
        if self.padding_factor != 1:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, trunc_array)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
        else:
            self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)

        self.si = islicedict(axis=self.axis, dimensions=U.ndim)
        self.sl = slicedict(axis=self.axis, dimensions=U.ndim)

# Note that all composite spaces rely on the fast transforms of
# the orthogonal space. For this reason we have an intermediate
# class CompositeBase for all composite spaces, where common code
# is implemented and reused by all.
CompositeBase = getCompositeBase(Orthogonal)
BCGeneric = getBCGeneric(CompositeBase)

class ShenDirichlet(CompositeBase):
    r"""Function space for Dirichlet boundary conditions.

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_k - T_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{1}{2}(T_0-T_1), \\
        \phi_{N-1} &= \frac{1}{2}(T_0+T_1),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a \text{ and } u(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of floats, optional
        Boundary conditions at, respectively, x=(-1, 1).
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 2: -1}

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'SD'

    #def _evaluate_scalar_product(self, kind='fast'):
    #    if kind != 'fast':
    #        SpectralBase._evaluate_scalar_product(self, kind=kind)
    #        self.scalar_product.tmp_array[self.si[-2]] = 0
    #        self.scalar_product.tmp_array[self.si[-1]] = 0
    #        return
    #    Orthogonal._evaluate_scalar_product(self, kind=kind)
    #    output = self.scalar_product.tmp_array
    #    s0 = self.sl[slice(0, self.N-2)]
    #    s1 = self.sl[slice(2, self.N)]
    #    output[s0] -= output[s1]
    #    output[self.si[-2]] = 0
    #    output[self.si[-1]] = 0

    #def to_ortho(self, input_array, output_array=None):
    #    if output_array is None:
    #        output_array = np.zeros_like(input_array)
    #    else:
    #        output_array.fill(0)
    #    s0 = self.sl[slice(0, self.N-2)]
    #    s1 = self.sl[slice(2, self.N)]
    #    output_array[s0] = input_array[s0]
    #    output_array[s1] -= input_array[s0]
    #    self.bc._add_to_orthogonal(output_array, input_array)
    #    return output_array

class Phi1(CompositeBase):
    r"""Function space for Dirichlet boundary conditions.

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= \frac{T_k - T_{k+2}}{\pi (k+1)} = \frac{2(1-x^2)}{\pi k(k+1)} T'_{k+1}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{1}{2}(T_0-T_1), \\
        \phi_{N-1} &= \frac{1}{2}(T_0+T_1),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a \text{ and } u(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of floats, optional
        Boundary conditions at, respectively, x=(-1, 1).
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(b(-half, -half, n+1, n, cn) / (h(-half, -half, n, 0, cn))),
        #   2: sp.simplify(b(-half, -half, n+1, n+2, cn) / (h(-half, -half, n+2, 0, cn)))}
        self._stencil = {0: 1/sp.pi/(n+1), 2: -1/sp.pi/(n+1)}

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'


class Heinrichs(CompositeBase):
    r"""Function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= \alpha_k(1-x^2) T_{k}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{1}{2}(T_0-T_1), \\
        \phi_{N-1} &= \frac{1}{2}(T_0+T_1),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a \text{ and } u(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`. If the parameter `scaled=True`, then
    :math:`\alpha_k=1/(k+1)/(k+2)`, otherwise, :math:`\alpha_k=1`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of floats, optional
        Boundary conditions at, respectively, x=(-1, 1).
    domain : 2-tuple of numbers, optional
        The computational domain
    dtype : data-type, optiona
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    scaled : bool, optional
        Whether or not to use scaled basis
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'HH'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = 0.5*np.ones(N, dtype=int)
        d[-2:] = 0
        d[1] = 0.25
        dm2 = -0.25*np.ones(N-2, dtype=int)
        dm2[-2:] = 0
        dp2 = -0.25*np.ones(N-2, dtype=int)
        dp2[0] = -0.5
        if self.is_scaled():
            k = np.arange(N)
            d /= ((k+1)*(k+2))
            dm2 /= ((k[:-2]+3)*(k[:-2]+4))
            dp2 /= ((k[:-2]+1)*(k[:-2]+2))
        return SparseMatrix({-2: dm2, 0: d, 2: dp2}, (N, N))

    def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True)):
        return RuntimeError, "Not possible for current basis"


class ShenNeumann(CompositeBase):
    r"""Function space for Neumann boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} - \left(\frac{k}{k+2}\right)^2 T_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \frac{1}{8}(4T_1-T_2), \\
        \phi_{N-1} &= \frac{1}{8}(4T_1+T_2),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u'(-1) &= a \text{ and } u'(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of floats, optional
        Boundary condition values at, respectively, x=(-1, 1).
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 2: -(n/(n+2))**2}

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'SN'


class CombinedShenNeumann(CompositeBase):
    r"""Function space for Neumann boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k = \begin{cases}
            T_0, \quad &k=0, \\
            T_1 - T_3/9, \quad &k=1, \\
            T_2/4 - T_4/16, \quad &k=2, \\
            -\frac{T_{k-2}}{(k-2)^2} +2\frac{T_k}{k^2} - \frac{T_{k+2}}{(k+2)^2}, &k=3, 4, \ldots, N-3, \\
            \frac{1}{8}(4T_1-T_2), \quad &k=N-2 \\
            \frac{1}{8}(4T_1+T_2), \quad &k=N-1
        \end{cases}

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u'(-1) &= a \text{ and } u'(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of floats, optional
        Boundary condition values at, respectively, x=(-1, 1).
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'CN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        k[0] = 1
        d = 2/k**2
        d[-2:] = 0
        d[0] = 1
        d[1] = 1
        d[2] = 0.25
        dm2 = -1/k[:-2]**2
        dm2[0] = 0
        dm2[-2:] = 0
        dp2 = -1/k[2:]**2
        dp2[0] = 0
        return SparseMatrix({-2: dm2, 0: d, 2: dp2}, (N, N))

    def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True)):
        return RuntimeError, "Not possible for current basis"

class MikNeumann(CompositeBase):
    r"""Function space for Neumann boundary conditions

    The basis functions :math:`\phi_k`  for :math:`k=0,1, \ldots, N-3` are

    .. math::

        \phi_k &= \frac{2}{k+1}\int (T_{k-1}-T_{k+1}),

    which (with also boundary functions) leads to the basis

    .. math::

        \phi_k = \frac{1}{k+1} \begin{cases}
            T_0, &k=0, \\
            3T_1-T_3/3, &k=1, \\
            T_2-T_4/4, &k=2, \\
            -\frac{T_{k-2}}{k-2} + 2\frac{T_k}{k} - \frac{T_{k+2}}{k+2} , &k=3, 4, \ldots, N-3, \\
            \frac{1}{8}(4T_1-T_2), \quad &k=N-2 \\
            \frac{1}{8}(4T_1+T_2), \quad &k=N-1
        \end{cases}

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u'(-1) &= a \text{ and } u'(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of floats, optional
        Boundary condition values at, respectively, x=(-1, 1).
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
    """
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'MN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        k[0] = 1
        d = 2/k/(k+1)
        d[-2:] = 0
        d[0] = 1
        d[1] = 3/2
        d[2] = 1/3
        dm2 = -1/k[:-2]/(k[2:]+1)
        dm2[0] = 0
        dm2[-2:] = 0
        dp2 = -1/k[2:]/(k[2:]-1)
        dp2[0] = 0
        #dp2[1] = -1/6
        #dp2[2] = -1/12
        return SparseMatrix({-2: dm2, 0: d, 2: dp2}, (N, N))

    def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True)):
        return RuntimeError, "Not possible for current basis"


class ShenBiharmonic(CompositeBase):
    r"""Function space for biharmonic equation.

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_n - \frac{2(k+2)}{k+3}T_{k+2}+\frac{k+1}{k+3}T_{k+4}, \, k=0, 1, \ldots, N-5, \\
        \phi_{N-4} &= \frac{1}{16}(8T_0-9T_1+T_3), \\
        \phi_{N-3} &= \frac{1}{16}(2T_0-T_1-2T_2+T_3), \\
        \phi_{N-2} &= \frac{1}{16}(8T_0+9T_1-T_3), \\
        \phi_{N-1} &= \frac{1}{16}(2T_0-T_1+2T_2+T_3),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1)&=a, u'(-1) = b, u(1)=c, u'(1) = d.

    The last four bases are for boundary conditions and only used if a, b, c or d are
    different from 0. In one dimension :math:`\hat{u}_{N-4}=a`, :math:`\hat{u}_{N-3}=b`,
    :math:`\hat{u}_{N-2}=c` and :math:`\hat{u}_{N-1}=d`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 4-tuple of numbers
        The values of the 4 boundary conditions at x=(-1, 1).
        The two conditions on x=-1 first, and then x=1.
        With (a, b, c, d) corresponding to
        bc = {'left': [('D', a), ('N', b)], 'right': [('D', c), ('N', d)]}
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 2: -(2*n + 4)/(n + 3), 4: (n + 1)/(n + 3)}

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SB'

class PolarDirichlet(CompositeBase):
    r"""Function space for polar coordinates.

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_k - T_{k+2}, \, k=0, 1
        \phi_k &= T_{k-2} - T_{k+2}, \, k=2, 3, \ldots, N-3, \\
        \phi_{N-2} &= \frac{1}{2}(T_0-T_1), \\
        \phi_{N-1} &= \frac{1}{2}(T_0+T_1),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1)&=a, u(1)=b

    The last four bases are for boundary conditions and only used if a, b, c or d are
    different from 0. In one dimension :math:`\hat{u}_{N-4}=a`, :math:`\hat{u}_{N-3}=b`,
    :math:`\hat{u}_{N-2}=c` and :math:`\hat{u}_{N-1}=d`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 4-tuple of numbers
        The values of the 4 boundary conditions at x=(-1, 1).
        The two conditions on x=-1 first, and then x=1.
        With (a, b, c, d) corresponding to
        bc = {'left': [('D', a), ('N', b)], 'right': [('D', c), ('N', d)]}
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N)
        d[2:] = 0
        dm2 = np.ones(N)
        dm2[-2:] = 0
        return SparseMatrix({-2: dm2, 0: d, 2: -1}, (N, N))

    @staticmethod
    def short_name():
        return 'PD'

    def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True)):
        return RuntimeError, "Not possible for current basis"


class Phi2(CompositeBase):
    r"""Function space for biharmonic equation.

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-5` are

    .. math::

        \phi_k = \frac{2(1-x^2)^2 T''_{k+2}}{\pi (k+1)(k+2)^2(k+3)} ,

    which (along with boundary functions) gives the basis

    .. math::

        \phi_k &= \frac{1}{2 \pi (k+1)(k+2)}(T_k - \frac{2(k+2)}{k+3}T_{k+2} + \frac{k+1}{k+3}T_{k+4}), \, k=0, 1, \ldots, N-5, \\
        \phi_{N-4} &= \frac{1}{16}(8T_0-9T_1+T_3), \\
        \phi_{N-3} &= \frac{1}{16}(2T_0-T_1-2T_2+T_3), \\
        \phi_{N-2} &= \frac{1}{16}(8T_0+9T_1-T_3), \\
        \phi_{N-1} &= \frac{1}{16}(2T_0-T_1+2T_2+T_3),

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1)&=a, u'(-1) = b, u(1)=c, u'(1) = d.

    The last four bases are for boundary conditions and only used if a, b, c or d are
    different from 0. In one dimension :math:`\hat{u}_{N-4}=a`, :math:`\hat{u}_{N-3}=b`,
    :math:`\hat{u}_{N-2}=c` and :math:`\hat{u}_{N-1}=d`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 4-tuple of numbers
        The values of the 4 boundary conditions at x=(-1, 1).
        The two conditions at x=-1 first and then x=1.
        With (a, b, c, d) corresponding to
        `bc = {'left': {'D': a, 'N': b}, 'right': {'D': c, 'N': d}}`
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #    0: sp.simplify(matpow(b, 2, -half, -half, n+2, n, cn) / h(-half, -half, n, 0, cn)),
        #    2: sp.simplify(matpow(b, 2, -half, -half, n+2, n+2, cn) / h(-half, -half, n+2, 0, cn)),
        #    4: sp.simplify(matpow(b, 2, -half, -half, n+2, n+4, cn) / h(-half, -half, n+4, 0, cn))}
        self._stencil = {
            0: 1/(2*sp.pi*(n + 1)*(n + 2)),
            2: -1/(sp.pi*(n**2 + 4*n + 3)),
            4:  1/(2*sp.pi*(n + 2)*(n + 3))
        }

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'


class Phi3(CompositeBase):
    r"""Function space for 6'th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-7` are

    .. math::

        \phi_k &= \frac{(1-x^2)^3}{h^{(3)}_{k+3}} T^{(3)}_{k+3} \\
        h^{(3)}_{k+3} &= \frac{\pi (k+3) \Gamma (k+6)}{2k!} = \int_{-1}^1 T^{(3)}_k T^{(3)}_k (1-x^2)^{2.5} dx.

    where :math:`T^{(3)}_k` is the 3rd derivative of :math:`T_k`. The boundary
    basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u(1)=d u'(1)=e, u''(1)=f.

    The last 6 basis functions are for boundary conditions and only used if there
    are nonzero boundary conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 6-tuple of numbers
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

    """
    def __init__(self, N, quad="GC", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #    0: sp.simplify(matpow(b, 3, -half, -half, n+3, n, cn) / h(-half, -half, n, 0, cn)),
        #    2: sp.simplify(matpow(b, 3, -half, -half, n+3, n+2, cn) / h(-half, -half, n+2, 0, cn)),
        #    4: sp.simplify(matpow(b, 3, -half, -half, n+3, n+4, cn) / h(-half, -half, n+4, 0, cn)),
        #    6: sp.simplify(matpow(b, 3, -half, -half, n+3, n+6, cn) / h(-half, -half, n+6, 0, cn))}
        # Below is the same but faster since already simplified
        self._stencil = {
            0: 1/(4*sp.pi*(n + 1)*(n + 2)*(n + 3)),
            2: -3/(4*sp.pi*(n + 1)*(n + 3)*(n + 4)),
            4: 3/(4*sp.pi*(n + 2)*(n + 3)*(n + 5)),
            6: -1/(4*sp.pi*(n + 3)*(n + 4)*(n + 5))
        }

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'P3'


class Phi4(CompositeBase):
    r"""Function space with 4 Dirichlet and 4 Neumann boundary conditions

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-9` are

    .. math::

        \phi_k &= \frac{(1-x^2)^4}{h^{(4)}_{k+4}} T^{(4)}_{k+4} \\
        h^{(4)}_k &= \frac{\pi k \Gamma (k+4)}{2(k-4)!} = \int_{-1}^1 T^{(4)}_k T^{(4)}_k (1-x^2)^{3.5} dx,

    where :math:`T^{(4)}_k` is the 4th derivative of :math:`T_k`. The boundary
    basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 8-tuple of numbers
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

    """
    def __init__(self, N, quad="GC", bc=(0,)*8, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #    0: sp.simplify(matpow(b, 4, -half, -half, n+4, n, cn) / h(-half, -half, n, 0, cn)),
        #    2: sp.simplify(matpow(b, 4, -half, -half, n+4, n+2, cn) / h(-half, -half, n+2, 0, cn)),
        #    4: sp.simplify(matpow(b, 4, -half, -half, n+4, n+4, cn) / h(-half, -half, n+4, 0, cn)),
        #    6: sp.simplify(matpow(b, 4, -half, -half, n+4, n+6, cn) / h(-half, -half, n+6, 0, cn)),
        #    8: sp.simplify(matpow(b, 4, -half, -half, n+4, n+8, cn) / h(-half, -half, n+8, 0, cn))}
        # Below is the same but faster since already simplified
        self._stencil = {
            0: 1/(8*sp.pi*(n + 1)*(n + 2)*(n + 3)*(n + 4)),
            2: -1/(2*sp.pi*(n + 1)*(n + 3)*(n + 4)*(n + 5)),
            4: 3/(4*sp.pi*(n + 2)*(n + 3)*(n + 5)*(n + 6)),
            6: -1/(2*sp.pi*(n + 3)*(n + 4)*(n + 5)*(n + 7)),
            8: 1/(8*sp.pi*(n + 4)*(n + 5)*(n + 6)*(n + 7))
        }

    @staticmethod
    def boundary_condition():
        return 'Biharmonic*2'

    @staticmethod
    def short_name():
        return 'P4'

class Phi6(CompositeBase):
    r"""Function space for 12th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-13` are

    .. math::

        \phi_k &= \frac{(1-x^2)^6}{h^{(6)}_{k+6}} T^{(6)}_{k+6} \\
        h^{(6)}_k &= \frac{\pi k (k+5)!}{2(k-6)!} = \int_{-1}^1 T^{(6)}_k T^{(6)}_k (1-x^2)^{5.5} dx,

    where :math:`T^{(6)}_k` is the 6th derivative of :math:`T_k`. The boundary
    basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 12-tuple of numbers
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

    """
    def __init__(self, N, quad="GC", bc=(0,)*12, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #    0: sp.simplify(matpow(b, 6, -half, -half, n+6, n, cn) / h(-half, -half, n, 0, cn)),
        #    2: sp.simplify(matpow(b, 6, -half, -half, n+6, n+2, cn) / h(-half, -half, n+2, 0, cn)),
        #    4: sp.simplify(matpow(b, 6, -half, -half, n+6, n+4, cn) / h(-half, -half, n+4, 0, cn)),
        #    6: sp.simplify(matpow(b, 6, -half, -half, n+6, n+6, cn) / h(-half, -half, n+6, 0, cn)),
        #    8: sp.simplify(matpow(b, 6, -half, -half, n+6, n+8, cn) / h(-half, -half, n+8, 0, cn)),
        #   10: sp.simplify(matpow(b, 6, -half, -half, n+6, n+10, cn) / h(-half, -half, n+10, 0, cn)),
        #   12: sp.simplify(matpow(b, 6, -half, -half, n+6, n+12, cn) / h(-half, -half, n+12, 0, cn))}
        # Below is the same but faster since already simplified
        self._stencil = {
            0:  1/(32*sp.pi*(n + 1)*(n + 2)*(n + 3)*(n + 4)*(n + 5)*(n + 6)),
            2: -3/(16*sp.pi*(n + 1)*(n + 3)*(n + 4)*(n + 5)*(n + 6)*(n + 7)),
            4: 15/(32*sp.pi*(n + 2)*(n + 3)*(n + 5)*(n + 6)*(n + 7)*(n + 8)),
            6: -5/(8*sp.pi*(n + 3)*(n + 4)*(n + 5)*(n + 7)*(n + 8)*(n + 9)),
            8: 15/(32*sp.pi*(n + 4)*(n + 5)*(n + 6)*(n + 7)*(n + 9)*(n + 10)),
           10: -3/(16*sp.pi*(n + 5)*(n + 6)*(n + 7)*(n + 8)*(n + 9)*(n + 11)),
           12: 1/(32*sp.pi*(n + 6)*(n + 7)*(n + 8)*(n + 9)*(n + 10)*(n + 11))
        }

    @staticmethod
    def boundary_condition():
        return '12th order'

    @staticmethod
    def short_name():
        return 'P6'

class UpperDirichlet(CompositeBase):
    r"""Function space with single Dirichlet on upper edge

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} - T_{k+1}, \, k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= T_0,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(1) &= a.

    The last basis function is for boundary condition and only used if a is
    different from 0. In one dimension :math:`\hat{u}_{N-1}=a`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of (None, number), optional
        The number is the boundary condition value
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

    """
    def __init__(self, N, quad="GC", bc=(None, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 1: -1}

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'


class LowerDirichlet(CompositeBase):
    r"""Function space with single Dirichlet boundary condition

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} + T_{k+1}, \, k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= T_0,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a.

    The last basis funciton is for boundary condition and only used if a is
    different from 0. In one dimension :math:`\hat{u}_{N-1}=a`.

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : tuple of (number, None)
        Boundary conditions at edges of domain.
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
    def __init__(self, N, quad="GC", bc=(0, None), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 1: 1}

    @staticmethod
    def boundary_condition():
        return 'LowerDirichlet'

    @staticmethod
    def short_name():
        return 'LD'


class ShenBiPolar(CompositeBase):
    """Function space for the Biharmonic equation in polar coordinates

    u(-1)=a, u(1)=c, u'(-1)=b and u'(1)=d

    Parameters
    ----------
    N : int
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 4-tuple of numbers
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

    """
    def __init__(self, N, quad="GC", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SP'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d = (k+1)*3/8
        d[0] = 0.5
        d[1] = 0.5
        d[-4:] = 0
        dm2 = -(k[2:]+1)/8
        dm2[-4:] = 0
        dp2 = -(k[:-2]+1)*3/8
        dp2[-2:] = 0
        dp4 = (k[:-4]+1)/8
        return SparseMatrix({-2: dm2, 0: d, 2: dp2, 4: dp4}, (N, N))

    def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True)):
        return RuntimeError, "Not possible for current basis"


class DirichletNeumann(CompositeBase):
    r"""Function space for mixed Dirichlet/Neumann boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} + \frac{4(k+1)}{2k^2+6k+5}T_{k+1} - \frac{2k^2+2k+1}{2k^2+6k+5}T_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= T_0, \\
        \phi_{N-1} &= T_0+T_1,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(1)=b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of numbers
        Boundary condition values at x=-1 and x=1
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'D': bc[0]}, 'right': {'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {
            0: 1,
            1: 4*(n + 1)/(2*n**2 + 6*n + 5),
            2: -(2*n**2 + 2*n + 1)/(2*n**2 + 6*n + 5)
        }

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumann'

    @staticmethod
    def short_name():
        return 'DN'


class NeumannDirichlet(CompositeBase):
    r"""Function space for mixed Neumann/Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} - \frac{4(k+1)}{2k^2+6k+5}T_{k+1} - \frac{2k^2+2k+1}{2k^2+6k+5}T_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= -T_0+T_1, \\
        \phi_{N-1} &= T_0,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u'(-1) &= a, u(1)=b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of numbers
        Boundary condition values at x=-1 and x=1
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'D': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {
            0: 1,
            1: -(4*n + 4)/(2*n**2 + 6*n + 5),
            2: -(2*n**2 + 2*n + 1)/(2*n**2 + 6*n + 5)
        }

    @staticmethod
    def boundary_condition():
        return 'NeumannDirichlet'

    @staticmethod
    def short_name():
        return 'ND'


class UpperDirichletNeumann(CompositeBase):
    r"""Function space for both Dirichlet and Neumann boundary conditions
    on the right hand side.

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} - \frac{4(k+1)}{2k+3}T_{k+1} + \frac{2k+1}{2k+3}T_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= T_0, \\
        \phi_{N-1} &= -T_0+T_1,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(1) &= a, u'(1)=b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of numbers
        Boundary condition values at the right edge of domain
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'right': {'D': bc[0], 'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {
            0: 1,
            1: -(4*n + 4)/(2*n + 3),
            2: (2*n + 1)/(2*n + 3)
        }

    @staticmethod
    def boundary_condition():
        return 'UpperDirichletNeumann'

    @staticmethod
    def short_name():
        return 'US'


class LowerDirichletNeumann(CompositeBase):
    r"""Function space for both Dirichlet and Neumann boundary conditions
    on the left hand side

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= T_{k} + \frac{4(k+1)}{2k+3}T_{k+1} + \frac{2k+1}{2k+3}T_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= T_0, \\
        \phi_{N-1} &= T_0+T_1,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

    bc : 2-tuple of numbers
        Boundary condition values at the left edge of domain
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'D': bc[0], 'N': bc[1]}}, domain=domain)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {
            0: 1,
            1: 4*(n + 1)/(2*n + 3),
            2: (2*n + 1)/(2*n + 3)
        }

    @staticmethod
    def boundary_condition():
        return 'LowerDirichletNeumann'

    @staticmethod
    def short_name():
        return 'LS'

class Compact3(CompositeBase):
    r"""Function space for 6'th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-7` are

    .. math::

        \phi_k &= \frac{h_k}{b^{(3)}_{k+3,k}}\frac{(1-x^2)^3}{h^{(3)}_{k+3}} T^{(3)}_{k+3} \\
        h^{(3)}_{k+3} &= \frac{\pi (k+3) \Gamma (k+6)}{2k!} = \int_{-1}^1 T^{(3)}_k T^{(3)}_k (1-x^2)^{2.5} dx.

    where :math:`T^{(3)}_k` is the 3rd derivative of :math:`T_k`.
    This is :class:`.Phi3` scaled such that the main diagonal of the stencil
    matrix is unity.

    The boundary basis for inhomogeneous boundary conditions is too messy to
    print, but can be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`.
    We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u(1)=d u'(1)=e, u''(1)=f.

    The last 6 basis functions are for boundary conditions and only used if there
    are nonzero boundary conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 6-tuple of numbers
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

    """
    def __init__(self, N, quad="GC", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #    0: 1,
        #    2: sp.simplify(matpow(b, 3, -half, -half, n+3, n+2, cn) / matpow(b, 3, -half, -half, n+3, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+2, 0, cn)),
        #    4: sp.simplify(matpow(b, 3, -half, -half, n+3, n+4, cn) / matpow(b, 3, -half, -half, n+3, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+4, 0, cn)),
        #    6: sp.simplify(matpow(b, 3, -half, -half, n+3, n+6, cn) / matpow(b, 3, -half, -half, n+3, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+6, 0, cn))}
        # Below is the same but faster since already simplified
        # Can also use findbasis.get_stencil_matrix
        self._stencil = {
            0: 1,
            2: -(3*n + 6)/(n + 4),
            4: 3*(n + 1)/(n + 5),
            6: -(n + 1)*(n + 2)/((n + 4)*(n + 5))
        }

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'C3'

class Compact4(CompositeBase):
    r"""Function space for 8'th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, 1, \ldots, N-9` are

    .. math::

        \phi_k &= \frac{h_k}{b^{(4)}_{k+4,k}}\frac{(1-x^2)^4}{h^{(4)}_{k+4}} T^{(4)}_{k+4} \\
        h^{(4)}_{k+4} &= \int_{-1}^1 T^{(4)}_k T^{(4)}_k (1-x^2)^{3.5} dx.

    where :math:`T^{(4)}_k` is the 4rd derivative of :math:`T_k`.
    This is :class:`.Phi4` scaled such that the main diagonal of the stencil
    matrix is unity.

    The boundary basis for inhomogeneous boundary conditions is too messy to
    print, but can be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`.
    We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u'''(-1)=d, u(1)=e u'(1)=f, u''(1)=g, u'''(1)=h.

    The last 8 basis functions are for boundary conditions and only used if there
    are nonzero boundary conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    bc : 8-tuple of numbers
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

    """
    def __init__(self, N, quad="GC", bc=(0,)*8, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #    0: 1,
        #    2: sp.simplify(matpow(b, 4, -half, -half, n+4, n+2, cn) / matpow(b, 4, -half, -half, n+4, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+2, 0, cn)),
        #    4: sp.simplify(matpow(b, 4, -half, -half, n+4, n+4, cn) / matpow(b, 4, -half, -half, n+4, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+4, 0, cn)),
        #    6: sp.simplify(matpow(b, 4, -half, -half, n+4, n+6, cn) / matpow(b, 4, -half, -half, n+4, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+6, 0, cn)),
        #    8: sp.simplify(matpow(b, 4, -half, -half, n+4, n+8, cn) / matpow(b, 4, -half, -half, n+4, n, cn) * h(-half, -half, n, 0, cn) / h(-half, -half, n+8, 0, cn))}

        # Below is the same but faster since already simplified
        # Can also use findbasis.get_stencil_matrix
        self._stencil = {
            0: 1,
            2: -(4*n + 8)/(n + 5),
            4: 6*(n + 1)*(n + 4)/((n + 5)*(n + 6)),
            6: -4*(n + 1)*(n + 2)/((n + 5)*(n + 7)),
            8: (n + 1)*(n + 2)*(n + 3)/((n + 5)*(n + 6)*(n + 7))
        }
        #self._stencil = {
        #    0: 1/(8*sp.pi*(n + 1)*(n + 2)),
        #    2: -1/(2*sp.pi*(n + 1)*(n + 5)),
        #    4: 3*(n + 4)/(4*sp.pi*(n + 2)*(n + 5)*(n + 6)),
        #    6: -1/(2*sp.pi*(n + 5)*(n + 7)),
        #    8: (n + 3)/(8*sp.pi*(n + 5)*(n + 6)*(n + 7))
        #}
        #self._stencil = {
        #    0: 1/(8*sp.pi*(n + 1)*(n + 2)*(n + 3)*(n + 4)),
        #    2: -1/(2*sp.pi*(n + 1)*(n + 3)*(n + 4)*(n + 5)),
        #    4: 3/(4*sp.pi*(n + 2)*(n + 3)*(n + 5)*(n + 6)),
        #    6: -1/(2*sp.pi*(n + 3)*(n + 4)*(n + 5)*(n + 7)),
        #    8: 1/(8*sp.pi*(n + 4)*(n + 5)*(n + 6)*(n + 7))
        #}

    @staticmethod
    def boundary_condition():
        return '8th order'

    @staticmethod
    def short_name():
        return 'C4'

class Generic(CompositeBase):
    r"""Function space for space with any boundary conditions

    Any combination of Dirichlet and Neumann is possible.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss

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
    def __init__(self, N, quad="GC", bc={}, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        from shenfun.utilities.findbasis import get_stencil_matrix
        self._stencil = get_stencil_matrix(bc, 'chebyshev', -half, -half, cn)
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
        return 'GT'
