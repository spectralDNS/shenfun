"""
Module for defining function spaces in the Chebyshev family
"""
from __future__ import division
import functools
import numpy as np
from numpy.polynomial import chebyshev as n_cheb
import sympy as sp
from scipy.special import eval_chebyu
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, FuncWrap, \
    islicedict, slicedict, getCompositeBase, getBCGeneric, BoundaryConditions
from shenfun.matrixbase import SparseMatrix
from shenfun.config import config
from shenfun.jacobi.recursions import half, un, n
from shenfun.jacobi import JacobiBase

#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

bases = ['Orthogonal',
         'CompactDirichlet',
         'CompactNeumann',
         'UpperDirichlet',
         'LowerDirichlet',
         'CompactBiharmonic',
         'Compact3',
         'Generic']
bcbases = ['BCGeneric']
testbases = ['Phi1', 'Phi2', 'Phi3', 'Phi4', 'Phi6']

__all__ = bases + bcbases + testbases

xp = sp.Symbol('x', real=True)

class DCTWrap(FuncWrap):
    """DCT for complex input"""

    @property
    def dct(self):
        return object.__getattribute__(self, '_func')

    def __call__(self, input_array=None, output_array=None, **kw):
        dct_obj = self.dct

        if input_array is not None:
            self.input_array[...] = input_array

        dct_obj.input_array[...] = self.input_array.real
        dct_obj(None, None, **kw)
        self.output_array.real[...] = dct_obj.output_array
        dct_obj.input_array[...] = self.input_array.imag
        dct_obj(None, None, **kw)
        self.output_array.imag[...] = dct_obj.output_array

        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        return self.output_array

class Orthogonal(JacobiBase):
    r"""Function space for Chebyshev series of second kind

    The orthogonal basis is

    .. math::

        U_k, \quad k = 0, 1, \ldots, N-1,

    where :math:`U_k` is the :math:`k`'th Chebyshev polynomial of the second
    kind.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
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

    def __init__(self, N, quad='GU', domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        JacobiBase.__init__(self, N, quad=quad, alpha=half, beta=half, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        assert quad in ('GU', 'GC')
        self.gn = un
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=2)
            self._xfftn_bck = functools.partial(fftw.dstn, type=3)
            self._sinGC = np.sin((np.arange(N)+0.5)*np.pi/N)

        elif quad == 'GU':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=1)
            self._xfftn_bck = functools.partial(fftw.dstn, type=1)
        self._xfftn_fwd.opts = self._xfftn_bck.opts = config['fftw']['dst']
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def family():
        return 'chebyshevu'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)

        if weighted:
            if self.quad == "GC":
                points, weights = n_cheb.chebgauss(N)
                points = points.astype(float)
                weights = (weights*(1-points**2)).astype(float)

            elif self.quad == "GU":
                points = np.cos((np.arange(N)+1)*np.pi/(N+1))
                weights = np.full(N, np.pi/(N+1))*(1-points**2)

        else:
            if self.quad == "GU":
                theta = (np.arange(N)+1)*np.pi/(N+1)
                points = np.cos(theta)
                d = fftw.aligned(N, fill=0)
                k = np.arange(N)
                d[::2] = 2/(k[::2]+1)
                w = fftw.aligned_like(d)
                dst = fftw.dstn(w, axes=(0,), type=1)
                weights = dst(d, w)
                weights *= (np.sin(theta))/(N+1)

            elif self.quad == "GC":
                points = n_cheb.chebgauss(N)[0]
                d = fftw.aligned(N, fill=0)
                k = 2*(1 + np.arange((N-1)//2))
                d[::2] = (2./N)/np.hstack((1., 1.-k*k))
                w = fftw.aligned_like(d)
                dct = fftw.dctn(w, axes=(0,), type=3)
                weights = dct(d, w)

        if map_true_domain is True:
            points = self.map_true_domain(points)

        return points, weights

    def vandermonde(self, x):
        return chebvanderU(x, self.shape(False)-1)

    def weight(self, x=xp):
        return sp.sqrt(1-x**2)

    def get_orthogonal(self, **kwargs):
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return Orthogonal(self.N, **d)

    def basis_function(self, i=0, x=xp):
        return self.orthogonal_basis_function(i=i, x=x)

    def orthogonal_basis_function(self, i=0, x=xp):
        return sp.chebyshevu(i, x)

    def L2_norm_sq(self, i):
        return sp.pi/2

    def l2_norm_sq(self, i=None):
        if i is None:
            f = np.full(self.N, np.pi/2)
            if self.quad == 'GC':
                f[-1] *= 2
            return f
        elif i == self.N-1 and self.quad == 'GC':
            return np.pi
        return np.pi/2

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_chebyu(i, x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        # Implementing using U_k = \frac{1}{k+1}T^{'}_{k+1}
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True)+1)
        basis[i+1] = 1
        basis = n_cheb.Chebyshev(basis)
        k += 1
        basis = basis.deriv(k)
        output_array[:] = basis(x)/(i+1)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        # Implementing using  U_k = \frac{1}{k+1}T^{'}_{k+1}
        if x is None:
            x = self.mesh(False, False)
        V = n_cheb.chebvander(x, self.shape(False))
        M = V.shape[1]
        D = np.zeros((M, M))
        k = k+1
        D[slice(0, M-k)] = n_cheb.chebder(np.eye(M, M), k)
        V = np.dot(V, D)
        i = np.arange(1, M)[None, :]
        return V[:, 1:]/i

    def _evaluate_scalar_product(self, kind='fast'):
        if kind != 'fast':
            SpectralBase._evaluate_scalar_product(self, kind=kind)
            return

        if self.quad == 'GU':
            self.scalar_product._input_array *= self.broadcast_to_ndims(np.sin(np.pi/(self.N+1)*(np.arange(1, self.N+1))))
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*(self.N+1)*self.padding_factor*self.domain_factor()))

        elif self.quad == 'GC':
            self.scalar_product._input_array *= self.broadcast_to_ndims(self._sinGC)
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*self.N*self.padding_factor*self.domain_factor()))

    def _evaluate_expansion_all(self, input_array, output_array, x=None, kind='fast'):
        if kind != 'fast':
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, kind=kind)
            return

        # Fast transform. Make sure arrays are correct. Fast transforms are only for planned size
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        output_array = self.backward.xfftn()

        if self.quad == "GU":
            output_array *= 1/(2*self.broadcast_to_ndims(np.sin((np.arange(self.N)+1)*np.pi/(self.N+1))))

        elif self.quad == "GC":
            s0 = self.sl[slice(self.N-1, self.N)]
            se = self.sl[slice(0, self.N, 2)]
            so = self.sl[slice(1, self.N, 2)]
            output_array *= 0.5
            output_array[se] += input_array[s0]/2
            output_array[so] -= input_array[s0]/2
            output_array *= 1/(self.broadcast_to_ndims(np.sin((np.arange(self.N)+0.5)*np.pi/self.N)))

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'U'

    def to_ortho(self, input_array, output_array=None):
        assert input_array.__class__.__name__ == 'Orthogonal'
        if output_array:
            output_array[:] = input_array
            return output_array
        return input_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.output_array.dtype)
        x = self.map_reference_domain(x)
        output_array[:] = np.dot(chebvanderU(x, self.N-1), u)
        return output_array

    def stencil_matrix(self, N=None):
        return SparseMatrix({0: 1}, (N, N))

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

CompositeBase = getCompositeBase(Orthogonal)
BCGeneric = getBCGeneric(CompositeBase)

class Phi1(CompositeBase):
    r"""Function space for Dirichlet boundary conditions

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= \frac{1}{\pi}\left(\frac{U_k}{k+1}-\frac{U_{k+2}}{k+3}\right) = \frac{(1-x^2)U'_{k+1}}{h^{(1)}_{k+1}} \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \tfrac{1}{2}U_0 - \tfrac{1}{4}U_1, \\
        \phi_{N-1} &= \tfrac{1}{2}U_0 + \tfrac{1}{4}U_1,

    where :math:`h^{(1)}_n = \frac{\pi n(n+2)}{2}`. We have

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

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 2-tuple of floats, optional
        Boundary conditions at, respectively, x=(-1, 1).
    domain : 2-tuple of numbers, optional
        The computational domain
    scaled : boolean, optional
        Whether or not to scale basis function n by 1/(n+2)
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
    def __init__(self, N, quad="GU", bc=(0., 0.), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, scaled=False, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates, scaled=scaled)
        #self._stencil = {
        #   0: sp.simplify(b(half, half, n+1, n, un) / (h(half, half, n, 0, un))),
        #   2: sp.simplify(b(half, half, n+1, n+2, un) / (h(half, half, n+2, 0, un)))}
        self._stencil = {
            0: 1/(np.pi*(n+1)),
            2: -1/(np.pi*(n+3))
        }
        if self.is_scaled():
            self._stencil = {
                0: 1/(sp.pi*(n+1)*(n+2)),
                2: -1/(sp.pi*(n+3)*(n+2))
            }

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'


class Phi2(CompositeBase):
    r"""Function space for biharmonic equation.

    The basis functions :math:`\phi_k` for :math:`k=0, \ldots, N-5` are

    .. math::

        \phi_k &= \frac{(1-x^2)^2 U''_{k+2}}{h^{(2)}_{k+2}} \\
               &= \frac{1}{2\pi(k+1)(k+2)}\left(U_k- \frac{2(k+1)}{k+4}U_{k+2} + \frac{(k+1)(k+2)}{(k+4)(k+5)}U_{k+4} \right)

    where :math:`h^{(2)}_n = \frac{\pi (n+3)(n+2)n(n-1)}{2}`. The 4 boundary
    functions are

    .. math::
        \phi_{N-4} &= \tfrac{1}{2}U_0-\tfrac{5}{6}U_1+\tfrac{1}{32}U_3, \\
        \phi_{N-3} &= \tfrac{3}{16}U_0-\tfrac{1}{16}U_1-\tfrac{1}{16}U_2+\tfrac{1}{32}U_3, \\
        \phi_{N-2} &= \tfrac{1}{2}U_0+\tfrac{5}{16}U_1-\tfrac{1}{32}U_3), \\
        \phi_{N-1} &= -\tfrac{3}{16}U_0-\tfrac{1}{16}U_1+\tfrac{1}{16}U_2+\tfrac{1}{32}U_3,

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

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 4-tuple of floats, optional
        2 boundary conditions at, respectively, x=-1 and x=1.
    domain : 2-tuple of numbers, optional
        The computational domain
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    scaled : boolean, optional
        Whether or not to scale basis function n by 1/(n+3)
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
         Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """
    def __init__(self, N, quad="GU", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, scaled=False, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               scaled=scaled, coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 2, half, half, n+2, n, un) / (h(half, half, n, 0, un))),
        #   2: sp.simplify(matpow(b, 2, half, half, n+2, n+2, un) / (h(half, half, n+2, 0, un))),
        #   4: sp.simplify(matpow(b, 2, half, half, n+2, n+4, un) / (h(half, half, n+4, 0, un)))}
        self._stencil = {
            0: 1/(2*sp.pi*(n+1)*(n+2)),
            2: -1/(sp.pi*(n+2)*(n+4)),
            4: 1/(2*sp.pi*(n+4)*(n+5))
        }
        if self.is_scaled():
            self._stencil = {
                0: 1/(2*sp.pi*(n+1)*(n+2)*(n+3)),
                2: -1/(sp.pi*(n+2)*(n+4)*(n+3)),
                4: 1/(2*sp.pi*(n+4)*(n+5)*(n+3))
            }


    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'


class Phi3(CompositeBase):
    r"""Function space for 6'th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, \ldots, N-7` are

    .. math::

        \phi_k &= \frac{(1-x^2)^3}{h^{(3)}_{k+3}} U^{(3)}_{k+3} \\
        h^{(3)}_{k+3} &= \frac{\pi \Gamma (k+8)}{2(k+4)k!} = \int_{-1}^1 U^{(3)}_k U^{(3)}_k (1-x^2)^{3.5}} dx.

    where :math:`U^{(3)}_k` is the 3rd derivative of :math:`U_k`. The boundary
    basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u(1)=d u'(1)=e, u''(1)=f.

    The last 6 basis functions are only used if there are nonzero boundary
    conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 6-tuple of floats, optional
        3 boundary conditions at, respectively, x=-1 and x=1.
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
    def __init__(self, N, quad="GU", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 3, half, half, n+3, n, un) / (h(half, half, n, 0, un))),
        #   2: sp.simplify(matpow(b, 3, half, half, n+3, n+2, un) / (h(half, half, n+2, 0, un))),
        #   4: sp.simplify(matpow(b, 3, half, half, n+3, n+4, un) / (h(half, half, n+4, 0, un))),
        #   6: sp.simplify(matpow(b, 3, half, half, n+3, n+6, un) / (h(half, half, n+6, 0, un)))}
        self._stencil = {
            0: 1/(4*sp.pi*(n+1)*(n+2)*(n+3)),
            2: -3/(4*sp.pi*(n+2)*(n+3)*(n+5)),
            4: 3/(4*sp.pi*(n+3)*(n+5)*(n+6)),
            6: -1/(4*sp.pi*(n+5)*(n+6)*(n+7))
        }

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'P3'


class Phi4(CompositeBase):
    r"""Function space for 8th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, \ldots, N-9` are

    .. math::

        \phi_k &= \frac{(1-x^2)^4}{h^{(4)}_{k+4}} U^{(4)}_{k+4} \\
        h^{(4)}_{k+4} &= \frac{\pi \Gamma (k+10)}{2(k+5)k!} = \int_{-1}^1 U^{(4)}_{k+4} U^{(4)}_{k+4} (1-x^2)^{4.5}} dx.

    where :math:`U^{(4)}_k` is the 4th derivative of :math:`U_k`. The boundary
    basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u'''(-1)=d, u(1)=e u'(1)=f, u''(1)=g, u'''(1)=h

    The last 8 basis functions are only used if there are nonzero boundary
    conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
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
    def __init__(self, N, quad="GU", bc=(0,)*8, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 4, half, half, n+4, n, un) / h(half, half, n, 0, un),
        #   2: sp.simplify(matpow(b, 4, half, half, n+4, n+2, un) / h(half, half, n+2, 0, un)),
        #   4: sp.simplify(matpow(b, 4, half, half, n+4, n+4, un) / h(half, half, n+4, 0, un)),
        #   6: sp.simplify(matpow(b, 4, half, half, n+4, n+6, un) / h(half, half, n+6, 0, un)),
        #   8: sp.simplify(matpow(b, 4, half, half, n+4, n+8, un) / h(half, half, n+8, 0, un))}
        # Below are the same but faster since already simplified
        self._stencil = {
            0: 1/(8*sp.pi*(n + 1)*(n + 2)*(n + 3)*(n + 4)),
            2: -1/(2*sp.pi*(n + 2)*(n + 3)*(n + 4)*(n + 6)),
            4: 3/(4*sp.pi*(n + 3)*(n + 4)*(n + 6)*(n + 7)),
            6: -1/(2*sp.pi*(n + 4)*(n + 6)*(n + 7)*(n + 8)),
            8: 1/(8*sp.pi*(n + 6)*(n + 7)*(n + 8)*(n + 9))
        }

    @staticmethod
    def boundary_condition():
        return 'Biharmonic*2'

    @staticmethod
    def short_name():
        return 'P4'

class Phi6(CompositeBase):
    r"""Function space for 12th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, \ldots, N-13` are

    .. math::

        \phi_k &= \frac{(1-x^2)^6}{h^{(6)}_{k+6}} U^{(6)}_{k+6} \\
        h^{(6)}_k &= \frac{\pi (k+7)!}{2(k-6)!} = \int_{-1}^1 U^{(6)}_k U^{(6)}_k (1-x^2)^{6.5} dx,

    where :math:`U^{(6)}_k` is the 6th derivative of :math:`U_k`. The 12 boundary
    basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u^{(k)}(-1)&=a_k, u^{(k)}(1)=b_k, k=0, 1, \ldots, 5

    The last 12 basis functions are only used if there are nonzero boundary
    conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
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
    def __init__(self, N, quad="GU", bc=(0,)*12, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self._stencil = {
        #   0: sp.simplify(matpow(b, 6, half, half, n+6, n, un) / h(half, half, n, 0, un)),
        #   2: sp.simplify(matpow(b, 6, half, half, n+6, n+2, un) / h(half, half, n+2, 0, un)),
        #   4: sp.simplify(matpow(b, 6, half, half, n+6, n+4, un) / h(half, half, n+4, 0, un)),
        #   6: sp.simplify(matpow(b, 6, half, half, n+6, n+6, un) / h(half, half, n+6, 0, un)),
        #   8: sp.simplify(matpow(b, 6, half, half, n+6, n+8, un) / h(half, half, n+8, 0, un)),
        #  10: sp.simplify(matpow(b, 6, half, half, n+6, n+10, un) / h(half, half, n+10, 0, un)),
        #  12: sp.simplify(matpow(b, 6, half, half, n+6, n+12, un) / h(half, half, n+12, 0, un))}
        # Below are the same but faster since already simplified
        self._stencil = {
            0: 1/(32*sp.pi*(n + 1)*(n + 2)*(n + 3)*(n + 4)*(n + 5)*(n + 6)),
            2: -3/(16*sp.pi*(n + 2)*(n + 3)*(n + 4)*(n + 5)*(n + 6)*(n + 8)),
            4: 15/(32*sp.pi*(n + 3)*(n + 4)*(n + 5)*(n + 6)*(n + 8)*(n + 9)),
            6: -5/(8*sp.pi*(n + 4)*(n + 5)*(n + 6)*(n + 8)*(n + 9)*(n + 10)),
            8: 15/(32*sp.pi*(n + 5)*(n + 6)*(n + 8)*(n + 9)*(n + 10)*(n + 11)),
            10: -3/(16*sp.pi*(n + 6)*(n + 8)*(n + 9)*(n + 10)*(n + 11)*(n + 12)),
            12: 1/(32*sp.pi*(n + 8)*(n + 9)*(n + 10)*(n + 11)*(n + 12)*(n + 13))
        }

    @staticmethod
    def boundary_condition():
        return '12th order'

    @staticmethod
    def short_name():
        return 'P6'


class CompactDirichlet(CompositeBase):
    r"""Function space for Dirichlet boundary conditions.

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= U_k - \frac{k+1}{k+3}U_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \tfrac{1}{2}U_0 - \tfrac{1}{4}U_1, \\
        \phi_{N-1} &= \tfrac{1}{2}U_0 + \tfrac{1}{4}U_1,

    such that

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a \text{ and } u(1) = b.

    The last two bases are for boundary conditions and only used if a or b are
    different from 0. In one dimension :math:`\hat{u}_{N-2}=a` and
    :math:`\hat{u}_{N-1}=b`.

    If the parameter `scaled=True`, then the first N-2 basis functions are

    .. math::

        \phi_k &= \frac{U_k}{k+1} - \frac{U_{k+2}}{k+3}, \, k=0, 1, \ldots, N-3, \\

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 2-tuple of floats, optional
        Boundary conditions at, respectively, x=(-1, 1).
    domain : 2-tuple of numbers, optional
        The computational domain
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    scaled : boolean, optional
        Whether or not to use scaled basis function.
    padding_factor : float, optional
        Factor for padding backward transforms.
    dealias_direct : bool, optional
        Set upper 1/3 of coefficients to zero before backward transform
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
         Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    Note
    ----
    This basis function is a scaled version of :class:`Phi1`.

    """
    def __init__(self, N, quad="GU", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None,
                 scaled= False, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates, scaled=scaled)
        self._stencil = {0: 1, 2: -(n+1)/(n+3)}
        if self.is_scaled():
            self._stencil = {0: 1/(n+1), 2: -1/(n+3)}

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

        \phi_k = {U_k}-\frac{k(k+1)}{(k+3)(k+4)} U_{k+2}, \, k=0, 1, \ldots, N-3, \\
        \phi_{N-2} &= \tfrac{1}{16}(4U_1-U_2), \\
        \phi_{N-1} &= \tfrac{1}{16}(4U_1+U_2),

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

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
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
    def __init__(self, N, quad="GU", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}})
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 2: -n*(n+1)/((n+3)*(n+4))}

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'CN'


class UpperDirichlet(CompositeBase):
    r"""Function space with single Dirichlet on upper edge

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= U_{k} - \frac{k+1}{k+2} U_{k+1}, \, k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= U_0,

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

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 2-tuple of (None, number), optional
        Boundary condition at x=1.
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
    def __init__(self, N, quad="GU", bc=(None, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 1: -(n+1)/(n+2)}

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'

class LowerDirichlet(CompositeBase):
    r"""Function space with single Dirichlet on left edge

    The basis :math:`\{\phi_k\}_{k=0}^{N-1}` is

    .. math::

        \phi_k &= U_{k} + \frac{k+1}{k+2} U_{k+1}, \, k=0, 1, \ldots, N-2, \\
        \phi_{N-1} &= U_0,

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

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 2-tuple of (number, None), optional
        Boundary condition at x=-1.
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
    def __init__(self, N, quad="GU", bc=(None, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {0: 1, 1: (n+1)/(n+2)}

    @staticmethod
    def boundary_condition():
        return 'LowerDirichlet'

    @staticmethod
    def short_name():
        return 'LD'

class CompactBiharmonic(CompositeBase):
    r"""Function space for biharmonic equation.

    The basis functions :math:`\phi_k` for :math:`k=0, \ldots, N-5` are

    .. math::

        \phi_k &= \frac{h_k}{b^{(2)}_{k+2,k}}\frac{(1-x^2)^2 U''_{k+2}}{h^{(2)}_{k+2}} \\
               &= U_k - \frac{2k+2}{k+4}U_{k+2} + \frac{(k+1)(k+2)}{(k+4)(k+5)}U_{k+4}

    where :math:`h^{(2)}_k = \frac{\pi (k+3)(k+2)k(k-1)}{2}`, :math:`h_k=\pi/2` and
    :math:`b^{(2)}_{k+2, k}= \frac{1}{4(k+1)(k+2)}`. The 4 boundary
    functions are

    .. math::
        \phi_{N-4} &= \tfrac{1}{2}U_0-\tfrac{5}{6}U_1+\tfrac{1}{32}U_3, \\
        \phi_{N-3} &= \tfrac{3}{16}U_0-\tfrac{1}{16}U_1-\tfrac{1}{16}U_2+\tfrac{1}{32}U_3, \\
        \phi_{N-2} &= \tfrac{1}{2}U_0+\tfrac{5}{16}U_1-\tfrac{1}{32}U_3), \\
        \phi_{N-1} &= -\tfrac{3}{16}U_0-\tfrac{1}{16}U_1+\tfrac{1}{16}U_2+\tfrac{1}{32}U_3,

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

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 4-tuple of floats, optional
        2 boundary conditions at, respectively, x=-1 and x=1.
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
    def __init__(self, N, quad="GU", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, scaled=False, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               scaled=scaled, coordinates=coordinates)
        #self._stencil = {
        #   0: 1,
        #   2: sp.simplify(matpow(b, 2, half, half, n+2, n+2, un) / matpow(b, 2, half, half, n+2, n, un) * h(half, half, n, 0, un) / h(half, half, n+2, 0, un)),
        #   4: sp.simplify(matpow(b, 2, half, half, n+2, n+4, un) / matpow(b, 2, half, half, n+2, n, un) * h(half, half, n, 0, un) / h(half, half, n+4, 0, un))}
        self._stencil = {
            0: 1,
            2: -(2*n + 2)/(n + 4),
            4: (n + 1)*(n + 2)/((n + 4)*(n + 5))
        }

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'C2'


class Compact3(CompositeBase):
    r"""Function space for 6'th order equation

    The basis functions :math:`\phi_k` for :math:`k=0, \ldots, N-7` are

    .. math::

        \phi_k &= U_k - \frac{3(n+1)}{n+5}U_{k+2} + \frac{3(n+1)(n+2)}{(n+5)(n+6)}U_{k+4} - \frac{(n+1)(n+2)(n+3)}{(n+5)(n+6)(n+7)}U_{k+6}

    This is the same basis as :class:`.Phi3`, only scaled such that the main diagonal of
    the stencil matrix contains ones.

    The boundary basis for inhomogeneous boundary conditions is too messy to print, but can
    be obtained using :func:`~shenfun.utilities.findbasis.get_bc_basis`. We have

    .. math::
        u(x) &= \sum_{k=0}^{N-1} \hat{u}_k \phi_k(x), \\
        u(-1) &= a, u'(-1)=b, u''(-1)=c, u(1)=d u'(1)=e, u''(1)=f.

    The last 6 basis functions are only used if there are nonzero boundary
    conditions.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    bc : 6-tuple of floats, optional
        3 boundary conditions at, respectively, x=-1 and x=1.
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
    def __init__(self, N, quad="GU", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self._stencil = {
            0: 1,
            2: -3*(n+1)/(n+5),
            4: 3*(n+1)*(n+2)/((n+5)*(n+6)),
            6: -(n+1)*(n+2)*(n+3)/((n+5)*(n+6)*(n+7))
        }

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'C3'

class Generic(CompositeBase):
    r"""Function space for space with any boundary conditions

    Any combination of Dirichlet and Neumann is possible.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points
    quad : str, optional
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
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

    """
    def __init__(self, N, quad="GU", bc={}, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        from shenfun.utilities.findbasis import get_stencil_matrix
        self._stencil = get_stencil_matrix(bc, 'chebyshevu', half, half, un)
        bc = BoundaryConditions(bc)
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Generic'

    @staticmethod
    def short_name():
        return 'GU'


def chebvanderU(x, deg):
    """Pseudo-Vandermonde matrix of given degree for Chebyshev polynomials
    of second kind.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = U_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Chebyshev polynomial.  The dtype will be the same as
        the converted `x`.

    """
    import numpy.polynomial.polyutils as pu
    ideg = pu._deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    # Use forward recursion to generate the entries.
    v[0] = x*0 + 1
    if ideg > 0:
        x2 = 2*x
        v[1] = 2*x
        for i in range(2, ideg + 1):
            v[i] = v[i-1]*x2 - v[i-2]
    return np.moveaxis(v, 0, -1)
