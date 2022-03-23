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
    islicedict, slicedict, getCompositeBase, BoundaryConditions
from shenfun.matrixbase import SparseMatrix
from shenfun.config import config
from shenfun.jacobi.recursions import n

#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

__all__ = ['Orthogonal',
           'Phi1',
           'Phi2',
           'Phi3',
           'Phi4',
           'CompactDirichlet',
           'CompactNeumann',
           'CompositeBase',
           'BCBase',
           'BCGeneric']

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

class Orthogonal(SpectralBase):
    """Function space for Chebyshev series of second kind

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

    def __init__(self, N, quad='GU', domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None, **kw):
        SpectralBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        assert quad in ('GU', 'GC')
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
                import quadpy
                p = quadpy.c1.clenshaw_curtis(N)
                points = -p.points
                weights = p.weights

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

    def reference_domain(self):
        return (-1, 1)

    def get_orthogonal(self):
        return self

    def sympy_basis(self, i=0, x=xp):
        return sp.chebyshevu(i, x)

    @staticmethod
    def bnd_values(k=0, **kw):
        from shenfun.jacobi.recursions import bnd_values, un, half
        return bnd_values(half, half, k=k, gn=un)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_chebyu(i, x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

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

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            return

        if self.quad == 'GU':
            self.scalar_product._input_array *= self.broadcast_to_ndims(np.sin(np.pi/(self.N+1)*(np.arange(1, self.N+1))))
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*(self.N+1)*self.padding_factor*self.domain_factor()))

        elif self.quad == 'GC':
            self.scalar_product._input_array *= self.broadcast_to_ndims(self._sinGC)
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*self.N*self.padding_factor*self.domain_factor()))

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
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

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCGeneric(self.N, bc=self.bcs, domain=self.domain)
        return self._bc_basis

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
    domain : 2-tuple of floats, optional
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
    def __init__(self, N, quad="GU", bc=(0., 0.), domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, scaled=False, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates, scaled=scaled)
        #self.b0n = sp.simplify(b(half, half, n+1, n, un) / (h(half, half, n, 0, un)))
        #self.b2n = sp.simplify(b(half, half, n+1, n+2, un) / (h(half, half, n+2, 0, un)))
        self.b0n = 1/(np.pi*(n+1))
        self.b2n = -1/(np.pi*(n+3))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = sp.lambdify(n, self.b0n)(k[:N-2])
        d2[:] = sp.lambdify(n, self.b2n)(k[:N-2])
        if self.is_scaled():
            return SparseMatrix({0: d0/(k+2), 2: d2/(k[:-2]+2)}, (N, N))
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

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
    def __init__(self, N, quad="GU", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self.b0n = sp.simplify(matpow(b, 2, half, half, n+2, n, un) / (h(half, half, n, 0, un)))
        #self.b2n = sp.simplify(matpow(b, 2, half, half, n+2, n+2, un) / (h(half, half, n+2, 0, un)))
        #self.b4n = sp.simplify(matpow(b, 2, half, half, n+2, n+4, un) / (h(half, half, n+4, 0, un)))
        self.b0n = 1/(2*np.pi*(n+1)*(n+2))
        self.b2n = -1/(np.pi*(n+2)*(n+4))
        self.b4n = 1/(2*np.pi*(n+4)*(n+5))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2, d4 = np.zeros(N), np.zeros(N-2), np.zeros(N-4)
        d0[:-4] = sp.lambdify(n, self.b0n)(k[:N-4])
        d2[:-2] = sp.lambdify(n, self.b2n)(k[:N-4])
        d4[:] = sp.lambdify(n, self.b4n)(k[:N-4])
        return SparseMatrix({0: d0, 2: d2, 4: d4}, (N, N))

    def slice(self):
        return slice(0, self.N-4)

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
    def __init__(self, N, quad="GU", bc=(0,)*6, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self.b0n = sp.simplify(matpow(b, 3, half, half, n+3, n, un) / (h(half, half, n, 0, un)))
        #self.b2n = sp.simplify(matpow(b, 3, half, half, n+3, n+2, un) / (h(half, half, n+2, 0, un)))
        #self.b4n = sp.simplify(matpow(b, 3, half, half, n+3, n+4, un) / (h(half, half, n+4, 0, un)))
        #self.b6n = sp.simplify(matpow(b, 3, half, half, n+3, n+6, un) / (h(half, half, n+6, 0, un)))
        self.b0n = 1/(4*sp.pi*(n+1)*(n+2)*(n+3))
        self.b2n = -3/(4*sp.pi*(n+2)*(n+3)*(n+5))
        self.b4n = 3/(4*sp.pi*(n+3)*(n+5)*(n+6))
        self.b6n = -1/(4*sp.pi*(n+5)*(n+6)*(n+7))

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'P3'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2, d4, d6 = np.zeros(N), np.zeros(N-2), np.zeros(N-4), np.zeros(N-6)
        d0[:-6] = sp.lambdify(n, self.b0n)(k[:N-6])
        d2[:-4] = sp.lambdify(n, self.b2n)(k[:N-6])
        d4[:-2] = sp.lambdify(n, self.b4n)(k[:N-6])
        d6[:] = sp.lambdify(n, self.b6n)(k[:N-6])
        return SparseMatrix({0: d0, 2: d2, 4: d4, 6: d6}, (N, N))

    def slice(self):
        return slice(0, self.N-6)

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
    def __init__(self, N, quad="GC", bc=(0,)*8, domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        CompositeBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        #self.b0n = sp.simplify(matpow(b, 4, -half, -half, n+4, n, cn) / h(-half, -half, n, 0, cn))
        #self.b2n = sp.simplify(matpow(b, 4, -half, -half, n+4, n+2, cn) / h(-half, -half, n+2, 0, cn))
        #self.b4n = sp.simplify(matpow(b, 4, -half, -half, n+4, n+4, cn) / h(-half, -half, n+4, 0, cn))
        #self.b6n = sp.simplify(matpow(b, 4, -half, -half, n+4, n+6, cn) / h(-half, -half, n+6, 0, cn))
        #self.b8n = sp.simplify(matpow(b, 4, -half, -half, n+4, n+8, cn) / h(-half, -half, n+8, 0, cn))
        # Below are the same but faster since already simplified
        self.b0n = 1/(8*sp.pi*(n + 1)*(n + 2)*(n + 3)*(n + 4))
        self.b2n = -1/(2*sp.pi*(n + 1)*(n + 3)*(n + 4)*(n + 5))
        self.b4n = 3/(4*sp.pi*(n + 2)*(n + 3)*(n + 5)*(n + 6))
        self.b6n = -1/(2*sp.pi*(n + 3)*(n + 4)*(n + 5)*(n + 7))
        self.b8n = 1/(8*sp.pi*(n + 4)*(n + 5)*(n + 6)*(n + 7))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic*2'

    @staticmethod
    def short_name():
        return 'P4'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2, d4, d6, d8 = np.zeros(N), np.zeros(N-2), np.zeros(N-4), np.zeros(N-6), np.zeros(N-8)
        d0[:-8] = sp.lambdify(n, self.b0n)(k[:N-8])
        d2[:-6] = sp.lambdify(n, self.b2n)(k[:N-8])
        d4[:-4] = sp.lambdify(n, self.b4n)(k[:N-8])
        d6[:-2] = sp.lambdify(n, self.b6n)(k[:N-8])
        d8[:] = sp.lambdify(n, self.b8n)(k[:N-8])
        return SparseMatrix({0: d0, 2: d2, 4: d4, 6: d6, 8: d8}, (N, N))

    def slice(self):
        return slice(0, self.N-8)


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
    domain : 2-tuple of floats, optional
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

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'CD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        if not self.is_scaled():
            d0[:-2] = 1
            d2[:] = -(k[:-2]+1)/(k[:-2]+3)
        else:
            d0[:-2] = (k[:-2]+3)/(k[:-2]+1)
            d2[:] = -1
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)


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
    def __init__(self, N, quad="GU", bc=(0, 0), domain=(-1, 1), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        if isinstance(bc, (tuple, list)):
            bc = BoundaryConditions({'left': {'N': bc[0]}, 'right': {'N': bc[1]}})
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
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = 1
        d2[:] = -k[:-2]*(k[:-2]+1)/((k[:-2]+3)*(k[:-2]+4))
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
    def __init__(self, N, quad="GC", bc={}, domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        from shenfun.utilities.findbasis import get_stencil_matrix
        self._stencil = get_stencil_matrix(bc, 'chebyshevu')
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

    """
    def __init__(self, N, bc=None, domain=(-1, 1), **kw):
        CompositeBase.__init__(self, N, bc=bc, domain=domain)
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
        return chebvanderU(x, self.num_T-1)

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([sp.chebyshevu(j, x) for j in range(self.num_T)]))

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

class BCDirichlet(BCBase):
    r"""Basis for inhomogeneous Dirichlet boundary conditions

    .. math::

        \phi_0 &= \frac{1}{2}U_0 - \frac{1}{4}U_1 \\
        \phi_1 &= \frac{1}{2}U_0 + \frac{1}{4}U_1
    """
    @staticmethod
    def short_name():
        return 'BCD'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 4)*np.array([[2, -1],
                                           [2, 1]])

class BCNeumann(BCBase):
    r"""Basis for inhomogeneous Neumann boundary conditions

    .. math::

        \phi_0 &= \frac{1}{16}(4U_1 - U_2) \\
        \phi_1 &= \frac{1}{16}(4U_1 + U_2)
    """

    @staticmethod
    def short_name():
        return 'BCN'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 16)*np.array([[0, 4, -1],
                                            [0, 4, 1]])

class BCBiharmonic(BCBase):
    r"""Basis for inhomogeneous Biharmonic boundary conditions

    .. math::

        \phi_0 &= \frac{1}{32}(16U_0 - 10U_1 + U_3 \\
        \phi_1 &= \frac{1}{32}(6U_0 - 2U_1 - 2U_2 + U_3) \\
        \phi_2 &= \frac{1}{32}(16U_0 + 10U_1 - U_3) \\
        \phi_3 &= \frac{1}{32}(-6U_0 - 2U_1 + 2U_2 + U_3)
    """

    @staticmethod
    def short_name():
        return 'BCB'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 32)*np.array([[16, -10, 0, 1],
                                            [6, -2, -2, 1],
                                            [16, 10, 0, -1],
                                            [-6, -2, 2, 1]])

class BCGeneric(BCBase):

    @staticmethod
    def short_name():
        return 'BG'

    def stencil_matrix(self, N=None):
        if self._stencil_matrix is None:
            from shenfun.utilities import get_bc_basis
            self._stencil_matrix = np.array(get_bc_basis(self.bcs, 'chebyshevu'))
        return self._stencil_matrix

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
