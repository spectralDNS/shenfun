"""
Module for defining bases in the Chebyshev family
"""
from __future__ import division
import functools
import numpy as np
import sympy as sp
from numpy.polynomial import chebyshev as n_cheb
from scipy.special import eval_chebyt
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, FuncWrap, \
    islicedict, slicedict
from shenfun.optimization.cython import Cheb
from shenfun.utilities import inheritdocstrings
from shenfun.forms.arguments import Function

__all__ = ['ChebyshevBase', 'Basis', 'ShenDirichletBasis',
           'ShenNeumannBasis', 'ShenBiharmonicBasis',
           'SecondNeumannBasis', 'UpperDirichletBasis',
           'ShenBiPolarBasis', 'BCBasis', 'BCBiharmonicBasis',
           'DirichletNeumannBasis']

#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

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


@inheritdocstrings
class ChebyshevBase(SpectralBase):
    """Abstract base class for all Chebyshev bases

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        assert quad in ('GC', 'GL')
        SpectralBase.__init__(self, N, quad=quad, dtype=dtype, padding_factor=padding_factor,
                              dealias_direct=dealias_direct, domain=domain, coordinates=coordinates)

    @staticmethod
    def family():
        return 'chebyshev'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.N

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
        else:
            if self.quad == "GL":
                import quadpy
                p = quadpy.line_segment.clenshaw_curtis(N)
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
        return n_cheb.chebvander(x, int(self.N*self.padding_factor)-1)

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        return sp.chebyshevt(i, x)

    def sympy_weight(self, x=sp.symbols('x', real=True)):
        return 1/sp.sqrt(1-x**2)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        #output_array[:] = np.cos(i*np.arccos(x))
        output_array[:] = eval_chebyt(i, x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        N, M = self.shape(False), self.shape(True)
        if k > 0:
            D = np.zeros((M, N))
            D[:-k] = n_cheb.chebder(np.eye(M, N), k)
            V = np.dot(V, D)
        return self._composite_basis(V, argument=argument)

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        return self._composite_basis(V, argument=argument)

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

    def _composite_basis(self, V, argument=0):
        """Return composite basis, where ``V`` is primary Vandermonde matrix."""
        return V

    def reference_domain(self):
        return (-1., 1.)

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        if 'builders' in self._xfftn_fwd.func.__module__: #pragma: no cover
            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)

            U = fftw.aligned(shape, dtype=np.float)
            xfftn_fwd = plan_fwd(U, axis=axis, **opts)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, axis=axis, **opts)
            V.fill(0)
            U.fill(0)

            xfftn_fwd.update_arrays(U, V)
            xfftn_bck.update_arrays(V, U)
        else: # fftw wrapped with mpi4py-fft
            opts = dict(
                overwrite_input='FFTW_DESTROY_INPUT',
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)
            flags = (fftw.flag_dict[opts['planner_effort']],
                     fftw.flag_dict[opts['overwrite_input']])
            threads = opts['threads']

            U = fftw.aligned(shape, dtype=np.float)

            xfftn_fwd = plan_fwd(U, axes=(axis,), threads=threads, flags=flags)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, axes=(axis,), threads=threads, flags=flags, output_array=U)
            V.fill(0)
            U.fill(0)

        if np.dtype(dtype) is np.dtype('complex'):
            # dct only works on real data, so need to wrap it
            U = fftw.aligned(shape, dtype=np.complex)
            V = fftw.aligned(shape, dtype=np.complex)
            U.fill(0)
            V.fill(0)
            xfftn_fwd = DCTWrap(xfftn_fwd, U, V)
            xfftn_bck = DCTWrap(xfftn_bck, V, U)

        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_orthogonal(self):
        return Basis(self.N, quad=self.quad, domain=self.domain, coordinates=self.coors.coordinates)


@inheritdocstrings
class Basis(ChebyshevBase):
    """Basis for regular Chebyshev series

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad='GC', domain=(-1., 1.), dtype=np.float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dctn, type=2)
            self._xfftn_bck = functools.partial(fftw.dctn, type=3)

        else:
            self._xfftn_fwd = functools.partial(fftw.dctn, type=1)
            self._xfftn_bck = functools.partial(fftw.dctn, type=1)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def derivative_coefficients(fk):
        """Return coefficients of Chebyshev series for c = f'(x)

        Parameters
        ----------
            fk : input array
                 Coefficients of regular Chebyshev series

        Returns
        -------
        array
            Coefficients of derivative of fk-series

        """
        ck = np.zeros_like(fk)
        if len(fk.shape) == 1:
            ck = Cheb.derivative_coefficients(fk, ck)
        elif len(fk.shape) == 3:
            ck = Cheb.derivative_coefficients_3D(fk, ck)
        return ck

    def fast_derivative(self, fj):
        """Return derivative of :math:`f_j = f(x_j)` at quadrature points
        :math:`x_j`

        Parameters
        ----------
            fj : input array
                 Function values on quadrature mesh

        Returns
        -------
        array
             Array with derivatives on quadrature mesh
        """
        fk = self.forward(fj)
        ck = self.derivative_coefficients(fk)
        fd = self.backward(ck)
        return fd.copy()

    def apply_inverse_mass(self, array):
        array *= (2/np.pi)
        array[self.si[0]] /= 2
        if self.quad == 'GL':
            array[self.si[-1]] /= 2
        return array

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
            return

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

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        #assert self.N == self.scalar_product.input_array.shape[self.axis]
        out = self.scalar_product.xfftn()
        if self.quad == "GC":
            out *= (np.pi/(2*self.N*self.padding_factor))

        elif self.quad == "GL":
            out *= (np.pi/(2*(self.N*self.padding_factor-1)))

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.output_array.dtype)
        x = self.map_reference_domain(x)
        output_array[:] = n_cheb.chebval(x, u)
        return output_array

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self):
        return self

@inheritdocstrings
class ShenDirichletBasis(ChebyshevBase):
    """Shen basis for Dirichlet boundary conditions

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
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    A test function is always using homogeneous boundary conditions.

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=np.float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._scaled = scaled
        self._factor = np.ones(1)
        self.bc = BoundaryValues(self, bc=bc)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite_basis(self, V, argument=0):
        P = np.zeros_like(V)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
        if argument == 1: # if trial function
            P[:, -1] = (V[:, 0] + V[:, 1])/2    # x = +1
            P[:, -2] = (V[:, 0] - V[:, 1])/2    # x = -1
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        assert i < self.N
        if i < self.N-2:
            return sp.chebyshevt(i, x) - sp.chebyshevt(i+2, x)
        if i == self.N-2:
            return 0.5*(1-x)
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            output_array[:] = np.cos(i*w) - np.cos((i+2)*w)
        elif i == self.N-2:
            output_array[:] = 0.5*(1-x)
        elif i == self.N-1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -1)
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)
        output_array[self.si[-2]] = 0
        output_array[self.si[-1]] = 0

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        output[s0] -= output[s1]
        output[self.si[-2]] = 0
        output[self.si[-1]] = 0

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self._padding_backward(input_array, self.backward.tmp_array)
            SpectralBase.evaluate_expansion_all(self, self.backward.tmp_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= input_array[s0]
        self.bc.add_to_orthogonal(w_hat, input_array) # Correct bc-values must be in input_array (as they should be before a backward transform)
        self.CT.backward(w_hat)
        assert output_array is self.CT.backward.output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[2:] = u[:-2]
        output_array -= n_cheb.chebval(x, w_hat)
        output_array += 0.5*(u[-1]*(1+x)+u[-2]*(1-x))
        return output_array

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.bc.add_mass_rhs(u)
        self.apply_inverse_mass(u)
        self._truncation_forward(u, self.forward.output_array)
        self.bc.set_boundary_dofs(self.forward.output_array, False)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.evaluate_expansion_all(self.backward.input_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_bc_basis(self):
        return BCBasis(self.N, quad=self.quad, domain=self.domain, coordinates=self.coors.coordinates)

    def get_refined(self, N):
        return ShenDirichletBasis(N,
                                  quad=self.quad,
                                  domain=self.domain,
                                  dtype=self.dtype,
                                  padding_factor=self.padding_factor,
                                  dealias_direct=self.dealias_direct,
                                  coordinates=self.coors.coordinates,
                                  bc=self.bc.bc,
                                  scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenDirichletBasis(self.N,
                                  quad=self.quad,
                                  domain=self.domain,
                                  dtype=self.dtype,
                                  padding_factor=padding_factor,
                                  dealias_direct=dealias_direct,
                                  coordinates=self.coors.coordinates,
                                  bc=self.bc.bc,
                                  scaled=self._scaled)

    def _truncation_forward(self, padded_array, trunc_array):
        if not id(trunc_array) == id(padded_array):
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            s = self.sl[slice(0, N-2)]
            trunc_array[s] = padded_array[s]
            s = self.sl[slice(-2, None)]
            trunc_array[s] = padded_array[s]

    def _padding_backward(self, trunc_array, padded_array):
        if not id(trunc_array) == id(padded_array):
            padded_array.fill(0)
            N = trunc_array.shape[self.axis]
            _sn = self.sl[slice(0, N-2)]
            padded_array[_sn] = trunc_array[_sn]
            _sn = self.sl[slice(N-2, N)]
            _sp = self.sl[slice(-2, None)]
            padded_array[_sp] = trunc_array[_sn]

        elif self.dealias_direct:
            su = self.sl[slice(2*self.N//3, self.N-2)]
            padded_array[su] = 0

@inheritdocstrings
class ShenNeumannBasis(ChebyshevBase):
    """Shen basis for homogeneous Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        mean : float, optional
            Mean value
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="GC", mean=0, domain=(-1., 1.), dtype=np.float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self.mean = mean
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._factor = np.zeros(0)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    def _composite_basis(self, V, argument=0):
        assert self.N == V.shape[1]
        P = np.zeros_like(V)
        k = np.arange(self.N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*V[:, 2:]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if 0 < i < self.N-2:
            return sp.chebyshevt(i, x) - (i/(i+2))**2*sp.chebyshevt(i+2, x)
        return 0

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*np.cos((i+2)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -(i*1./(i+2))**2)
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(float)
            self._factor = (k/(k+2))**2

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=True)
        self.set_factor_array(output)
        sm2 = self.sl[slice(0, -2)]
        s2p = self.sl[slice(2, None)]
        output[sm2] -= self._factor * output[s2p]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.evaluate_scalar_product(self.scalar_product.input_array,
                                     self.scalar_product.output_array,
                                     fast_transform=fast_transform)

        output = self.scalar_product.output_array
        output[self.si[0]] = self.mean*np.pi
        output[self.sl[slice(-2, None)]] = 0

        if output_array is not None:
            output_array[...] = output
            return output_array
        return output

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self._padding_backward(input_array, self.backward.tmp_array)
            SpectralBase.evaluate_expansion_all(self, self.backward.tmp_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        self.set_factor_array(input_array)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= self._factor*input_array[s0]
        self.CT.backward(w_hat)
        assert output_array is self.CT.backward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.evaluate_expansion_all(self.backward.input_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        self.set_factor_array(input_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= self._factor*input_array[s0]
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[2:] = self._factor*u[:-2]
        output_array -= n_cheb.chebval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_refined(self, N):
        return ShenNeumannBasis(N,
                                quad=self.quad,
                                domain=self.domain,
                                dtype=self.dtype,
                                padding_factor=self.padding_factor,
                                dealias_direct=self.dealias_direct,
                                coordinates=self.coors.coordinates,
                                mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenNeumannBasis(self.N,
                                quad=self.quad,
                                domain=self.domain,
                                dtype=self.dtype,
                                padding_factor=padding_factor,
                                dealias_direct=dealias_direct,
                                coordinates=self.coors.coordinates,
                                mean=self.mean)


@inheritdocstrings
class ShenBiharmonicBasis(ChebyshevBase):
    """Shen biharmonic basis

    Using 2 Dirichlet and 2 Neumann boundary conditions. All possibly
    nonhomogeneous.

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
            The two Dirichlet at (-1, 1) first and then the Neumann at (-1, 1).
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    """
    def __init__(self, N, quad="GC", bc=(0, 0, 0, 0), domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        from shenfun.tensorproductspace import BoundaryValues
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self.bc = BoundaryValues(self, bc=bc)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite_basis(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]
        if argument == 1: # if trial function
            P[:, -4:] = np.tensordot(V[:, :4], BCBiharmonicBasis.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if i < self.N-4:
            f = sp.chebyshevt(i, x) - (2*(i+2)/(i+3))*sp.chebyshevt(i+2, x) + (i+1)/(i+3)*sp.chebyshevt(i+4, x)
        else:
            f = BCBiharmonicBasis.coefficient_matrix()[i]*np.array([sp.chebyshevt(j, x) for j in range(4)])
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (2*(i+2.)/(i+3.))*np.cos((i+2)*w) + ((i+1.)/(i+3.))*np.cos((i+4)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2, i+4])] = (1, -(2*(i+2.)/(i+3.)), ((i+1.)/(i+3.)))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-2*(k+2)/(k+3)).astype(float)
            self._factor2 = ((k+1)/(k+3)).astype(float)

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, self.N-2)]
        s4 = self.sl[slice(4, self.N)]
        output[s] += self._factor1*Tk[s2]
        output[s] += self._factor2*Tk[s4]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.evaluate_scalar_product(self.scalar_product.input_array,
                                     self.scalar_product.output_array,
                                     fast_transform=fast_transform)

        self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        return self.scalar_product.output_array

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, self.N-2)]
        s4 = self.sl[slice(4, self.N)]
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self._padding_backward(input_array, self.backward.tmp_array)
            SpectralBase.evaluate_expansion_all(self, self.backward.tmp_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        self.set_factor_arrays(input_array)
        w_hat = self.set_w_hat(w_hat, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(w_hat, input_array)
        self.CT.backward(w_hat)
        assert input_array is self.backward.input_array
        assert output_array is self.backward.output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.evaluate_expansion_all(self.backward.input_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.bc.add_mass_rhs(u)
        self.apply_inverse_mass(u)
        self._truncation_forward(u, self.forward.output_array)
        self.bc.set_boundary_dofs(self.forward.output_array, False)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def slice(self):
        return slice(0, self.N-4)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = n_cheb.chebval(x, u[:-4])
        w_hat[2:-2] = self._factor1*u[:-4]
        output_array += n_cheb.chebval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*u[:-4]
        w_hat[:4] = 0
        output_array += n_cheb.chebval(x, w_hat)
        return output_array

    def get_bc_basis(self):
        return BCBiharmonicBasis(self.N, quad=self.quad, domain=self.domain, coordinates=self.coors.coordinates)

    def get_refined(self, N):
        return ShenBiharmonicBasis(N,
                                   quad=self.quad,
                                   domain=self.domain,
                                   dtype=self.dtype,
                                   padding_factor=self.padding_factor,
                                   dealias_direct=self.dealias_direct,
                                   coordinates=self.coors.coordinates,
                                   bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenBiharmonicBasis(self.N,
                                   quad=self.quad,
                                   domain=self.domain,
                                   dtype=self.dtype,
                                   padding_factor=padding_factor,
                                   dealias_direct=dealias_direct,
                                   coordinates=self.coors.coordinates,
                                   bc=self.bc.bc)

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def _truncation_forward(self, padded_array, trunc_array):
        if not id(trunc_array) == id(padded_array):
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            s = self.sl[slice(0, N-4)]
            trunc_array[s] = padded_array[s]
            s = self.sl[slice(-4, None)]
            trunc_array[s] = padded_array[s]

    def _padding_backward(self, trunc_array, padded_array):
        if not id(trunc_array) == id(padded_array):
            padded_array.fill(0)
            N = trunc_array.shape[self.axis]
            _sn = self.sl[slice(0, N-4)]
            padded_array[_sn] = trunc_array[_sn]
            _sn = self.sl[slice(N-4, N)]
            _sp = self.sl[slice(-4, None)]
            padded_array[_sp] = trunc_array[_sn]

        elif self.dealias_direct:
            su = self.sl[slice(2*self.N//3, self.N-4)]
            padded_array[su] = 0

@inheritdocstrings
class SecondNeumannBasis(ChebyshevBase): #pragma: no cover
    """Shen basis for homogeneous second order Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        mean : float, optional
            Mean value
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="GC", mean=0, domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self.mean = mean
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._factor = np.zeros(0)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'Neumann2'

    def _composite_basis(self, V, argument=0):
        assert self.N == V.shape[1]
        P = np.zeros_like(V)
        k = np.arange(self.N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*(k[:-2]**2-1)/((k[:-2]+2)**2-1)*V[:, 2:]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        return sp.chebyshevt(i, x) - (i/(i+2))**2*(i**2-1)/((i+2)**2-1)*sp.chebyshevt(i+2, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*(i**2-1.)/((i+2)**2-1.)*np.cos((i+2)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -(i*1./(i+2))**2*(i**2-1.)/((i+2)**2-1.))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(float)
            self._factor = (k/(k+2))**2*(k**2-1)/((k+2)**2-1)

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=True)
        self.set_factor_array(output)
        sm2 = self.sl[slice(0, -2)]
        s2p = self.sl[slice(2, None)]
        output[sm2] -= self._factor*output[s2p]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.evaluate_scalar_product(self.scalar_product.input_array,
                                     self.scalar_product.output_array,
                                     fast_transform=fast_transform)

        output = self.scalar_product.output_array
        output[self.si[0]] = self.mean*np.pi
        output[self.sl[slice(-2, None)]] = 0

        if output_array is not None:
            output_array[...] = output
            return output_array
        return output

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        self.set_factor_array(input_array)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= self._factor*input_array[s0]
        self.CT.backward(w_hat)
        assert output_array is self.CT.backward.output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        self.set_factor_array(input_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= self._factor*input_array[s0]
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[2:] = self._factor*u[:-2]
        output_array -= n_cheb.chebval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_refined(self, N):
        return SecondNeumannBasis(N,
                                  quad=self.quad,
                                  domain=self.domain,
                                  dtype=self.dtype,
                                  padding_factor=self.padding_factor,
                                  dealias_direct=self.dealias_direct,
                                  coordinates=self.coors.coordinates,
                                  mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return SecondNeumannBasis(self.N,
                                  quad=self.quad,
                                  domain=self.domain,
                                  dtype=self.dtype,
                                  padding_factor=padding_factor,
                                  dealias_direct=dealias_direct,
                                  coordinates=self.coors.coordinates,
                                  mean=self.mean)

@inheritdocstrings
class UpperDirichletBasis(ChebyshevBase):
    """Basis with homogeneous Dirichlet on upper edge (x=1) of boundary

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), dtype=np.float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._scaled = scaled
        self._factor = np.ones(1)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def _composite_basis(self, V, argument=0):
        P = np.zeros_like(V)
        P[:, :-1] = V[:, :-1] - V[:, 1:]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        assert i < self.N-1
        return sp.chebyshevt(i, x) - sp.chebyshevt(i+1, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - np.cos((i+1)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1])] = (1, -1)
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        s0 = self.sl[slice(0, -1)]
        s1 = self.sl[slice(1, None)]
        output[s0] -= output[s1]
        output[self.si[-1]] = 0

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self._padding_backward(input_array, self.backward.tmp_array)
            SpectralBase.evaluate_expansion_all(self, self.backward.tmp_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        s0 = self.sl[slice(0, -1)]
        s1 = self.sl[slice(1, None)]
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= input_array[s0]
        self.CT.backward(w_hat)
        assert output_array is self.CT.backward.output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, -1)]
        s1 = self.sl[slice(1, None)]
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]
        return output_array

    def slice(self):
        return slice(0, self.N-1)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        output_array[:] = n_cheb.chebval(x, u[:-1])
        w_hat[1:] = u[:-1]
        output_array -= n_cheb.chebval(x, w_hat)
        return output_array

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.apply_inverse_mass(u)
        self._truncation_forward(u, self.forward.output_array)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.evaluate_expansion_all(self.backward.input_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)


@inheritdocstrings
class ShenBiPolarBasis(ChebyshevBase):
    """Basis for the Biharmonic equation in polar coordinates

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="GC", domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        #assert quad == "GC"
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'BiPolar'

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def to_ortho(self, input_array, output_array=None):
        raise(NotImplementedError)

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        f = (1-x)**2*(1+x)**2*(sp.chebyshevt(i+1, x).diff(x, 1))
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        X = sp.symbols('x', real=True)
        f = self.sympy_basis(i, X)
        output_array[:] = sp.lambdify(X, f)(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], self.N))
        D = np.zeros(x.shape[0])
        for j in range(self.N-4):
            output_array[:, j] = self.evaluate_basis(x, j, D)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        X = sp.symbols('x', real=True)
        f = self.sympy_basis(i, X).diff(X, k)
        output_array[:] = sp.lambdify(X, f)(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], self.N))
        D = np.zeros(x.shape[0])
        for j in range(self.N-4):
            output_array[:, j] = self.evaluate_basis_derivative(x, j, k, output_array=D)
        return output_array

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        output = SpectralBase.scalar_product(self, input_array, output_array, False)
        output[self.sl[slice(-4, None)]] = 0
        return output

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        fj = self.evaluate_basis_all(x)
        output_array[:] = np.dot(fj, u)
        return output_array

    def forward(self, input_array=None, output_array=None, fast_transform=False):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.apply_inverse_mass(u)
        self._truncation_forward(u, self.forward.output_array)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)


@inheritdocstrings
class DirichletNeumannBasis(ChebyshevBase):
    """Basis for mixed Dirichlet/Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), dtype=np.float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._scaled = scaled
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumann'

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = ((-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2)).astype(float)
            self._factor2 = ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2)).astype(float)

    def _composite_basis(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     + ((-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2))*V[:, 1:-1]
                     + ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2))*V[:, 2:])
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        assert i < self.N-2
        return (sp.chebyshevt(i, x)
                + ((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+1, x)
                + ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = (np.cos(i*w)
                           + ((-i**2 + (i+2)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+1)*w)
                           + ((-i**2 - (i+1)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+2)*w))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1,((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2)), ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2)))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s0] = fk[s0]
        w_hat[s1] += f1*fk[s0]
        w_hat[s2] += f2*fk[s0]
        return w_hat

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        output[s0] += self._factor1*Tk[s1]
        output[s0] += self._factor2*Tk[s2]

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self._padding_backward(input_array, self.backward.tmp_array)
            SpectralBase.evaluate_expansion_all(self, self.backward.tmp_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        self.set_factor_arrays(input_array)
        w_hat = self.set_w_hat(w_hat, input_array, self._factor1, self._factor2)
        self.CT.backward(w_hat)

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += n_cheb.chebval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += n_cheb.chebval(x, w_hat)
        return output_array

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.apply_inverse_mass(u)
        self._truncation_forward(u, self.forward.output_array)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.evaluate_expansion_all(self.backward.input_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

@inheritdocstrings
class NeumannDirichletBasis(ChebyshevBase):
    """Basis for mixed Neumann/Dirichlet boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), dtype=np.float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)
        self.CT = Basis(N, quad=quad, dtype=dtype, padding_factor=padding_factor, dealias_direct=dealias_direct)
        self._scaled = scaled
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'NeumannDirichlet'

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-(-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2)).astype(float)
            self._factor2 = ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2)).astype(float)

    def _composite_basis(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-2]
        P[:, :-2] = (V[:, :-2] 
                     - ((-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2))*V[:, 1:-1]
                     + ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2))*V[:, 2:])
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        assert i < self.N-2
        return (sp.chebyshevt(i, x)
                - ((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+1, x)
                + ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = (np.cos(i*w)
                           - ((-i**2 + (i+2)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+1)*w)
                           + ((-i**2 - (i+1)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+2)*w))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1,-((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2)), ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2)))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s0] = fk[s0]
        w_hat[s1] += f1*fk[s0]
        w_hat[s2] += f2*fk[s0]
        return w_hat

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        output[s0] += self._factor1*Tk[s1]
        output[s0] += self._factor2*Tk[s2]

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self._padding_backward(input_array, self.backward.tmp_array)
            SpectralBase.evaluate_expansion_all(self, self.backward.tmp_array, output_array, False)
            return
        w_hat = work[(input_array, 0, True)]
        self.set_factor_arrays(input_array)
        w_hat = self.set_w_hat(w_hat, input_array, self._factor1, self._factor2)
        self.CT.backward(w_hat)

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array.__array__())
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += n_cheb.chebval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += n_cheb.chebval(x, w_hat)
        return output_array

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.apply_inverse_mass(u)
        self._truncation_forward(u, self.forward.output_array)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.evaluate_expansion_all(self.backward.input_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        self.CT.tensorproductspace = self.tensorproductspace
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.backward_uniform = Transform(self.backward_uniform, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)


@inheritdocstrings
class BCBasis(ChebyshevBase):
    """Basis for Dirichlet boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to use scaled basis
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), scaled=False,
                 coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain,
                               coordinates=coordinates)
        self.plan(N, 0, np.float, {})

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]
        shape = list(shape) if np.ndim(shape) else [shape]
        assert shape[axis] == self.shape(False)
        U = np.zeros(shape, dtype=dtype)
        shape[axis] = 2
        V = np.zeros(shape, dtype=dtype)
        self.forward = Transform(self.forward, lambda: None, U, V, V)
        self.backward = Transform(self.backward, lambda: None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, lambda: None, U, V, V)

    def slice(self):
        return slice(self.N-2, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 2
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 1)

    def coefficient_matrix(self):
        return np.array([[0.5, -0.5],
                         [0.5, 0.5]])

    def _composite_basis(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        #P[:, 0] = (V[:, 0] - V[:, 1])/2
        #P[:, 1] = (V[:, 0] + V[:, 1])/2
        P[:] = np.tensordot(V[:, :2], self.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if i == 0:
            return 0.5*(1-x)
        elif i == 1:
            return 0.5*(1+x)
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 0.5*(1-x)
        elif i == 1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 1:
            output_array[:] = -0.5
        elif i == 1 and k == 1:
            output_array[:] = 0.5
        else:
            output_array[:] = 0
        return output_array

@inheritdocstrings
class BCBiharmonicBasis(ChebyshevBase):
    """Basis for inhomogeneous Biharmonic boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to use scaled basis
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), scaled=False,
                 coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, coordinates=coordinates)
        self.plan(N, 0, np.float, {})

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]
        shape = list(shape) if np.ndim(shape) else [shape]
        assert shape[axis] == self.shape(False)
        U = np.zeros(shape, dtype=dtype)
        shape[axis] = 4
        V = np.zeros(shape, dtype=dtype)
        self.forward = Transform(self.forward, lambda: None, U, V, V)
        self.backward = Transform(self.backward, lambda: None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, lambda: None, U, V, V)

    def slice(self):
        return slice(self.N-4, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 4
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 3)

    @staticmethod
    def coefficient_matrix():
        return np.array([[0.5, -9./16., 0, 1./16],
                         [0.5, 9./16., 0, -1./16.],
                         [1./8., -1./16., -1./8., 1./16.],
                         [-1./8., -1./16., 1./8., 1./16.]])

    def _composite_basis(self, V, argument=0):
        P = np.tensordot(V[:, :4], self.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        assert i < 4, 'Only four bases, i < 4'
        return np.sum(self.coefficient_matrix()[i]*np.array([sp.chebyshevt(j, x) for j in range(4)]))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        V = self.vandermonde(x)
        output_array[:] = np.dot(V, self.coefficient_matrix()[i])
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        output_array = SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)
        return output_array
