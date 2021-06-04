"""
Module for defining function spaces in the Legendre family
"""

from __future__ import division
import os
import functools
import sympy as sp
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.special import eval_legendre
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, islicedict, \
    slicedict
from shenfun.matrixbase import SparseMatrix
from .lobatto import legendre_lobatto_nodes_and_weights
from . import fastgl

__all__ = ['LegendreBase',
           'Orthogonal',
           'ShenDirichlet',
           'ShenBiharmonic',
           'ShenNeumann',
           'ShenBiPolar',
           'ShenBiPolar0',
           'LowerDirichlet',
           'NeumannDirichlet',
           'DirichletNeumann',
           'UpperDirichlet',
           'UpperDirichletNeumann',
           'DirichletNeumannDirichlet',
           'BCDirichlet',
           'BCBiharmonic',
           'BCNeumann',
           'BCNeumannDirichlet',
           'BCDirichletNeumann',
           'BCBeamFixedFree',
           'BCLowerDirichlet',
           'BCUpperDirichlet',
           'BCUpperDirichletNeumann',
           'BCDirichletNeumannDirichlet',
           'BCShenBiPolar0']

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

try:
    import quadpy
    from mpmath import mp
    mp.dps = 30
    has_quadpy = True
except:
    has_quadpy = False
    mp = None

mode = os.environ.get('SHENFUN_LEGENDRE_MODE', 'numpy')
mode = mode if has_quadpy else 'numpy'

xp = sp.Symbol('x', real=True)

class LegendreBase(SpectralBase):
    """Base class for all Legendre spaces

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        domain : 2-tuple of floats, optional
                 The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="LG", domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        SpectralBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        self.plan(int(padding_factor*N), 0, dtype, {})

    @staticmethod
    def family():
        return 'legendre'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        if self.quad == "LG":
            points, weights = fastgl.leggauss(N)
            #points, weights = leg.leggauss(N)

        elif self.quad == "GL":
            points, weights = legendre_lobatto_nodes_and_weights(N)
        else:
            raise NotImplementedError

        if map_true_domain is True:
            points = self.map_true_domain(points)

        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        if self.quad == 'LG':
            pw = quadpy.c1.gauss_legendre(N, 'mpmath')
        elif self.quad == 'GL':
            pw = quadpy.c1.gauss_lobatto(N) # No mpmath in quadpy for lobatto:-(
        points = pw.points_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, pw.weights_symbolic

    def vandermonde(self, x):
        return leg.legvander(x, self.shape(False)-1)

    def reference_domain(self):
        return (-1, 1)

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
            self.forward = Transform(self.forward, None, U, V, trunc_array)
            self.backward = Transform(self.backward, None, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, None, U, V, V)
            self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_orthogonal(self):
        return Orthogonal(self.N, quad=self.quad, dtype=self.dtype,
                          domain=self.domain,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates)

class Orthogonal(LegendreBase):
    """Function space for regular (orthogonal) Legendre series

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="LG", domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)

    def sympy_basis(self, i=0, x=xp):
        return sp.legendre(i, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_legendre(i, x, out=output_array)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[i] = 1
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        M = V.shape[-1]
        if k > 0:
            D = np.zeros((M, M))
            D[:-k] = leg.legder(np.eye(M, M), k)
            V = np.dot(V, D)
        return V

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        output_array[:] = leg.legval(x, u)
        return output_array

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'L'

    def to_ortho(self, input_array, output_array=None):
        assert input_array.__class__.__name__ == 'Orthogonal'
        if output_array:
            output_array[:] = input_array
            return output_array
        return input_array

class CompositeSpace(Orthogonal):
    """Common class for all spaces based on composite bases"""

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        Orthogonal.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    def plan(self, shape, axis, dtype, options):
        Orthogonal.plan(self, shape, axis, dtype, options)
        LegendreBase.plan(self, shape, axis, dtype, options)

    def evaluate_basis_all(self, x=None, argument=0):
        V = Orthogonal.evaluate_basis_all(self, x=x, argument=argument)
        return self._composite(V, argument=argument)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        V = Orthogonal.evaluate_basis_derivative_all(self, x=x, k=k, argument=argument)
        return self._composite(V, argument=argument)

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
            return
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        input_array[:] = self.to_ortho(input_array, output_array)
        Orthogonal._evaluate_expansion_all(self, input_array, output_array, x, True)

    def _evaluate_scalar_product(self, fast_transform=True):
        output = self.scalar_product.output_array
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            output[self.sl[slice(-(self.N-self.dim()), None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        nbcs = self.N-self.dim()
        s = [np.newaxis]*self.dimensions
        w0 = output.copy()
        output.fill(0)
        for key, val in self.stencil_matrix(self.shape(False)).items():
            M = self.N if key >= 0 else self.dim()
            s1 = slice(max(0, -key), M-max(0, key))
            Q = s1.stop-s1.start
            s2 = self.sl[slice(max(0, key), Q+max(0, key))]
            sk = self.sl[s1]
            s[self.axis] = slice(0, Q)
            output[sk] += val[tuple(s)]*w0[s2]
        output[self.sl[slice(-nbcs, None)]] = 0

    @property
    def is_orthogonal(self):
        return False

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return self._scaled

    def stencil_matrix(self, N=None):
        """Matrix describing the linear combination of orthogonal basis
        functions for the current basis.

        Parameters
        ----------
        N : int, optional
            The number of quadrature points
        """
        raise NotImplementedError

    def _composite(self, V, argument=0):
        """Return Vandermonde matrix V adjusted for basis composition

        Parameters
        ----------
        V : Vandermonde type matrix
        argument : int
                Zero for test and 1 for trialfunction

        """
        P = np.zeros_like(V)
        P[:] = V * self.stencil_matrix(V.shape[1]).diags().T
        if argument == 1: # if trial function
            P[:, slice(-(self.N-self.dim()), None)] = self.get_bc_basis()._composite(V)
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N
        if i < self.dim():
            row = self.stencil_matrix().diags().getrow(i)
            f = 0
            for j, val in zip(row.indices, row.data):
                f += val*Orthogonal.sympy_basis(self, i=j, x=x)
        else:
            f = self.get_bc_basis().sympy_basis(i=i-self.dim(), x=x)
        return f

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        nbcs = self.N-self.dim()
        s = [np.newaxis]*self.dimensions
        for key, val in self.stencil_matrix().items():
            M = self.N if key >= 0 else self.dim()
            s0 = slice(max(0, -key), min(self.dim(), M-max(0, key)))
            Q = s0.stop-s0.start
            s1 = slice(max(0, key), max(0, key)+Q)
            s[self.axis] = slice(0, Q)
            output_array[self.sl[s1]] += val[tuple(s)]*input_array[self.sl[s0]]

        if self.has_nonhomogeneous_bcs:
            self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.dim():
            row = self.stencil_matrix().diags().getrow(i)
            w0 = np.zeros_like(output_array)
            output_array.fill(0)
            for j, val in zip(row.indices, row.data):
                output_array[:] += val*Orthogonal.evaluate_basis(self, x, i=j, output_array=w0)
        else:
            assert i < self.N
            output_array = self.get_bc_basis().evaluate_basis(x, i=i-self.dim(), output_array=output_array)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)

        if i < self.dim():
            basis = np.zeros(self.shape(True))
            M = self.stencil_matrix()
            row = M.diags().getrow(i)
            indices = row.indices
            vals = row.data
            basis[np.array(indices)] = vals
            basis = leg.Legendre(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            output_array[:] = self.get_bc_basis().evaluate_basis_derivative(x, i-self.dim(), k)
        return output_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w = self.to_ortho(u)
        output_array[:] = leg.legval(x, w)
        return output_array

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              scaled=self._scaled,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              scaled=self._scaled,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              scaled=self._scaled,
                              bc=self.bc.bc)

class ShenDirichlet(CompositeSpace):
    """Legendre Function space for Dirichlet boundary conditions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        bc : tuple of numbers
            Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to scale test functions with 1/sqrt(4k+6).
            Scaled test functions give a stiffness matrix equal to the
            identity matrix.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0., 0.), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
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
        d[-2:] = 0
        if self.is_scaled():
            k = np.arange(N)
            d /= np.sqrt(4*k+6)
        return SparseMatrix({0: d, 2: -d[:-2]}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def _evaluate_expansion_all(self, input_array, output_array,
                               x=None, fast_transform=False):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, fast_transform)
            return
        assert input_array.ndim == 1, 'Use fast_transform=False'
        xj, _ = self.points_and_weights(self.N)
        from shenfun.optimization.numba import legendre as legn
        legn.legendre_shendirichlet_evaluate_expansion_all(xj, input_array, output_array, self.is_scaled())

    def _evaluate_scalar_product(self, fast_transform=False):
        input_array = self.scalar_product.input_array
        output_array = self.scalar_product.output_array
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            output_array[self.si[-2]] = 0
            output_array[self.si[-1]] = 0
            return
        M = self.shape(False)
        xj, wj = self.points_and_weights(M)
        assert input_array.ndim == 1, 'Use fast_transform=False'
        from shenfun.optimization.numba import legendre as legn
        legn.legendre_shendirichlet_scalar_product(xj, wj, input_array, output_array, self.is_scaled())
        output_array[self.si[-2]] = 0
        output_array[self.si[-1]] = 0

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     scaled=self._scaled, coordinates=self.coors.coordinates)
        return self._bc_basis

class ShenNeumann(CompositeSpace):
    """Function space for homogeneous Neumann boundary conditions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        mean : number
            mean value
        bc : 2-tuple of numbers
            Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="LG", mean=None, bc=(0., 0.), domain=(-1., 1.), padding_factor=1,
                 scaled=False, dealias_direct=False, dtype=float, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        self.mean = mean

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'SN'

    @property
    def use_fixed_gauge(self):
        if self.mean is None:
            return False
        T = self.tensorproductspace
        if T:
            return T.use_fixed_gauge
        return True

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-2:] = 0
        k = np.arange(N-2)
        return SparseMatrix({0: d, 2: -k*(k+1)/(k+2)/(k+3)}, (N, N))

    def _evaluate_scalar_product(self, fast_transform=False):
        input_array = self.scalar_product.input_array
        output_array = self.scalar_product.output_array
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            output_array[self.sl[slice(-2, None)]] = 0
            if self.use_fixed_gauge:
                output_array[self.si[0]] = self.mean*np.pi
            return

        assert input_array.ndim == 1, 'Use fast_transform=False'

        xj, wj = self.points_and_weights(self.N)
        from shenfun.optimization.numba import legendre as legn
        legn.legendre_shenneumann_scalar_product(xj, wj, input_array, output_array)

        output_array[self.sl[slice(-2, None)]] = 0
        if self.use_fixed_gauge:
            output_array[self.si[0]] = self.mean*np.pi

    def _evaluate_expansion_all(self, input_array, output_array,
                                x=None, fast_transform=False):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, fast_transform)
            return

        assert input_array.ndim == 1, 'Use fast_transform=False'
        xj, _ = self.points_and_weights(self.N)
        try:
            from shenfun.optimization.numba import legendre as legn
            legn.legendre_shenneumann_evaluate_expansion_all(xj, input_array, output_array)
        except:
            raise RuntimeError('Requires Numba')

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumann(self.N, quad=self.quad, domain=self.domain,
                                   coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return ShenNeumann(N,
                           quad=self.quad,
                           domain=self.domain,
                           bc=self.bc.bc,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenNeumann(self.N,
                           quad=self.quad,
                           domain=self.domain,
                           bc=self.bc.bc,
                           dtype=self.dtype,
                           padding_factor=padding_factor,
                           dealias_direct=dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

    def get_unplanned(self):
        return ShenNeumann(self.N,
                           quad=self.quad,
                           domain=self.domain,
                           bc=self.bc.bc,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

class ShenBiharmonic(CompositeSpace):
    """Function space for biharmonic basis

    Both Dirichlet and Neumann boundary conditions.

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto
        4-tuple of numbers, optional
            The values of the 4 boundary conditions at x=(-1, 1).
            The two Dirichlet first and then the Neumann.
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0, 0, 0, 0), domain=(-1., 1.), padding_factor=1,
                 scaled=False, dealias_direct=False, dtype=float, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SB'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-4:] = 0
        k = np.arange(N)
        return SparseMatrix({0: d, 2: -2*(2*k[:-2]+5)/(2*k[:-2]+7), 4: (2*k[:-4]+3)/(2*k[:-4]+7)}, (N, N))

    def slice(self):
        return slice(0, self.N-4)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

class BeamFixedFree(CompositeSpace):
    """Function space for biharmonic basis

    Function space for biharmonic basis

    Fulfills the following boundary conditions:

        u(-1) = a, u'(-1) = b, u''(1) = c, u'''(1) = d.

    Both Dirichlet and Neumann boundary conditions.

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto
        4-tuple of numbers, optional
            The values of the 4 boundary conditions
            u(-1) = a, u'(-1) = b, u''(1) = c, u'''(1) = d

        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbol('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0, 0, 0, 0), domain=(-1., 1.), padding_factor=1,
                 scaled=False, dealias_direct=False, dtype=float, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'BeamFixedFree'

    @staticmethod
    def short_name():
        return 'BF'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-4:] = 0
        k = np.arange(N)
        f1 =4*(2*k[:-1]+3)/(k[:-1]+3)**2
        f2 =-(2*(k[:-2]-1)*(k[:-2]+1)*(k[:-2]+6)*(2*k[:-2]+5)/((k[:-2]+3)**2*(k[:-2]+4)*(2*k[:-2]+7)))
        f3 =- 4*(k[:-3]+1)**2*(2*k[:-3]+3)/((k[:-3]+3)**2*(k[:-3]+4)**2)
        f4 =(((k[:-4]+1)/(k[:-4]+3))*((k[:-4]+2)/(k[:-4]+4)))**2*(2*k[:-4]+3)/(2*k[:-4]+7)
        return SparseMatrix({0: d, 1: f1, 2: f2,3: f3, 4: f4}, (N, N))

    def slice(self):
        return slice(0, self.N-4)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBeamFixedFree(self.N, quad=self.quad, domain=self.domain,
                                         coordinates=self.coors.coordinates)
        return self._bc_basis

class UpperDirichlet(CompositeSpace):
    """Legendre function space with homogeneous Dirichlet boundary conditions on x=1

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(None, 0), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        assert quad == "LG"
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-1] = 0
        return SparseMatrix({0: d, 1: -d[:-1]}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-1)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichlet(self.N, quad=self.quad, domain=self.domain,
                                          coordinates=self.coors.coordinates)
        return self._bc_basis

class ShenBiPolar(LegendreBase):
    """Legendre function space for the Biharmonic equation in polar coordinates

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", domain=(-1., 1.), bc=(0, 0), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        assert quad == "LG"
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SP'

    def to_ortho(self, input_array, output_array=None):
        raise(NotImplementedError)

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        return (1-x)**2*(1+x)**2*(sp.legendre(i+1, x).diff(x, 1))

    def evaluate_basis(self, x=None, i=0, output_array=None):
        output_array = SpectralBase.evaluate_basis(self, x=x, i=i, output_array=output_array)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        output_array = SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            #x = self.mesh(False, False)
            x = self.mpmath_points_and_weights()[0]
        output_array = np.zeros((x.shape[0], self.N))
        for j in range(self.N-4):
            output_array[:, j] = self.evaluate_basis(x, j, output_array=output_array[:, j])
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        V = np.zeros((x.shape[0], self.N))
        for i in range(self.N-2):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        else:
            output_array.fill(0)
        x = self.map_reference_domain(x)
        fj = self.evaluate_basis_all(x)
        output_array[:] = np.dot(fj, u)
        return output_array

class ShenBiPolar0(CompositeSpace):
    """Legendre function space for biharmonic basis for polar coordinates

    Three boundary conditions at v(1), v'(1) and v'(-1)

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
        bc : 3-tuple of numbers, optional
            The values of the 3 boundary conditions at x=(-1, 1).
            The two Dirichlet first and then the Neumann.
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", domain=(-1., 1.), bc=(0, 0, 0), padding_factor=1,
                 scaled=False, dealias_direct=False, dtype=float, coordinates=None):
        assert quad == "LG"
        CompositeSpace.__init__(self, N, quad="LG", domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'BiPolar0'

    @staticmethod
    def short_name():
        return 'SP0'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-3:] = 0
        k = np.arange(self.N)
        f1 = -((2*k[:-1]+3)*(k[:-1]+4)/(2*k[:-1]+5)/(k[:-1]+2))
        f2 = -(k[:-2]*(k[:-2]+1)/(k[:-2]+2)/(k[:-2]+3))
        f3 = (k[:-3]+1)*(2*k[:-3]+3)/(k[:-3]+3)/(2*k[:-3]+5)
        return SparseMatrix({0: d, 1: f1, 2: f2, 3: f3}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-3)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCShenBiPolar0(self.N, quad=self.quad, domain=self.domain,
                                        coordinates=self.coors.coordinates)
        return self._bc_basis

class DirichletNeumann(CompositeSpace):
    """Function space for mixed Dirichlet/Neumann boundary conditions
	u(-1)=0, u'(1)=0
    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        bc : tuple of numbers
            Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to scale test functions with 1/sqrt(4k+6).
            Scaled test functions give a stiffness matrix equal to the
            identity matrix.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0., 0.), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumann'

    @staticmethod
    def short_name():
        return 'DN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-2:] = 0
        k = np.arange(self.N)
        f1 = (2*k[:-1]+3)/(k[:-1]+2)**2
        f2 = -((k[:-2]+1)/(k[:-2]+2))**2
        return SparseMatrix({0: d, 1: f1, 2: f2}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

class LowerDirichlet(CompositeSpace):
    """Legendre function space with homogeneous Dirichlet boundary conditions on x = -1

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0, None), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        assert quad == "LG"
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'LowerDirichlet'

    @staticmethod
    def short_name():
        return 'LD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-1] = 0
        return SparseMatrix({0: d, 1: d[:-1]}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-1)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCLowerDirichlet(self.N, quad=self.quad, domain=self.domain,
                                          coordinates=self.coors.coordinates)
        return self._bc_basis

class DirichletNeumannDirichlet(CompositeSpace):
    """Function space for biharmonic basis

    Boundary conditions: u(-1) = 0, u'(-1) = 0, u(1) = 0..

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto
        3-tuple of numbers, optional
            The values of the 3 boundary conditions at x=(-1, 1).
            The two Dirichlet first and then the Neumann.
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0, 0, 0), domain=(-1., 1.), padding_factor=1,
                 scaled=False, dealias_direct=False, dtype=float, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumannDirichlet'

    @staticmethod
    def short_name():
        return 'DND'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-2:] = 0
        k = np.arange(self.N)
        f1 = (2*k[:-1]+3)/(2*k[:-1]+5)
        f2 = -np.ones(N-2)
        f3 = -(2*k[:-3]+3)/(2*k[:-3]+5)
        return SparseMatrix({0: d, 1: f1, 2: f2, 3: f3}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-3)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichletNeumannDirichlet(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

class NeumannDirichlet(CompositeSpace):
    """Function space for mixed Dirichlet/Neumann boundary conditions
	u'(-1)=0, u(1)=0
    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        bc : tuple of numbers
            Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to scale test functions with 1/sqrt(4k+6).
            Scaled test functions give a stiffness matrix equal to the
            identity matrix.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """
    def __init__(self, N, quad="LG", bc=(0., 0.), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'NeumannDirichlet'

    @staticmethod
    def short_name():
        return 'ND'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-2:] = 0
        k = np.arange(self.N)
        f1 = -((2*k[:-1]+3)/(k[:-1]+2)**2)
        f2 = -((k[:-2]+1)**2/(k[:-2]+2)**2)
        return SparseMatrix({0: d, 1: f1, 2: f2}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumannDirichlet(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

class UpperDirichletNeumann(CompositeSpace):
    """Function space for mixed Dirichlet/Neumann boundary conditions
	u(1)=0, u'(1)=0

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        bc : tuple of numbers
            Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to scale test functions with 1/sqrt(4k+6).
            Scaled test functions give a stiffness matrix equal to the
            identity matrix.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    This basis is not recommended as it leads to a poorly conditioned
    stiffness matrix.
    """
    def __init__(self, N, quad="LG", bc=(0., 0.), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichletNeumann'

    @staticmethod
    def short_name():
        return 'UDN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(self.N)
        d[-2:] = 0
        k = np.arange(self.N)
        f1 = -((2*k[:-1]+3)/(k[:-1]+2))
        f2 = ((k[:-2]+1)/(k[:-2]+2))
        return SparseMatrix({0: d, 1: f1, 2: f2}, (self.N, self.N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                                 coordinates=self.coors.coordinates)
        return self._bc_basis

class BCBase(CompositeSpace):
    """Function space for inhomogeneous boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
            - GL - Legendre-Gauss-Lobatto

        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to use scaled basis
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="GC", domain=(-1., 1.),
                 dtype=float, coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain,
                                dtype=dtype, coordinates=coordinates)

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
        return leg.legvander(x, self.num_T-1)

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([sp.legendre(j, x) for j in range(self.num_T)]))

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

class BCDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCD'

    def stencil_matrix(self, N=None):
        return np.array([[0.5, -0.5],
                         [0.5, 0.5]])

class BCNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCN'

    def stencil_matrix(self, N=None):
        return np.array([[0, 1/2, -1/6],
                         [0, 1/2, 1/6]])

class BCBiharmonic(BCBase):

    @staticmethod
    def short_name():
        return 'BCB'

    def stencil_matrix(self, N=None):
        return np.array([[0.5, -0.6, 0, 0.1],
                         [0.5, 0.6, 0, -0.1],
                         [1./6., -1./10., -1./6., 1./10.],
                         [-1./6., -1./10., 1./6., 1./10.]])

class BCBeamFixedFree(BCBase):

    # u(-1), u'(-1), u''(1), u'''(1)

    @staticmethod
    def short_name():
        return 'BCBF'

    def stencil_matrix(self, N=None):
        return np.array([[1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [2/3, 1, 1/3, 0],
                         [-1, -1.4, -1/3, 1/15]])

class BCLowerDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCLD'

    def stencil_matrix(self, N=None):
        return np.array([[0.5, -0.5]])

class BCUpperDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCUD'

    def stencil_matrix(self, N=None):
        return np.array([[0.5, 0.5]])

class BCNeumannDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCND'

    def stencil_matrix(self, N=None):
        return np.array([[1, -0.5, -0.5],
                         [1, 0, 0]])

class BCDirichletNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCDN'

    def stencil_matrix(self, N=None):
        return np.array([[1, 0],
                         [1, 1]])

class BCUpperDirichletNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCUDN'

    def stencil_matrix(self, N=None):
        return np.array([[1, 0, 0],
                         [1, -2, 1]])

class BCDirichletNeumannDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCST'

    def stencil_matrix(self, N=None):
        return np.array([[2/3, -1/2, -1/6],
                         [1/3, 0, -1/3],
                         [1/3, 1/2, 1/6]])

class BCShenBiPolar0(BCBase):

    @staticmethod
    def short_name():
        return 'BCSP'

    def stencil_matrix(self, N=None):
        return np.array([[1, 0, 0],
                         [-2/3, 1/2, 1/6],
                         [-1/3, 1/2, -1/6]])
