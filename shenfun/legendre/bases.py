"""
Module for defining function spaces in the Legendre family
"""

from __future__ import division
import os
import functools
import sympy
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.special import eval_legendre
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, islicedict, \
    slicedict
from .lobatto import legendre_lobatto_nodes_and_weights
from .fastgl import fastgl_wrap

__all__ = ['LegendreBase', 'Orthogonal', 'ShenDirichlet',
           'ShenBiharmonic', 'ShenNeumann',
           'ShenBiPolar', 'ShenBiPolar0',
           'NeumannDirichlet', 'DirichletNeumann',
           'UpperDirichletNeumann',
           'UpperDirichlet',
           'BCDirichlet', 'BCBiharmonic', 'BCNeumann']

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

    def reference_domain(self):
        return (-1, 1)

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        if self.quad == "LG":
            points, weights = fastgl_wrap.leggauss(N)
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

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        return sympy.legendre(i, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_legendre(i, x, out=output_array)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        #N, M = self.shape(False), self.shape(True)
        M = V.shape[-1]
        if k > 0:
            D = np.zeros((M, M))
            D[:-k] = leg.legder(np.eye(M, M), k)
            V = np.dot(V, D)
        return self._composite(V, argument=argument)

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        return self._composite(V, argument=argument)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[i] = 1
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        return basis(x)

    def _composite(self, V, argument=0):
        """Return composite basis, where ``V`` is primary Vandermonde matrix."""
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


class ShenDirichlet(LegendreBase):
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
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._factor = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'SD'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_array(self, v):
        if self.is_scaled():
            if not self._factor.shape == v.shape:
                k = self.wavenumbers().astype(float)
                self._factor = 1./np.sqrt(4*k+6)

    def is_scaled(self):
        return self._scaled

    def _composite(self, V, argument=0):
        P = np.zeros(V.shape)
        if not self.is_scaled():
            P[:, :-2] = V[:, :-2] - V[:, 2:]
        else:
            k = np.arange(self.N-2).astype(float)
            P[:, :-2] = (V[:, :-2] - V[:, 2:])/np.sqrt(4*k+6)
        if argument == 1:
            P[:, -2] = (V[:, 0] - V[:, 1])/2
            P[:, -1] = (V[:, 0] + V[:, 1])/2
        return P

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]

        if self.is_scaled():
            k = self.wavenumbers()
            output_array[s0] = input_array[s0]/np.sqrt(4*k+6)
            output_array[s1] -= input_array[s0]/np.sqrt(4*k+6)

        else:
            output_array[s0] = input_array[s0]
            output_array[s1] -= input_array[s0]

        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        f = sympy.legendre(i, x)-sympy.legendre(i+2, x)
        if self.is_scaled():
            f /= np.sqrt(4*i+6)
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_legendre(i, x) - eval_legendre(i+2, x)
        if self.is_scaled():
            output_array /= np.sqrt(4*i+6)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -1)
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        if self.is_scaled():
            output_array /= np.sqrt(4*i+6)
        return output_array

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
        #phi_i = np.zeros_like(xj)
        #Lnm = eval_legendre(0, xj)
        #Ln = eval_legendre(1, xj)
        #Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
        #for i in range(self.N-2):
        #    phi_i[:] = Lnm-Lnp
        #    output_array[i] = np.sum(phi_i*wj*input_array)
        #    Lnm[:] = Ln
        #    Ln[:] = Lnp
        #    Lnp[:] = ((2*(i+2)+1)*xj*Ln - (i+2)*Lnm)/(i+3)

        output_array[self.si[-2]] = 0
        output_array[self.si[-1]] = 0

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = leg.legval(x, u[:-2]*self._factor)
        w_hat[2:] = u[:-2]*self._factor
        output_array -= leg.legval(x, w_hat)
        output_array += 0.5*(u[-1]*(1+x) + u[-2]*(1-x))
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     scaled=self._scaled, coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return ShenDirichlet(N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc,
                             scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenDirichlet(self.N,
                             quad=self.quad,
                             dtype=self.dtype,
                             padding_factor=padding_factor,
                             dealias_direct=dealias_direct,
                             domain=self.domain,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc,
                             scaled=self._scaled)

    def get_unplanned(self):
        return ShenDirichlet(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc,
                             scaled=self._scaled)


class ShenNeumann(LegendreBase):
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

    def __init__(self, N, quad="LG", mean=0, bc=(0., 0.), domain=(-1., 1.), padding_factor=1,
                 dealias_direct=False, dtype=float, coordinates=None):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.mean = mean
        self._factor = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

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

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros(V.shape)
        k = np.arange(V.shape[1]).astype(float)
        P[:, :-2] = V[:, :-2] - (k[:-2]*(k[:-2]+1)/(k[:-2]+2))/(k[:-2]+3)*V[:, 2:]
        if argument == 1:
            P[:, -2] = 0.5*V[:, 1] - 1/6*V[:, 2]
            P[:, -1] = 0.5*V[:, 1] + 1/6*V[:, 2]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(float)
            self._factor = k*(k+1)/(k+2)/(k+3)

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

        #try:
        xj, wj = self.points_and_weights(self.N)
        from shenfun.optimization.numba import legendre as legn
        legn.legendre_shenneumann_scalar_product(xj, wj, input_array, output_array)
        #except:
        #    raise RuntimeError('Requires Numba')

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

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        f = sympy.legendre(i, x) - (i*(i+1))/((i+2)*(i+3))*sympy.legendre(i+2, x)
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_legendre(i, x) - i*(i+1.)/(i+2.)/(i+3.)*eval_legendre(i+2, x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -i*(i+1.)/(i+2.)/(i+3.))
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        self.set_factor_array(input_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= self._factor*input_array[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = leg.legval(x, u[:-2])
        w_hat[2:] = self._factor*u[:-2]
        output_array -= leg.legval(x, w_hat)
        output_array += u[-2]*(0.5*x-1/3*(3*x**2-1)) + u[-1]*(0.5*x+1/3*(3*x**2-1))
        return output_array

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


class ShenBiharmonic(LegendreBase):
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
                 dealias_direct=False, dtype=float, coordinates=None):
        from shenfun.tensorproductspace import BoundaryValues
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SB'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(2*k+5)/(2*k+7))*V[:, 2:-2] + ((2*k+3)/(2*k+7))*V[:, 4:]
        if argument == 1:
            P[:, -4:] = np.tensordot(V[:, :4], BCBiharmonic.coefficient_matrix(), (1, 1))
        return P

    def set_factor_arrays(self, v):
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-2*(2*k+5)/(2*k+7)).astype(float)
            self._factor2 = ((2*k+3)/(2*k+7)).astype(float)

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2): # pragma: no cover
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, -2)]
        s4 = self.sl[slice(4, None)]
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i < self.N-4:
            f = (sympy.legendre(i, x)
                 -2*(2*i+5.)/(2*i+7.)*sympy.legendre(i+2, x)
                 +((2*i+3.)/(2*i+7.))*sympy.legendre(i+4, x))
        else:
            f = 0
            for j, c in enumerate(BCBiharmonic.coefficient_matrix()[i-(self.N-4)]):
                f += c*sympy.legendre(j, x)
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-4:
            output_array[:] = eval_legendre(i, x) - 2*(2*i+5.)/(2*i+7.)*eval_legendre(i+2, x) + ((2*i+3.)/(2*i+7.))*eval_legendre(i+4, x)
        else:
            X = sympy.symbols('x', real=True)
            output_array[:] = sympy.lambdify(X, self.sympy_basis(i, x=X))(x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        if i < self.N-4:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+2, i+4])] = (1, -2*(2*i+5.)/(2*i+7.), ((2*i+3.)/(2*i+7.)))
            basis = leg.Legendre(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            X = sympy.symbols('x', real=True)
            output_array[:] = sympy.lambdify(X, self.sympy_basis(i, X).diff(X, k))(x)
        return output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-4)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(u)
        output_array[:] = leg.legval(x, u[:-4])
        w_hat[2:-2] = self._factor1*u[:-4]
        output_array += leg.legval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*u[:-4]
        w_hat[:4] = 0
        output_array += leg.legval(x, w_hat)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return ShenBiharmonic(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenBiharmonic(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return ShenBiharmonic(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

class BeamFixedFree(LegendreBase):
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
                 dealias_direct=False, dtype=float, coordinates=None):
        from shenfun.tensorproductspace import BoundaryValues
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._factor3 = np.zeros(0)
        self._factor4 = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'BeamFixedFree'

    @staticmethod
    def short_name():
        return 'BF'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-4]
        P[:, :-4] = (V[:, :-4] + 4*(2*k+3)/((k+3)**2)*V[:, 1:-3] - 2*(k-1)*(k+1)*(k+6)*(2*k+5)/((k+3)**2*(k+4)*(2*k+7))*V[:, 2:-2]
         - 4*(k+1)**2*(2*k+3)/((k+3)**2*(k+4)**2)*V[:, 3:-1] + ((k+1)**2*(k+2)**2*(2*k+3)/((k+3)**2*(k+4)**2*(2*k+7)))*V[:, 4:])
        if argument == 1:
            P[:, -4:] = np.tensordot(V[:, :4], BCBeamFixedFree.coefficient_matrix(), (1, 1))
        return P

    def set_factor_arrays(self, v):
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (4*(2*k+3)/((k+3)**2)).astype(float)
            self._factor2 = (-(2*(k-1)*(k+1)*(k+6)*(2*k+5)/((k+3)**2*(k+4)*(2*k+7)))).astype(float)
            self._factor3 = (- 4*(k+1)**2*(2*k+3)/((k+3)**2*(k+4)**2)).astype(float)
            self._factor4 = ((((k+1)/(k+3))*((k+2)/(k+4)))**2*(2*k+3)/(2*k+7)).astype(float)

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0

    def set_w_hat(self, w_hat, fk, f1, f2): # pragma: no cover
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, -2)]
        s4 = self.sl[slice(4, None)]
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i < self.N-4:
            f = (sympy.legendre(i, x)
                 +(4*(2*i+3)/((i+3)**2))*sympy.legendre(i+1, x)
                 -(2*(i-1)*(i+1)*(i+6)*(2*i+5)/((i+3)**2*(i+4)*(2*i+7)))*sympy.legendre(i+2, x)
                 -4*(i+1)**2*(2*i+3)/((i+3)**2*(i+4)**2)*sympy.legendre(i+3, x)
                 +(i+1)**2*(i+2)**2*(2*i+3)/((i+3)**2*(i+4)**2*(2*i+7))*sympy.legendre(i+4, x))
        else:
            f = 0
            for j, c in enumerate(BCBeamFixedFree.coefficient_matrix()[i-(self.N-4)]):
                f += c*sympy.legendre(j, x)
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-4:
            output_array[:] = eval_legendre(i, x) + (4*(2*i+3)/((i+3)**2))*eval_legendre(i+1, x) \
                -(2*(i-1)*(i+1)*(i+6)*(2*i+5)/((i+3)**2*(i+4)*(2*i+7)))*eval_legendre(i+2, x) \
                    -4*(i+1)**2*(2*i+3)/((i+3)**2*(i+4)**2)*eval_legendre(i+3, x) \
                    +(i+1)**2*(i+2)**2*(2*i+3)/((i+3)^2*(i+4)**2*(2*i+7))*eval_legendre(i+4, x)
        else:
            X = sympy.symbols('x', real=True)
            output_array[:] = sympy.lambdify(X, self.sympy_basis(i, x=X))(x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        if i < self.N-4:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+1, i+2, i+3, i+4])] = (1, 4*(2*i+3)/((i+3)**2), -(2*(i-1)*(i+1)*(i+6)*(2*i+5)/((i+3)**2*(i+4)*(2*i+7))), \
                                                        -4*(i+1)**2*(2*i+3)/((i+3)**2*(i+4)**2), \
                                                            (i+1)**2*(i+2)**2*(2*i+3)/((i+3)**2*(i+4)**2*(2*i+7)))
            basis = leg.Legendre(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            X = sympy.symbols('x', real=True)
            output_array[:] = sympy.lambdify(X, self.sympy_basis(i, X).diff(X, k))(x)
        return output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(self.get_orthogonal())
        else:
            output_array.fill(0)

        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-4)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(u)
        output_array[:] = leg.legval(x, u[:-4])
        w_hat[1:-3] = self._factor1*u[:-4]
        w_hat[0] = 0
        output_array += leg.legval(x, w_hat[:-3])
        w_hat[2:-2] = self._factor2*u[:-4]
        w_hat[:2] = 0
        output_array += leg.legval(x, w_hat[:-2])
        w_hat[3:-1] = self._factor3*u[:-4]
        w_hat[:3] = 0
        output_array += leg.legval(x, w_hat[:-1])
        w_hat[4:] = self._factor3*u[:-4]
        w_hat[:4] = 0
        output_array += leg.legval(x, w_hat)

        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBeamFixedFree(self.N, quad=self.quad, domain=self.domain,
                                         coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return BeamFixedFree(N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return BeamFixedFree(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=padding_factor,
                             dealias_direct=dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc)

    def get_unplanned(self):
        return BeamFixedFree(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc)


class UpperDirichlet(LegendreBase):
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
                 padding_factor=1, dealias_direct=False, coordinates=None):
        assert quad == "LG"
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._factor = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def is_scaled(self):
        return False

    def _composite(self, V, argument=0):
        P = np.zeros(V.shape)
        P[:, :-1] = V[:, :-1] - V[:, 1:]
        if argument == 1: # if trial function
            P[:, -1] = (V[:, 0] + V[:, 1])/2    # x = +1
        return P

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        s0 = self.sl[slice(0, -1)]
        s1 = self.sl[slice(1, None)]
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-1)

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i < self.N-1:
            return sympy.legendre(i, x)-sympy.legendre(i+1, x)
        assert i == self.N-1
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-1:
            output_array[:] = eval_legendre(i, x) - eval_legendre(i+1, x)
        elif i == self.N-1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        if i < self.N-1:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+1])] = (1, -1)
            basis = leg.Legendre(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            if k == 1:
                output_array[:] = 0.5
            else:
                output_array[:] = 0
        return output_array

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.si[-1]] = 0

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        output_array[:] = leg.legval(x, u[:-1])
        w_hat[1:] = u[:-1]
        output_array -= leg.legval(x, w_hat)
        output_array += 0.5*u[-1]*(1+x)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichlet(self.N, quad=self.quad, domain=self.domain,
                                          coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return UpperDirichlet(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return UpperDirichlet(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return UpperDirichlet(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)


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
    def __init__(self, N, quad="LG", domain=(-1., 1.), dtype=float,
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

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def to_ortho(self, input_array, output_array=None):
        raise(NotImplementedError)

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        return (1-x)**2*(1+x)**2*(sympy.legendre(i+1, x).diff(x, 1))

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


class ShenBiPolar0(LegendreBase):
    """Legendre function space for biharmonic basis for polar coordinates

    Homogeneous Dirichlet and Neumann boundary conditions.

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - LG - Legendre-Gauss
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
    def __init__(self, N, quad="LG", domain=(-1., 1.), padding_factor=1,
                 dealias_direct=False, dtype=float, coordinates=None):
        assert quad == "LG"
        LegendreBase.__init__(self, N, quad="LG", domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._factor3 = np.zeros(0)

    @staticmethod
    def boundary_condition():
        return 'BiPolar0'

    @staticmethod
    def short_name():
        return 'SP0'

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-3]
        P[:, :-3] = V[:, :-3] - ((2*k+3)*(k+4)/(2*k+5)/(k+2))*V[:, 1:-2] - (k*(k+1)/(k+2)/(k+3))*V[:, 2:-1] + (k+1)*(2*k+3)/(k+3)/(2*k+5)*V[:, 3:]
        return P

    def set_factor_arrays(self, v):
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-(2*k+3)*(k+4)/(2*k+5)/(k+2)).astype(float)
            self._factor2 = (-k*(k+1)/(k+2)/(k+3)).astype(float)
            self._factor3 = ((k+1)*(2*k+3)/(k+3)/(2*k+5)).astype(float)

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2, f3): # pragma: no cover
        s = self.sl[self.slice()]
        s1 = self.sl[slice(1, -2)]
        s2 = self.sl[slice(2, -1)]
        s3 = self.sl[slice(3, None)]
        w_hat[s] = fk[s]
        w_hat[s1] += f1*fk[s]
        w_hat[s2] += f2*fk[s]
        w_hat[s3] += f3*fk[s]
        return w_hat

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        #x = self.map_reference_domain(x)
        return (sympy.legendre(i, x)
                -(2*i+3)*(i+4)/(2*i+5)/(i+2)*sympy.legendre(i+1, x)
                -i*(i+1)/(i+2)/(i+3)*sympy.legendre(i+2, x)
                +(i+1)*(2*i+3)/(i+3)/(2*i+5)*sympy.legendre(i+3, x))
        #return
        # (sympy.legendre(i, x) -(2*i+3)*(i+4)/(2*i+5)*sympy.legendre(i+1, x) -i*(i+1)/(i+2)/(i+3)*sympy.legendre(i+2, x) +(i+1)*(i+2)*(2*i+3)/(i+3)/(2*i+5)*sympy.legendre(i+3, x))

    def evaluate_basis(self, x=None, i=0, output_array=None):
        output_array = SpectralBase.evaluate_basis(self, x=x, i=i, output_array=output_array)
        return output_array

    #def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
    #    output_array = SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)
    #    return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)

        if i < self.N-3:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+1, i+2, i+3])] = (1,
                                                   -(2*i+3)*(i+4)/(2*i+5)/(i+2),
                                                   -i*(i+1)/(i+2)/(i+3),
                                                   (i+1)*(2*i+3)/(i+3)/(2*i+5))
            basis = leg.Legendre(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            raise RuntimeError
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        V = np.zeros((x.shape[0], self.N))
        for i in range(self.N-3):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2, self._factor3)
        return output_array

    def slice(self):
        return slice(0, self.N-3)

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-3, None)]] = 0

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(u)
        output_array[:] = leg.legval(x, u[:-3])
        w_hat[1:-2] = self._factor1*u[:-3]
        output_array += leg.legval(x, w_hat[:-2])
        w_hat[2:-1] = self._factor2*u[:-3]
        w_hat[:2] = 0
        output_array += leg.legval(x, w_hat)
        w_hat[3:] = self._factor3*u[:-3]
        w_hat[:3] = 0
        output_array += leg.legval(x, w_hat)
        return output_array


class DirichletNeumann(LegendreBase):
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
                 padding_factor=1, dealias_direct=False, coordinates=None):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._factor1 = np.ones(1)
        self._factor2 = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumann'

    @staticmethod
    def short_name():
        return 'DN'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = ((2*k+3)/(k+2)**2).astype(float)
            self._factor2 = -(((k+1)/(k+2))**2).astype(float)

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     +((2*k+3)/(k+2)**2)*V[:, 1:-1]
                     -(((k+1)/(k+2))**2)*V[:, 2:])
        if argument == 1:
            P[:, -2] = V[:, 0]
            P[:, -1] = V[:, 0]+V[:, 1]
        return P

    def set_w_hat(self, w_hat, fk, f1, f2):
        s = self.sl[self.slice()]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s] = fk[s]
        w_hat[s1] += f1*fk[s]
        w_hat[s2] += f2*fk[s]
        return w_hat

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        assert i < self.N-2
        return (sympy.legendre(i, x)
                +(2*i+3)/(i+2)**2*sympy.legendre(i+1, x)
                -(i+1)**2/(i+2)**2*sympy.legendre(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = (eval_legendre(i, x)
                           +(2*i+3)/(i+2)**2*eval_legendre(i+1, x)
                           -(i+1)**2/(i+2)**2*eval_legendre(i+2, x))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1, (2*i+3)/(i+2)**2, -(i+1)**2/(i+2)**2)
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(w_hat)
        output_array[:] = leg.legval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += leg.legval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += leg.legval(x, w_hat)
        output_array += u[-2] + u[-1]*(1+x)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              domain=self.domain,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)


class NeumannDirichlet(LegendreBase):
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
                 padding_factor=1, dealias_direct=False, coordinates=None):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._factor1 = np.ones(1)
        self._factor2 = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'NeumannDirichlet'

    @staticmethod
    def short_name():
        return 'ND'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-(2*k+3)/(k+2)**2).astype(float)
            self._factor2 = -((k+1)**2/(k+2)**2).astype(float)

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     -((2*k+3)/(k+2)**2)*V[:, 1:-1]
                     -((k+1)**2/(k+2)**2)*V[:, 2:])
        if argument == 1:
            P[:, -2] = V[:, 0]-0.5*V[:, 1]-0.5*V[:, 2]
            P[:, -1] = V[:, 0]
        return P

    def set_w_hat(self, w_hat, fk, f1, f2): # pragma: no cover
        s = self.sl[self.slice()]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s] = fk[s]
        w_hat[s1] += f1*fk[s]
        w_hat[s2] += f2*fk[s]
        return w_hat

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        assert i < self.N-2
        return (sympy.legendre(i, x)
                -(2*i+3)/(i+2)**2*sympy.legendre(i+1, x)
                -(i+1)**2/(i+2)**2*sympy.legendre(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = (eval_legendre(i, x)
                           -(2*i+3)/(i+2)**2*eval_legendre(i+1, x)
                           -(i+1)**2/(i+2)**2*eval_legendre(i+2, x))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1, -(2*i+3)/(i+2)**2, -(i+1)**2/(i+2)**2)
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(w_hat)
        output_array[:] = leg.legval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += leg.legval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += leg.legval(x, w_hat)
        output_array += u[-1] + u[-2]*(1-0.5*x-0.25*(3*x**2-1))
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumannDirichlet(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              domain=self.domain,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

class UpperDirichletNeumann(LegendreBase):
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
                 padding_factor=1, dealias_direct=False, coordinates=None):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._factor1 = np.ones(1)
        self._factor2 = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichletNeumann'

    @staticmethod
    def short_name():
        return 'UDN'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-(2*k+3)/(k+2)).astype(float)
            self._factor2 = ((k+1)/(k+2)).astype(float)

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     -((2*k+3)/(k+2))*V[:, 1:-1]
                     +((k+1)/(k+2))*V[:, 2:])
        if argument == 1:
            P[:, -2] = V[:, 0]
            P[:, -1] = V[:, 0]-2*V[:, 1]+V[:, 2]
        return P

    def set_w_hat(self, w_hat, fk, f1, f2):
        s = self.sl[self.slice()]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s] = fk[s]
        w_hat[s1] += f1*fk[s]
        w_hat[s2] += f2*fk[s]
        return w_hat

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        assert i < self.N-2
        return (sympy.legendre(i, x)
                -(2*i+3)/(i+2)*sympy.legendre(i+1, x)
                +(i+1)/(i+2)*sympy.legendre(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = (eval_legendre(i, x)
                           -(2*i+3)/(i+2)*eval_legendre(i+1, x)
                           +(i+1)/(i+2)*eval_legendre(i+2, x))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1, -(2*i+3)/(i+2), (i+1)/(i+2))
        basis = leg.Legendre(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(w_hat)
        output_array[:] = leg.legval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += leg.legval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += leg.legval(x, w_hat)
        output_array += u[-2] + u[-1]*(1-2*x+0.5*(3*x**2-1))
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                                 coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              domain=self.domain,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)


class BCDirichlet(LegendreBase):

    def __init__(self, N, quad="LG", scaled=False, dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)
        self._scaled = scaled

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

    @staticmethod
    def short_name():
        return 'BCD'

    def vandermonde(self, x):
        return leg.legvander(x, 1)

    @staticmethod
    def coefficient_matrix():
        return np.array([[0.5, -0.5],
                         [0.5, 0.5]])

    def _composite(self, V, argument=0):
        P = np.zeros(V.shape)
        P[:, 0] = (V[:, 0] - V[:, 1])/2
        P[:, 1] = (V[:, 0] + V[:, 1])/2
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i == 0:
            return 0.5*(1-x)
        elif i == 1:
            return 0.5*(1+x)
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
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

class BCNeumann(LegendreBase):

    def __init__(self, N, quad="LG", scaled=False, dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)
        self._scaled = scaled

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

    @staticmethod
    def short_name():
        return 'BCN'

    def vandermonde(self, x):
        return leg.legvander(x, 2)

    @staticmethod
    def coefficient_matrix():
        return np.array([[0, 1/2, -1/6],
                         [0, 1/2, 1/6]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = 0.5*V[:, 1] - 1/6*V[:, 2]
        P[:, 1] = 0.5*V[:, 1] + 1/6*V[:, 2]
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i == 0:
            return x/2-(3*x**2-1)/3
        elif i == 1:
            return x/2+(3*x**2-1)/3
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = x/2-(3*x**2-1)/3
        elif i == 1:
            output_array[:] = x/2+(3*x**2-1)/3
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = x/2-(3*x**2-1)/3
        elif i == 0 and k == 1:
            output_array[:] = 0.5-2*x
        elif i == 0 and k == 2:
            output_array[:] = -2
        elif i == 1 and k == 0:
            output_array[:] = x/2+(3*x**2-1)/3
        elif i == 1 and k == 1:
            output_array[:] = 0.5+2*x
        elif i == 1 and k == 2:
            output_array[:] = 2
        else:
            output_array[:] = 0
        return output_array


class BCBiharmonic(LegendreBase):
    """Function space for inhomogeneous Biharmonic boundary conditions

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
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="LG", domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)

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

    @staticmethod
    def short_name():
        return 'BCB'

    def vandermonde(self, x):
        return leg.legvander(x, 3)

    @staticmethod
    def coefficient_matrix():
        return np.array([[0.5, -0.6, 0, 0.1],
                         [0.5, 0.6, 0, -0.1],
                         [1./6., -1./10., -1./6., 1./10.],
                         [-1./6., -1./10., 1./6., 1./10.]])

    def _composite(self, V, argument=0):
        P = np.tensordot(V[:, :4], self.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i < 4:
            f = 0
            for j, c in enumerate(self.coefficient_matrix()[i]):
                f += c*sympy.legendre(j, x)
            return f
        else:
            raise AttributeError('Only four bases, i < 4')

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

class BCBeamFixedFree(LegendreBase):
    """Function space for inhomogeneous Biharmonic boundary conditions

    u(-1), u'(-1), u''(1), u'''(1)

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
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="LG", domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        LegendreBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)

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

    @staticmethod
    def short_name():
        return 'BCBF'

    def vandermonde(self, x):
        return leg.legvander(x, 3)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, 0, 0, 0],
                          [1, 1, 0, 0],
                          [2/3, 1, 1/3, 0],
                          [-1, -1.4, -1/3, 1/15]])

    def _composite(self, V, argument=0):
        P = np.tensordot(V[:, :4], self.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i < 4:
            f = 0
            for j, c in enumerate(self.coefficient_matrix()[i]):
                f += c*sympy.legendre(j, x)
            return f
        else:
            raise AttributeError('Only four bases, i < 4')

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

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 1))
        self.evaluate_basis_derivative(x=x, k=k, output_array=output_array[:, 0])
        return output_array


class BCUpperDirichlet(LegendreBase):
    """Function space for Dirichlet boundary conditions at x=1

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
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="GC", domain=(-1., 1.), scaled=False,
                 dtype=float, coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, domain=domain,
                              dtype=dtype, coordinates=coordinates)

    def slice(self):
        return slice(self.N-1, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 1
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCUD'

    def vandermonde(self, x):
        return leg.legvander(x, 1)

    def coefficient_matrix(self):
        return np.array([[0.5, 0.5]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :1].shape)
        P[:, 0] = (V[:, 0] + V[:, 1])/2
        return P

    def sympy_basis(self, i=0, x=sympy.Symbol('x', real=True)):
        if i == 0:
            return 0.5*(1+x)
        else:
            raise AttributeError('Only one basis, i == 0')

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 0.5*(1+x)
        else:
            raise AttributeError('Only one basis, i == 0')
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        assert i == 0
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = 0
        if k == 1:
            output_array[:] = 0.5
        elif k == 0:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 1))
        self.evaluate_basis_derivative(x=x, k=k, output_array=output_array[:, 0])
        return output_array

class BCNeumannDirichlet(LegendreBase):

    def __init__(self, N, quad="LG", scaled=False, dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)
        self._scaled = scaled

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

    @staticmethod
    def short_name():
        return 'BCND'

    def vandermonde(self, x):
        return leg.legvander(x, 2)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, -0.5, -0.5],
                         [1, 0, 0]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = V[:, 0] - 0.5*V[:, 1] -0.5*V[:, 2]
        P[:, 1] = V[:, 0]
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i == 0:
            return 1-0.5*x-0.25*(3*x**2-1)
        elif i == 1:
            return 1
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 1-0.5*x-0.25*(3*x**2-1)
        elif i == 1:
            output_array[:] = 1
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = 1-0.5*x-0.25*(3*x**2-1)
        elif i == 0 and k == 1:
            output_array[:] = -0.5-1.5*x
        elif i == 0 and k == 2:
            output_array[:] = -1.5
        elif i == 1 and k == 0:
            output_array[:] = 1
        else:
            output_array[:] = 0
        return output_array

    #def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
    #    if x is None:
    #        x = self.mesh(False, False)
    #    output_array = np.zeros((self.N, 2))
    #    self.evaluate_basis_derivative(x=x, i=0, k=k, output_array=output_array[:, 0])
    #    self.evaluate_basis_derivative(x=x, i=1, k=k, output_array=output_array[:, 1])
    #    return output_array

class BCDirichletNeumann(LegendreBase):

    def __init__(self, N, quad="LG", dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)

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

    @staticmethod
    def short_name():
        return 'BCDN'

    def vandermonde(self, x):
        return leg.legvander(x, 1)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, 0],
                         [1, 1]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = V[:, 0]
        P[:, 1] = V[:, 0] + V[:, 1]
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i == 0:
            return 1
        elif i == 1:
            return 1+x
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 1
        elif i == 1:
            output_array[:] = 1+x
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = 1
        elif i == 1 and k == 0:
            output_array[:] = 1+x
        elif i == 1 and k == 1:
            output_array[:] = 1
        else:
            output_array[:] = 0
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 2))
        self.evaluate_basis_derivative(x=x, i=0, k=k, output_array=output_array[:, 0])
        self.evaluate_basis_derivative(x=x, i=1, k=k, output_array=output_array[:, 1])
        return output_array

class BCUpperDirichletNeumann(LegendreBase):

    def __init__(self, N, quad="LG", dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        LegendreBase.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)

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

    @staticmethod
    def short_name():
        return 'BCUDN'

    def vandermonde(self, x):
        return leg.legvander(x, 2)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, 0, 0],
                         [1, -2, 1]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = V[:, 0]
        P[:, 1] = V[:, 0] - 2*V[:, 1] + V[:, 2]
        return P

    def sympy_basis(self, i=0, x=sympy.symbols('x', real=True)):
        if i == 0:
            return 1
        elif i == 1:
            return 1-2*x+0.5*(3*x**2-1)
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 1
        elif i == 1:
            output_array[:] = 1-2*x+0.5*(3*x**2-1)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = 1
        elif i == 1 and k == 0:
            output_array[:] = 1-2*x+0.5*(3*x**2-1)
        elif i == 1 and k == 1:
            output_array[:] = -2+3*x
        elif i == 1 and k == 2:
            output_array[:] = 3
        else:
            output_array[:] = 0
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 2))
        self.evaluate_basis_derivative(x=x, i=0, k=k, output_array=output_array[:, 0])
        self.evaluate_basis_derivative(x=x, i=1, k=k, output_array=output_array[:, 1])
        return output_array
