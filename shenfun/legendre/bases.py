"""
Module for defining bases in the Legendre family
"""

import functools
import sympy
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.special import eval_legendre
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, islicedict, \
    slicedict
from shenfun.utilities import inheritdocstrings
from .lobatto import legendre_lobatto_nodes_and_weights

__all__ = ['LegendreBase', 'Basis', 'ShenDirichletBasis',
           'ShenBiharmonicBasis', 'ShenNeumannBasis',
           'SecondNeumannBasis']

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import


@inheritdocstrings
class LegendreBase(SpectralBase):
    """Base class for all Legendre bases

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
    """

    def __init__(self, N=0, quad="LG", domain=(-1., 1.)):
        SpectralBase.__init__(self, N, quad, domain=domain)
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)

    @staticmethod
    def family():
        return 'legendre'

    def reference_domain(self):
        return (-1., 1.)

    def points_and_weights(self, N=None, map_true_domain=False, **kw):
        if N is None:
            N = self.N
        if self.quad == "LG":
            points, weights = leg.leggauss(N)
        elif self.quad == "GL":
            points, weights = legendre_lobatto_nodes_and_weights(N)
        else:
            raise NotImplementedError

        if map_true_domain is True:
            points = self.map_true_domain(points)

        return points, weights

    def vandermonde(self, x):
        return leg.legvander(x, self.N-1)

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
        #assert self.N == V.shape[1]
        N, M = self.shape(False), self.shape(True)
        if k > 0:
            D = np.zeros((M, N))
            D[:-k] = leg.legder(np.eye(M, N), k)
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
        x = np.atleast_1d(x)
        v = self.evaluate_basis(x, i, output_array)
        N, M = self.shape(False), self.shape(True)
        if k > 0:
            D = np.zeros((M, N))
            D[:-k] = leg.legder(np.eye(M, N), k)
            v = np.dot(v, D)
        return v

    def _composite_basis(self, V, argument=0):
        """Return composite basis, where ``V`` is primary Vandermonde matrix."""
        return V

    def plan(self, shape, axis, dtype, options):
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
        self.forward = Transform(self.forward, None, U, V, V)
        self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_orthogonal(self):
        return Basis(self.N, quad=self.quad, domain=self.domain)

@inheritdocstrings
class Basis(LegendreBase):
    """Basis for regular Legendre series

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
    """

    def __init__(self, N=0, quad="LG", domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.plan(N, 0, np.float, {})

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        output_array[:] = leg.legval(x, u)
        return output_array

    @property
    def is_orthogonal(self):
        return True

@inheritdocstrings
class ShenDirichletBasis(LegendreBase):
    """Shen Legendre basis for Dirichlet boundary conditions

    Parameters
    ----------
        N : int, optional
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
    """
    def __init__(self, N=0, quad="LG", bc=(0., 0.),
                 domain=(-1., 1.), scaled=False):
        LegendreBase.__init__(self, N, quad, domain=domain)
        from shenfun.tensorproductspace import BoundaryValues
        self.LT = Basis(N, quad)
        self._scaled = scaled
        self._factor = np.ones(1)
        self.plan(N, 0, np.float, {})
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    def set_factor_array(self, v):
        if self.is_scaled():
            if not self._factor.shape == v.shape:
                k = self.wavenumbers().astype(np.float)
                self._factor = 1./np.sqrt(4*k+6)

    def is_scaled(self):
        return self._scaled

    def _composite_basis(self, V, argument=0):
        P = np.zeros(V.shape)
        if not self.is_scaled():
            P[:, :-2] = V[:, :-2] - V[:, 2:]
        else:
            k = np.arange(self.N-2).astype(np.float)
            P[:, :-2] = (V[:, :-2] - V[:, 2:])/np.sqrt(4*k+6)
        if argument == 1:
            P[:, -2] = (V[:, 0] + V[:, 1])/2
            P[:, -1] = (V[:, 0] - V[:, 1])/2
        return P

    #def evaluate_expansion_all(self, input_array, output_array,
    #                           fast_transform=False): # pragma: no cover
    #    # Not used since there are no fast transforms for Legendre
    #    w_hat = work[(input_array, 0)]
    #    s0 = self.sl[slice(0, -2)]
    #    s1 = self.sl[slice(2, None)]
    #    self.set_factor_array(input_array)
    #    w_hat[s0] = input_array[s0]*self._factor
    #    w_hat[s1] -= input_array[s0]*self._factor
    #    self.bc.apply_before(w_hat, False, (0.5, 0.5))
    #    output_array = self.LT.backward(w_hat)
    #    assert input_array is self.backward.input_array
    #    assert output_array is self.backward.output_array

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array.v)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]

        if self.is_scaled():
            k = self.wavenumbers()
            output_array[s0] = input_array[s0]/np.sqrt(4*k+6)
            output_array[s1] -= input_array[s0]/np.sqrt(4*k+6)

        else:
            output_array[s0] = input_array[s0]
            output_array[s1] -= input_array[s0]

        self.bc.apply_before(output_array, True, (0.5, 0.5))
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_legendre(i, x) - eval_legendre(i+2, x)
        if self.is_scaled():
            output_array /= np.sqrt(4*i+6)
        return output_array

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)
        output_array[self.si[-2]] = 0
        output_array[self.si[-1]] = 0

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = leg.legval(x, u[:-2]*self._factor)
        w_hat[2:] = u[:-2]*self._factor
        output_array -= leg.legval(x, w_hat)
        output_array += 0.5*(u[-1]*(1-x)+u[-2]*(1+x))
        return output_array

    def apply_bc_rhs(self, u, final=False, scales=(-1., -1./3.)):
        if self.is_scaled():
            scales = (-1./np.sqrt(6.), -1./3./np.sqrt(10.))
        self.bc.apply_before(u, final, scales=scales)

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        self.scalar_product(input_array, fast_transform=fast_transform)
        u = self.scalar_product.tmp_array
        self.apply_bc_rhs(u)
        self.apply_inverse_mass(u)
        self.bc.apply_after(u, False)
        self._truncation_forward(u, self.forward.output_array)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.axis = axis
        self.forward = Transform(self.forward, None, U, V, V)
        self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def get_bc_basis(self):
        return BCBasis(self.N, quad=self.quad, domain=self.domain, scaled=self._scaled)


@inheritdocstrings
class ShenNeumannBasis(LegendreBase):
    """Shen basis for homogeneous Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - LG - Legendre-Gauss
               - GL - Legendre-Gauss-Lobatto

        mean : number
               mean value
        domain : 2-tuple of floats, optional
                 The computational domain
    """

    def __init__(self, N=0, quad="LG", mean=0, domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.mean = mean
        self.LT = Basis(N, quad)
        self._factor = np.zeros(0)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    def _composite_basis(self, V, argument=0):
        assert self.N == V.shape[1]
        P = np.zeros(V.shape)
        k = np.arange(self.N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]*(k[:-2]+1)/(k[:-2]+2))/(k[:-2]+3)*V[:, 2:]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(np.float)
            self._factor = k*(k+1)/(k+2)/(k+3)

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        output = SpectralBase.scalar_product(self, input_array, output_array, False)
        output[self.si[0]] = self.mean*np.pi
        output[self.sl[slice(-2, None)]] = 0
        return output

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_legendre(i, x) - i*(i+1.)/(i+2.)/(i+3.)*eval_legendre(i+2, x)
        return output_array

    #def evaluate_expansion_all(self, input_array, output_array): # pragma: no cover
    #    # Not used since there are no fast transforms for Legendre
    #    w_hat = work[(input_array, 0)]
    #    self.set_factor_array(input_array)
    #    s0 = self.sl[slice(0, -2)]
    #    s1 = self.sl[slice(2, None)]
    #    w_hat[s0] = input_array[s0]
    #    w_hat[s1] -= self._factor*input_array[s0]
    #    output_array = self.LT.backward(w_hat)

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array.v)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        self.set_factor_array(input_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= self._factor*input_array[s0]
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = leg.legval(x, u[:-2])
        w_hat[2:] = self._factor*u[:-2]
        output_array -= leg.legval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.axis = axis
        self.forward = Transform(self.forward, None, U, V, V)
        self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

@inheritdocstrings
class ShenBiharmonicBasis(LegendreBase):
    """Shen biharmonic basis

    Homogeneous Dirichlet and Neumann boundary conditions.

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
    """
    def __init__(self, N=0, quad="LG", domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.LT = Basis(N, quad)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    def _composite_basis(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(2*k+5)/(2*k+7))*V[:, 2:-2] + ((2*k+3)/(2*k+7))*V[:, 4:]
        return P

    def set_factor_arrays(self, v):
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(np.float)
            self._factor1 = (-2*(2*k+5)/(2*k+7)).astype(float)
            self._factor2 = ((2*k+3)/(2*k+7)).astype(float)

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        output = LegendreBase.scalar_product(self, input_array, output_array, False)
        output[self.sl[slice(-4, None)]] = 0
        return output

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2): # pragma: no cover
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, -2)]
        s4 = self.sl[slice(4, None)]
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_legendre(i, x) - 2*(2*i+5.)/(2*i+7.)*eval_legendre(i+2, x) + ((2*i+3.)/(2*i+7.))*eval_legendre(i+4, x)
        return output_array

    #def evaluate_expansion_all(self, input_array, output_array): # pragma: no cover
    #    # Not used since there are no fast transforms for Legendre
    #    w_hat = work[(input_array, 0)]
    #    self.set_factor_arrays(input_array)
    #    w_hat = self.set_w_hat(w_hat, input_array, self._factor1, self._factor2)
    #    output_array = self.LT.backward(w_hat)

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array.v)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        return output_array

    def slice(self):
        return slice(0, self.N-4)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
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

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.axis = axis
        self.forward = Transform(self.forward, None, U, V, V)
        self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

## Experimental!
@inheritdocstrings
class SecondNeumannBasis(LegendreBase): # pragma: no cover
    """Shen basis for homogeneous second order Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - LG - Legendre-Gauss
               - GL - Legendre-Gauss-Lobatto
        mean : number
               Mean value of solution
        domain : 2-tuple of floats, optional
                 The computational domain
    """
    def __init__(self, N=0, quad="LG", mean=0, domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.mean = mean
        self.LT = Basis(N, quad)
        self._factor = np.zeros(0)
        self.plan(N, 0, np.float, {})

    def _composite_basis(self, V, argument=0):
        assert self.N == V.shape[1]
        P = np.zeros(V.shape)
        k = np.arange(self.N).astype(np.float)[:-4]
        a_k = -(k+1)*(k+2)*(2*k+3)/((k+3)*(k+4)*(2*k+7))
        P[:, :-4] = V[:, :-4] + (a_k-1)*V[:, 2:-2] - a_k*V[:, 4:]
        P[:, -4] = V[:, 0]
        P[:, -3] = V[:, 1]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(np.float)
            self._factor = -(k+1)*(k+2)*(2*k+3)/((k+3)*(k+4)*(2*k+7))

    #def evaluate_expansion_all(self, u, output_array):
        #w_hat = work[(u, 0)]
        #self.set_factor_array(u)
        #s0 = self.sl[slice(0, -4)]
        #s1 = self.sl[slice(2, -2)]
        #s2 = self.sl[slice(4, None)]
        #w_hat[s0] = u[s0]
        #w_hat[s1] += (self._factor-1)*u[s0]
        #w_hat[s2] -= self._factor*u[s0]
        #output_array = self.LT.backward(w_hat)
        #return output_array

    def slice(self):
        return slice(0, self.N-2)

    #def eval(self, x, input_array):
        #w_hat = work[(input_array, 0)]
        #self.set_factor_array(input_array)
        #f = leg.legval(x, input_array[:-2])
        #w_hat[2:] = self._factor*input_array[:-2]
        #f -= leg.legval(x, w_hat)
        #return f

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.axis = axis
        self.forward = Transform(self.forward, None, U, V, V)
        self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

@inheritdocstrings
class BCBasis(LegendreBase):

    def __init__(self, N=0, quad="GC", bc=(0, 0), scaled=False,
                 domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self._scaled = scaled
        self.plan(N, 0, np.float, {})
        self.bc = bc

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
        return leg.legvander(x, 1)

    def _composite_basis(self, V, argument=0):
        P = np.zeros(V.shape)
        P[:, 0] = (V[:, 0] + V[:, 1])/2
        P[:, 1] = (V[:, 0] - V[:, 1])/2
        return P

    def sympy_basis(self, i=0):
        x = sympy.symbols('x')
        if i == 0:
            return 0.5*(1+x)
        else:
            return 0.5*(1-x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 0.5*(1+x)
        elif i == 1:
            output_array[:] = 0.5*(1-x)
        return output_array
