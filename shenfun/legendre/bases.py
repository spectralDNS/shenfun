"""
Module for defining bases in the Legendre family
"""

import numpy as np
from numpy.polynomial import legendre as leg
import pyfftw
from shenfun.spectralbase import SpectralBase, work
from shenfun.utilities import inheritdocstrings
from .lobatto import legendre_lobatto_nodes_and_weights

__all__ = ['LegendreBase', 'Basis', 'ShenDirichletBasis',
           'ShenBiharmonicBasis', 'ShenNeumannBasis',
           'SecondNeumannBasis']

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

class _Wrap(object):

    __slots__ = ('_func', '__doc__', '_input_array', '_output_array')

    def __init__(self, func, input_array, output_array):
        object.__setattr__(self, '_func', func)
        object.__setattr__(self, '_input_array', input_array)
        object.__setattr__(self, '_output_array', output_array)
        object.__setattr__(self, '__doc__', func.__doc__)

    @property
    def input_array(self):
        return object.__getattribute__(self, '_input_array')

    @property
    def output_array(self):
        return object.__getattribute__(self, '_output_array')

    @property
    def func(self):
        return object.__getattribute__(self, '_func')

    def __call__(self, input_array=None, output_array=None, **kw):
        if input_array is not None:
            self.input_array[...] = input_array

        self.func(None, None, **kw)

        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        return self.output_array


@inheritdocstrings
class LegendreBase(SpectralBase):
    """Base class for all Legendre bases

    args:
        N             int         Number of quadrature points
        quad      ('LG', 'GL')    Legendre-Gauss or Legendre-Gauss-Lobatto
        domain   (float, float)   The computational domain

    """

    def __init__(self, N=0, quad="LG", domain=(-1., 1.)):
        SpectralBase.__init__(self, N, quad, domain=domain)

    def points_and_weights(self, N, scaled=False):
        if self.quad == "LG":
            points, weights = leg.leggauss(N)
        elif self.quad == "GL":
            points, weights = legendre_lobatto_nodes_and_weights(N)
        else:
            raise NotImplementedError

        if scaled is True:
            a, b = self.domain
            points = a+(b-a)/2+points*(b-a)/2
        return points, weights

    def vandermonde(self, x):
        """Return Legendre Vandermonde matrix

        args:
            x               points for evaluation

        """
        V = leg.legvander(x, self.N-1)
        return V

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivatives of basis as a Vandermonde matrix

        args:
            V               Legendre Vandermonde matrix

        kwargs:
            k     integer   Use k'th derivative of basis

        """
        assert self.N == V.shape[1]
        if k > 0:
            D = np.zeros((self.N, self.N))
            D[:-k, :] = leg.legder(np.eye(self.N), k)
            #a, b = self.domain
            V = np.dot(V, D) #*(2./(b-a))**k

        return self.get_vandermonde_basis(V)

    def get_mass_matrix(self):
        from .matrices import mat
        return mat[(self.__class__, 0), (self.__class__, 0)]

    def _get_mat(self):
        from .matrices import mat
        return mat

    def domain_factor(self):
        a, b = self.domain
        if abs(b-a-2) < 1e-12:
            return 1
        return 2./(b-a)

    def forward(self, input_array=None, output_array=None, fast_transform=False):
        """Fast forward transform

        args:
            input_array    (input)     Function values on quadrature mesh
            output_array   (output)    Expansion coefficients

        kwargs:
            fast_transform   bool - If True use fast transforms,
                             if False use Vandermonde type

        """
        assert fast_transform is False
        if input_array is not None:
            self.forward.input_array[...] = input_array

        self.scalar_product(fast_transform=fast_transform)
        self.apply_inverse_mass(self.forward.output_array)

        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        else:
            return self.forward.output_array


    def backward(self, input_array=None, output_array=None, fast_transform=False):
        """Fast backward transform

        args:
            input_array    (input)     Expansion coefficients
            output_array   (output)    Function values on quadrature mesh

        kwargs:
            fast_transform   bool - If True use fast transforms,
                             if False use Vandermonde type

        """
        assert fast_transform is False
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self.vandermonde_evaluate_expansion_all(self.backward.input_array,
                                                self.backward.output_array)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        else:
            return self.backward.output_array

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        assert fast_transform is False
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.vandermonde_scalar_product(self.scalar_product.input_array,
                                        self.scalar_product.output_array)

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        else:
            return self.scalar_product.output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            axis = axis[0]

        if isinstance(self.forward, _Wrap):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        if isinstance(axis, tuple):
            axis = axis[0]

        U = pyfftw.empty_aligned(shape, dtype=dtype)
        V = pyfftw.empty_aligned(shape, dtype=dtype)
        U.fill(0)
        V.fill(0)

        self.axis = axis
        self.forward = _Wrap(self.forward, U, V)
        self.backward = _Wrap(self.backward, V, U)
        self.scalar_product = _Wrap(self.scalar_product, U, V)


@inheritdocstrings
class Basis(LegendreBase):
    """Basis for regular Legendre series

    kwargs:
        N             int         Number of quadrature points
        quad        ('LG',)       Legendre-Gauss
        plan          bool        Plan transforms on __init__ or not. If
                                  basis is part of a TensorProductSpace,
                                  then planning needs to be delayed.
        domain   (float, float)   The computational domain

    """

    def __init__(self, N=0, quad="GL", plan=False, domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        if plan:
            self.plan(N, 0, np.float, {})

    def eval(self, x, fk, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = leg.legval(x, fk)
        return output_array


@inheritdocstrings
class ShenDirichletBasis(LegendreBase):
    """Shen Legendre basis for Dirichlet boundary conditions

    kwargs:
        N             int         Number of quadrature points
        quad        ('LG',)       Legendre-Gauss
        bc           (a, b)       Boundary conditions at x=(1,-1)
        plan          bool        Plan transforms on __init__ or not. If
                                  basis is part of a TensorProductSpace,
                                  then planning needs to be delayed.
        domain   (float, float)   The computational domain
        scaled        bool        Whether or not to scale test functions
                                  with 1/sqrt(4k+6). Scaled test functions
                                  give a stiffness matrix equal to the
                                  identity matrix.

    """

    def __init__(self, N=0, quad="LG", bc=(0., 0.), plan=False,
                 domain=(-1., 1.), scaled=False):
        LegendreBase.__init__(self, N, quad, domain=domain)
        from shenfun.tensorproductspace import BoundaryValues
        self.LT = Basis(N, quad)
        self._scaled = scaled
        self._factor = np.ones(1)
        if plan:
            self.plan(N, 0, np.float, {})
        self.bc = BoundaryValues(self, bc=bc)

    def set_factor_array(self, v):
        if self.is_scaled():
            if not self._factor.shape == v.shape:
                k = self.wavenumbers(v.shape, self.axis).astype(np.float)
                self._factor = 1./np.sqrt(4*k+6)

    def is_scaled(self):
        return self._scaled

    def get_vandermonde_basis(self, V):
        P = np.zeros(V.shape)
        if not self.is_scaled():
            P[:, :-2] = V[:, :-2] - V[:, 2:]
        else:
            k = np.arange(self.N-2).astype(np.float)
            P[:, :-2] = (V[:, :-2] - V[:, 2:])/np.sqrt(4*k+6)
        P[:, -2] = (V[:, 0] + V[:, 1])/2
        P[:, -1] = (V[:, 0] - V[:, 1])/2
        return P

    def forward(self, input_array=None, output_array=None, fast_transform=False):
        assert fast_transform is False
        if input_array is not None:
            self.forward.input_array[...] = input_array

        output = self.scalar_product(fast_transform=fast_transform)
        assert output is self.forward.output_array

        self.apply_inverse_mass(output)

        assert output is self.forward.output_array
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        else:
            return self.forward.output_array

    def evaluate_expansion_all(self, input_array, output_array):
        w_hat = work[(input_array, 0)]
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        self.set_factor_array(input_array)
        w_hat[s0] = input_array[s0]*self._factor
        w_hat[s1] -= input_array[s0]*self._factor
        self.bc.apply_before(w_hat, False, (0.5, 0.5))

        output_array = self.LT.backward(w_hat)
        assert input_array is self.backward.input_array
        assert output_array is self.backward.output_array
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def spectral_shape(self):
        return self.N-2

    def eval(self, x, fk, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        output_array[:] = leg.legval(x, fk[:-2]*self._factor)
        w_hat[2:] = fk[:-2]*self._factor
        output_array -= leg.legval(x, w_hat)
        output_array += 0.5*(fk[-1]*(1+x)+fk[-2]*(1-x))
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            axis = axis[0]

        if isinstance(self.forward, _Wrap):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        self.axis = self.LT.axis
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.forward = _Wrap(self.forward, U, V)
        self.backward = _Wrap(self.backward, V, U)
        self.scalar_product = _Wrap(self.scalar_product, U, V)


@inheritdocstrings
class ShenNeumannBasis(LegendreBase):
    """Shen basis for homogeneous Neumann boundary conditions

    kwargs:
        N             int         Number of quadrature points
        quad        ('LG')        Legendre-Gauss
        mean         float        Mean value
        plan          bool        Plan transforms on __init__ or not. If
                                  basis is part of a TensorProductSpace,
                                  then planning needs to be delayed.
        domain   (float, float)   The computational domain

    """

    def __init__(self, N=0, quad="LG", mean=0, plan=False, domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.mean = mean
        self.LT = Basis(N, quad)
        self._factor = np.zeros(0)
        if plan:
            self.plan(N, 0, np.float, {})

    def get_vandermonde_basis(self, V):
        assert self.N == V.shape[1]
        P = np.zeros(V.shape)
        k = np.arange(self.N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]*(k[:-2]+1)/(k[:-2]+2))/(k[:-2]+3)*V[:, 2:]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            k = self.wavenumbers(v.shape, self.axis).astype(np.float)
            self._factor = k*(k+1)/(k+2)/(k+3)

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        assert fast_transform is False
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.vandermonde_scalar_product(self.scalar_product.input_array,
                                        self.scalar_product.output_array)

        fk = self.scalar_product.output_array
        s = self.sl(0)
        fk[s] = self.mean*np.pi
        s[self.axis] = slice(-2, None)
        fk[s] = 0

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        else:
            return self.scalar_product.output_array

    def evaluate_expansion_all(self, input_array, output_array):
        w_hat = work[(input_array, 0)]
        self.set_factor_array(input_array)
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= self._factor*input_array[s0]
        output_array = self.LT.backward(w_hat)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def spectral_shape(self):
        return self.N-2

    def eval(self, x, fk, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        output_array[:] = leg.legval(x, fk[:-2])
        w_hat[2:] = self._factor*fk[:-2]
        output_array -= leg.legval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            axis = axis[0]

        if isinstance(self.forward, _Wrap):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        self.axis = self.LT.axis
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.forward = _Wrap(self.forward, U, V)
        self.backward = _Wrap(self.backward, V, U)
        self.scalar_product = _Wrap(self.scalar_product, U, V)


@inheritdocstrings
class ShenBiharmonicBasis(LegendreBase):
    """Shen biharmonic basis

    Homogeneous Dirichlet and Neumann boundary conditions.

    kwargs:
        N             int         Number of quadrature points
        quad        ('LG',)       Legendre-Gauss
        plan          bool        Plan transforms on __init__ or not. If
                                  basis is part of a TensorProductSpace,
                                  then planning needs to be delayed.
        domain   (float, float)   The computational domain

    """

    def __init__(self, N=0, quad="LG", plan=False, domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.LT = Basis(N, quad)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        if plan:
            self.plan(N, 0, np.float, {})

    def get_vandermonde_basis(self, V):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(2*k+5)/(2*k+7))*V[:, 2:-2] + ((2*k+3)/(2*k+7))*V[:, 4:]
        return P

    def set_factor_arrays(self, v):
        s = [slice(None)]*v.ndim
        s[self.axis] = self.slice()
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers(v.shape, axis=self.axis).astype(np.float)
            self._factor1 = (-2*(2*k+5)/(2*k+7)).astype(float)
            self._factor2 = ((2*k+3)/(2*k+7)).astype(float)

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        assert fast_transform is False
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        output = self.vandermonde_scalar_product(self.scalar_product.input_array,
                                                 self.scalar_product.output_array)

        output[self.sl(slice(-4, None))] = 0

        if output_array is not None:
            output_array[...] = output
            return output_array
        else:
            return output

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2):
        s = self.sl(self.slice())
        s2 = self.sl(slice(2, -2))
        s4 = self.sl(slice(4, None))
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def evaluate_expansion_all(self, input_array, output_array):
        w_hat = work[(input_array, 0)]
        self.set_factor_arrays(input_array)
        w_hat = self.set_w_hat(w_hat, input_array, self._factor1, self._factor2)
        output_array = self.LT.backward(w_hat)
        assert input_array is self.backward.input_array
        assert output_array is self.backward.output_array
        return output_array

    def slice(self):
        return slice(0, self.N-4)

    def spectral_shape(self):
        return self.N-4

    def eval(self, x, fk, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        w_hat = work[(fk, 0)]
        self.set_factor_arrays(fk)
        output_array[:] = leg.legval(x, fk[:-4])
        w_hat[2:-2] = self._factor1*fk[:-4]
        output_array += leg.legval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*fk[:-4]
        w_hat[:4] = 0
        output_array += leg.legval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            axis = axis[0]

        if isinstance(self.forward, _Wrap):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        self.axis = self.LT.axis
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.forward = _Wrap(self.forward, U, V)
        self.backward = _Wrap(self.backward, V, U)
        self.scalar_product = _Wrap(self.scalar_product, U, V)


## Experimental!
@inheritdocstrings
class SecondNeumannBasis(LegendreBase):
    """Shen basis for homogeneous second order Neumann boundary conditions

    kwargs:
        N             int         Number of quadrature points
        quad        ('LG')        Legendre-Gauss
        mean         float        Mean value
        plan          bool        Plan transforms on __init__ or not. If
                                  basis is part of a TensorProductSpace,
                                  then planning needs to be delayed.
        domain   (float, float)   The computational domain

    """

    def __init__(self, N=0, quad="LG", mean=0, plan=False, domain=(-1., 1.)):
        LegendreBase.__init__(self, N, quad, domain=domain)
        self.mean = mean
        self.LT = Basis(N, quad)
        self._factor = np.zeros(0)
        if plan:
            self.plan(N, 0, np.float, {})

    def get_vandermonde_basis(self, V):
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
            k = self.wavenumbers(v.shape, self.axis).astype(np.float)
            self._factor = -(k+1)*(k+2)*(2*k+3)/((k+3)*(k+4)*(2*k+7))

    def scalar_product(self, input_array=None, output_array=None, fast_transform=False):
        assert fast_transform is False
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.vandermonde_scalar_product(self.scalar_product.input_array,
                                        self.scalar_product.output_array)

        #fk = self.scalar_product.output_array
        #s = self.sl(0)
        #fk[s] = self.mean*np.pi
        #s[self.axis] = slice(-2, None)
        #fk[s] = 0

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        else:
            return self.scalar_product.output_array

    #def evaluate_expansion_all(self, fk, output_array):
        #w_hat = work[(fk, 0)]
        #self.set_factor_array(fk)
        #s0 = self.sl(slice(0, -4))
        #s1 = self.sl(slice(2, -2))
        #s2 = self.sl(slice(4, None))
        #w_hat[s0] = fk[s0]
        #w_hat[s1] += (self._factor-1)*fk[s0]
        #w_hat[s2] -= self._factor*fk[s0]
        #output_array = self.LT.backward(w_hat)
        #return output_array

    def slice(self):
        return slice(0, self.N-2)

    def spectral_shape(self):
        return self.N-2

    #def eval(self, x, input_array):
        #w_hat = work[(input_array, 0)]
        #self.set_factor_array(input_array)
        #f = leg.legval(x, input_array[:-2])
        #w_hat[2:] = self._factor*input_array[:-2]
        #f -= leg.legval(x, w_hat)
        #return f

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            axis = axis[0]

        if isinstance(self.forward, _Wrap):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.LT.plan(shape, axis, dtype, options)
        self.axis = self.LT.axis
        U, V = self.LT.forward.input_array, self.LT.forward.output_array
        self.forward = _Wrap(self.forward, U, V)
        self.backward = _Wrap(self.backward, V, U)
        self.scalar_product = _Wrap(self.scalar_product, U, V)
