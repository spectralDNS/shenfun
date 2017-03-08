r"""
This module contains classes for working with the spectral-Galerkin method

There are classes for 8 bases and corresponding function spaces

All bases have expansions

    u(x_j) = \sum_k \hat{u}_k \phi_k

where j = 0, 1, ..., N and k = indexset(basis), and the indexset differs from
base to base, see function space definitions below. \phi_k is the k't basis
function of the basis span(\phi_k, k=indexset(basis)).

Chebyshev:
    ChebyshevBasis:
        basis functions:                 basis:
        \phi_k = T_k                     span(T_k, k=0,1,..., N)

    ShenDirichletBasis:
        basis functions:                 basis:
        \phi_k = T_k-T_{k+2}             span(\phi_k, k=0,1,...,N)
        \phi_{N-1} = 0.5(T_0+T_1)
        \phi_{N} = 0.5(T_0-T_1)

        u(1)=a, u(-1)=b, \hat{u}{N-1}=a, \hat{u}_{N}=b

        Note that there are only N-1 unknown coefficients, \hat{u}_k, since
        \hat{u}_{N-1} and \hat{u}_{N} are determined by boundary conditions.

    ShenNeumannBasis:
        basis function:                  basis:
        \phi_k = T_k-(k/(k+2))**2T_{k+2} span(\phi_k, k=1,2,...,N-2)

        Homogeneous Neumann boundary conditions, u'(\pm 1) = 0, and
        zero weighted mean: \int_{-1}^{1}u(x)w(x)dx = 0

    ShenBiharmonicBasis:
        basis function:
        \phi_k = T_k - (2*(k+2)/(k+3))*T_{k+2} + ((k+1)/(k+3))*T_{k+4}

        basis:
        span(\phi_k, k=0,1,...,N-4)

        Homogeneous Dirichlet and Neumann, u(\pm 1)=0 and u'(\pm 1)=0

Legendre:
    LegendreBasis:
        basis function:                  basis:
        \phi_k = L_k                     span(L_k, k=0,1,...N)

    ShenDirichletBasis:
        basis function:                  basis:
        \phi_k = L_k-L_{k+2}             span(\phi_k, k=0,1,...,N)
        \phi_{N-1} = 0.5(L_0+L_1)
        \phi_{N} = 0.5(L_0-L_1)

        u(1)=a, u(-1)=b, \hat{u}{N-1}=a, \hat{u}_{N}=b

        Note that there are only N-1 unknown coefficients, \hat{u}_k, since
        \hat{u}_{N-1} and \hat{u}_{N} are determined by boundary conditions.

    ShenNeumannBasis:
        basis function:                  basis:
        \phi_k = L_k-(k(k+1)/(k+2)/(k+3))L_{k+2} span(\phi_k, k=1,2,...,N-2)

        Homogeneous Neumann boundary conditions, u'(\pm 1) = 0, and
        zero mean: \int_{-1}^{1}u(x)dx = 0

    ShenBiharmonicBasis:
        basis function:
        \phi_k = L_k - (2*(2k+5)/(2k+7))*L_{k+2} + ((2k+3)/(2k+7))*L_{k+4}

        basis:
        span(\phi_k, k=0,1,...,N-4)

        Homogeneous Dirichlet and Neumann, u(\pm 1)=0 and u'(\pm 1)=0


Each class has methods for moving fast between spectral and physical space, and
for computing the (weighted) scalar product.

"""
import numpy as np
import pyfftw
from .utilities import inheritdocstrings
from mpiFFT4py import work_arrays

work = work_arrays()

class SpectralBase(object):
    """Abstract base class for all spectral function spaces

    args:
        N             int          Number of quadrature points
        quad   ('GL', 'GC', 'LG')  Chebyshev-Gauss-Lobatto, Chebyshev-Gauss
                                   or Legendre-Gauss

    """

    def __init__(self, N, quad):
        self.N = N
        self.quad = quad
        self._mass = None # Mass matrix (if needed)
        self.axis = 0
        self.xfftn_fwd = None
        self.xfftn_bck = None

    def points_and_weights(self):
        """Return points and weights of quadrature"""
        raise NotImplementedError

    def mesh(self, N, axis=0):
        """Return the computational mesh

        All dimensions, except axis, are obtained through broadcasting.

        """
        N = list(N) if np.ndim(N) else [N]
        assert self.N == N[axis]
        x = self.points_and_weights()[0]
        X = self.broadcast_to_ndims(x, len(N), axis)
        return X

    def wavenumbers(self, N, axis=0):
        """Return the wavenumbermesh

        All dimensions, except axis, are obtained through broadcasting.

        """
        N = list(N) if np.ndim(N) else [N]
        assert self.N == N[axis]
        s = self.slice()
        k = np.arange(s.start, s.stop)
        K = self.broadcast_to_ndims(k, len(N), axis)
        return K

    def broadcast_to_ndims(self, x, ndims, axis=0):
        s = [np.newaxis]*ndims
        s[axis] = slice(None)
        return x[s]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        r"""Return scalar product

          f_k = (f, \phi_k)_w      for all k = 0, 1, ..., N
              = \sum_j f(x_j) \phi_k(x_j) \sigma(x_j)

        args:
            input_array    (input)     Function values on quadrature mesh
            output_array   (output)    Expansion coefficients

        """
        raise NotImplementedError

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        """Fast forward transform

        args:
            input_array    (input)     Function values on quadrature mesh
            output_array   (output)    Expansion coefficients

        kwargs:
            fast_transform   bool - If True use fast transforms,
                             if False use Vandermonde type

        """
        if input_array is not None:
            self.xfftn_fwd.input_array[...] = input_array

        self.scalar_product(fast_transform=fast_transform)
        self.apply_inverse_mass(self.xfftn_fwd.output_array)

        if output_array is not None:
            output_array[...] = self.xfftn_fwd.output_array
            return output_array
        else:
            return self.xfftn_fwd.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        """Fast backward transform

        args:
            input_array    (input)     Expansion coefficients
            output_array   (output)    Function values on quadrature mesh

        kwargs:
            fast_transform   bool - If True use fast transforms,
                             if False use Vandermonde type

        """
        if input_array is not None:
            self.xfftn_bck.input_array[...] = input_array

        if fast_transform:
            self.evaluate_expansion_all(self.xfftn_bck.input_array,
                                        self.xfftn_bck.output_array)
        else:
            self.vandermonde_evaluate_expansion_all(self.xfftn_bck.input_array,
                                                    self.xfftn_bck.output_array)

        if output_array is not None:
            output_array[...] = self.xfftn_bck.output_array
            return output_array
        else:
            return self.xfftn_bck.output_array


    def vandermonde(self, x):
        """Return Vandermonde matrix

        args:
            x               points for evaluation

        """
        raise NotImplementedError

    def get_vandermonde_basis(self, V):
        """Return basis as a Vandermonde matrix

        V is a Vandermonde matrix

        """
        return V

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivative of basis as a Vandermonde matrix

        args:
            V               Vandermonde matrix

        kwargs:
            k    integer    k'th derivative

        """
        raise NotImplementedError

    def vandermonde_scalar_product(self, input_array, output_array):
        """Naive implementation of scalar product

        args:
            input_array   (input)    Function values on quadrature mesh
            output_array   (output)   Expansion coefficients

        """
        assert self.N == input_array.shape[self.axis]
        points, weights = self.points_and_weights()
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)
        if input_array.ndim == 1:
            output_array[:] = np.dot(input_array*weights, np.conj(P))

        else: # broadcasting
            bc_shape = [np.newaxis,]*input_array.ndim
            bc_shape[self.axis] = slice(None)
            fc = np.moveaxis(input_array*weights[bc_shape], self.axis, -1)
            output_array[:] = np.moveaxis(np.dot(fc, np.conj(P)), -1, self.axis)

        assert output_array is self.xfftn_fwd.output_array
        return output_array

    def vandermonde_evaluate_expansion_all(self, input_array, output_array):
        """Naive implementation of evaluate_expansion_all

        args:
            input_array   (input)    Expansion coefficients
            output_array   (output)   Function values on quadrature mesh

        """
        assert self.N == output_array.shape[self.axis]
        points = self.points_and_weights()[0]
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)

        if output_array.ndim == 1:
            output_array = np.dot(P, input_array, out=output_array)
        else:
            fc = np.moveaxis(input_array, self.axis, -2)
            array = np.dot(P, fc)
            output_array[:] = np.moveaxis(array, 0, self.axis)

        assert output_array is self.xfftn_bck.output_array
        return output_array

    def apply_inverse_mass(self, array):
        """Apply inverse mass matrix

        args:
            array   (input/output)    Expansion coefficients. Overwritten by
                                      applying the inverse mass matrix, and
                                      returned.

        """
        B = self.get_mass_matrix()
        if self._mass is None:
            assert self.N == array.shape[self.axis]
            self._mass = B((self, 0), (self, 0))

        if (self._mass.testfunction[0].quad != self.quad or
            self._mass.testfunction[0].N != array.shape[self.axis]):
            self._mass = B((self, 0), (self, 0))

        array = self._mass.solve(array, axis=self.axis, sol=self)

        return array

    def plan(self, shape, axis, dtype, options):
        opts = dict(
            avoid_copy=True,
            overwrite_input=False,
            auto_align_input=True,
            auto_contiguous=True,
            planner_effort='FFTW_MEASURE',
            threads=1,
        )
        opts.update(options)

        plan_fwd = self.xfftn_fwd
        plan_bck = self.xfftn_bck
        if isinstance(axis, tuple):
            axis = axis[0]

        U = pyfftw.empty_aligned(shape, dtype=dtype)
        xfftn_fwd = plan_fwd(U, axis=axis, **opts)
        U.fill(0)
        V = xfftn_fwd.output_array
        xfftn_bck = plan_bck(V, axis=axis, **opts)
        V.fill(0)

        xfftn_fwd.update_arrays(U, V)
        xfftn_bck.update_arrays(V, U)

        self.axis = axis
        self.xfftn_fwd = xfftn_fwd
        self.xfftn_bck = xfftn_bck
        self.forward = _func_wrap(self.forward, xfftn_fwd)
        self.backward = _func_wrap(self.backward, xfftn_bck)
        self.scalar_product = _func_wrap(self.scalar_product, xfftn_fwd)


    def evaluate_expansion_all(self, input_array, output_array):
        r"""Evaluate expansion on entire mesh

           f(x_j) = \sum_k f_k \T_k(x_j)  for all j = 0, 1, ..., N

        args:
            input_array   (input)     Expansion coefficients
            output_array   (output)    Function values on quadrature mesh

        """
        raise NotImplementedError

    def eval(self, x, fk):
        """Evaluate basis at position x

        args:
            x    float or array of floats
            fk   Array of expansion coefficients

        """
        raise NotImplementedError

    def get_mass_matrix(self):
        """Return mass matrix associated with current basis"""
        raise NotImplementedError

    def slice(self):
        """Return index set of current basis, with N points in real space"""
        return slice(0, self.N)

    def spectral_shape(self):
        """Return the shape of current basis used to build a ShenMatrix"""
        s = self.slice()
        return s.stop - s.start

    def __hash__(self):
        return hash(repr(self.__class__))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def sl(self, a):
        s = [slice(None)]*self.xfftn_fwd.output_array.ndim
        s[self.axis] = a
        return s


def _transform_exec(func_obj, in_array, out_array, xfftn_obj, options):
    if in_array is not None:
        xfftn_obj.input_array[...] = in_array
    func_obj(None, None, **options)
    if out_array is not None:
        out_array[...] = xfftn_obj.output_array
        return out_array
    else:
        return xfftn_obj.output_array


class _func_wrap(object):

    # pylint: disable=too-few-public-methods

    __slots__ = ('_func', '_xfftn', '__doc__')

    def __init__(self, func, xfftn):
        object.__setattr__(self, '_xfftn', xfftn)
        object.__setattr__(self, '_func', func)

    def __getattribute__(self, name):
        if name in ('input_array', 'output_array'):
            xfftn = object.__getattribute__(self, '_xfftn')
            return getattr(xfftn, name)
        else:
            func = object.__getattribute__(self, '_func')
            return getattr(func, name)

    def __call__(self, input_array=None, output_array=None, **kw):
        func_obj = object.__getattribute__(self, '_func')
        xfftn_obj = object.__getattribute__(self, '_xfftn')
        return _transform_exec(func_obj, input_array, output_array, xfftn_obj, kw)
