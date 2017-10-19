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

    def __init__(self, N, quad, padding_factor=1, domain=(-1., 1.)):
        self.N = N
        self.domain = domain
        self.quad = quad
        self._mass = None # Mass matrix (if needed)
        self.axis = 0
        self.xfftn_fwd = None
        self.xfftn_bck = None
        self.padding_factor = np.floor(N*padding_factor)/N

    def points_and_weights(self, N):
        """Return points and weights of quadrature"""
        raise NotImplementedError

    def mesh(self, N, axis=0):
        """Return the computational mesh

        All dimensions, except axis, are obtained through broadcasting.

        """
        N = list(N) if np.ndim(N) else [N]
        x = self.points_and_weights(N[axis], scaled=True)[0]
        X = self.broadcast_to_ndims(x, len(N), axis)
        return X

    def wavenumbers(self, N, axis=0, **kw):
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
        """Scalar product

        kwargs:
            input_array    (input)     Function values on quadrature mesh
            output_array   (output)    Expansion coefficients
            fast_transform   bool      If True use fast transforms, if False
                                       use Vandermonde type

        If kwargs input_array/output_array are not given, then use predefined
        arrays as planned with self.plan

        """
        raise NotImplementedError

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        """Forward transform

        kwargs:
            input_array    (input)     Function values on quadrature mesh
            output_array   (output)    Expansion coefficients
            fast_transform   bool      If True use fast transforms, if False
                                       use Vandermonde type

        If kwargs input_array/output_array are not given, then use predefined
        arrays as planned with self.plan

        """
        if input_array is not None:
            self.forward.input_array[...] = input_array

        self.scalar_product(fast_transform=fast_transform)
        self.apply_inverse_mass(self.xfftn_fwd.output_array)
        self._truncation_forward(self.xfftn_fwd.output_array,
                                 self.forward.output_array)

        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        else:
            return self.forward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        """Inverse transform

        kwargs:
            input_array    (input)     Function values on quadrature mesh
            output_array   (output)    Expansion coefficients
            fast_transform   bool      If True use fast transforms, if False
                                       use Vandermonde type

        If kwargs input_array/output_array are not given, then use predefined
        arrays as planned with self.plan

        """
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self._padding_backward(self.backward.input_array,
                               self.xfftn_bck.input_array)

        if fast_transform:
            self.evaluate_expansion_all(self.xfftn_bck.input_array,
                                        self.backward.output_array)
        else:
            self.vandermonde_evaluate_expansion_all(self.xfftn_bck.input_array,
                                                    self.backward.output_array)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        else:
            return self.backward.output_array


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
        points, weights = self.points_and_weights(self.N)
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)
        if input_array.ndim == 1:
            output_array[:] = np.dot(input_array*weights, np.conj(P))

        else: # broadcasting
            bc_shape = [np.newaxis,]*input_array.ndim
            bc_shape[self.axis] = slice(None)
            fc = np.moveaxis(input_array*weights[bc_shape], self.axis, -1)
            output_array[:] = np.moveaxis(np.dot(fc, np.conj(P)), -1, self.axis)

        assert output_array is self.forward.output_array
        return output_array

    def vandermonde_evaluate_expansion_all(self, input_array, output_array):
        """Naive implementation of evaluate_expansion_all

        args:
            input_array   (input)    Expansion coefficients
            output_array   (output)   Function values on quadrature mesh

        """
        assert self.N == output_array.shape[self.axis]
        points = self.points_and_weights(self.N)[0]
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)

        if output_array.ndim == 1:
            output_array = np.dot(P, input_array, out=output_array)
        else:
            fc = np.moveaxis(input_array, self.axis, -2)
            array = np.dot(P, fc)
            output_array[:] = np.moveaxis(array, 0, self.axis)

        assert output_array is self.backward.output_array
        assert input_array is self.backward.input_array
        return output_array

    def apply_inverse_mass(self, array):
        """Apply inverse mass matrix

        args:
            array   (input/output)    Expansion coefficients. Overwritten by
                                      applying the inverse mass matrix, and
                                      returned.

        """
        if self._mass is None:
            assert self.N == array.shape[self.axis]
            B = self.get_mass_matrix()
            self._mass = B((self, 0), (self, 0))

        if (self._mass.testfunction[0].quad != self.quad or
            self._mass.testfunction[0].N != array.shape[self.axis]):
            B = self.get_mass_matrix()
            self._mass = B((self, 0), (self, 0))

        array = self._mass.solve(array, axis=self.axis)
        return array

    def plan(self, shape, axis, dtype, options):
        """Plan transform

        Allocate work arrays for transforms and set up methods

          forward
          backward
          scalar_product

        with or without padding
        """
        if isinstance(axis, tuple):
            axis = axis[0]

        if isinstance(self.forward, _func_wrap):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        opts = dict(
            avoid_copy=True,
            overwrite_input=True,
            auto_align_input=True,
            auto_contiguous=True,
            planner_effort='FFTW_MEASURE',
            threads=1,
        )
        opts.update(options)

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        n = shape[axis]
        U = pyfftw.empty_aligned(shape, dtype=dtype)
        xfftn_fwd = plan_fwd(U, n=n, axis=axis, **opts)
        U.fill(0)
        V = xfftn_fwd.output_array
        xfftn_bck = plan_bck(V, n=n, axis=axis, **opts)
        V.fill(0)

        xfftn_fwd.update_arrays(U, V)
        xfftn_bck.update_arrays(V, U)

        self.axis = axis
        self.xfftn_fwd = xfftn_fwd
        self.xfftn_bck = xfftn_bck

        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = _func_wrap(self.forward, xfftn_fwd, U, trunc_array)
            self.backward = _func_wrap(self.backward, xfftn_bck, trunc_array, U)
        else:
            self.forward = _func_wrap(self.forward, xfftn_fwd, U, V)
            self.backward = _func_wrap(self.backward, xfftn_bck, V, U)

        # scalar_product is not padded, just the forward/backward
        self.scalar_product = _func_wrap(self.scalar_product, xfftn_fwd, U, V)

    def _get_truncarray(self, shape, dtype):
        shape = list(shape)
        shape[self.axis] = int(np.round(shape[self.axis] / self.padding_factor))
        return pyfftw.empty_aligned(shape, dtype=dtype)

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
        """Return the shape of current basis used to build a SpectralMatrix"""
        s = self.slice()
        return s.stop - s.start

    def domain_factor(self):
        return 1

    def __hash__(self):
        return hash(repr(self.__class__))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def sl(self, a):
        s = [slice(None)]*self.forward.output_array.ndim
        s[self.axis] = a
        return s

    def rank(self):
        return 1

    def ndim(self):
        return 1

    def num_components(self):
        return 1

    def __len__(self):
        return 1

    def __getitem__(self, i):
        assert i == 0
        return self

    def _get_mat(self):
        raise NotImplementedError

    def is_forward_output(self, u):
        return (np.all(u.shape == self.forward.output_array.shape) and
                u.dtype == self.forward.output_array.dtype)

    def as_function(self, u):
        from .forms.arguments import Function
        assert isinstance(u, np.ndarray)
        forward_output = self.is_forward_output(u)
        return Function(self, forward_output=forward_output, buffer=u)

    def _truncation_forward(self, padded_array, trunc_array):
        if self.padding_factor > 1.0+1e-8:
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            su = [slice(None)]*trunc_array.ndim
            su[self.axis] = slice(0, N//2+1)
            trunc_array[su] = padded_array[su]
            su[self.axis] = slice(-(N//2), None)
            trunc_array[su] += padded_array[su]

    def _padding_backward(self, trunc_array, padded_array):
        if self.padding_factor > 1.0+1e-8:
            padded_array.fill(0)
            N = trunc_array.shape[self.axis]
            su = [slice(None)]*trunc_array.ndim
            su[self.axis] = slice(0, np.ceil(N/2.).astype(np.int))
            padded_array[su] = trunc_array[su]
            su[self.axis] = slice(-(N//2), None)
            padded_array[su] = trunc_array[su]


def inner_product(test, trial, out=None, axis=0, fast_transform=False):
    """Return inner product of linear or bilinear form

    args:
        test     (Basis, integer)     Basis is any of the classes from
                                      shenfun.chebyshev.bases,
                                      shenfun.legendre.bases or
                                      shenfun.fourier.bases
                                      The integer determines the numer of times
                                      the basis is differentiated.
                                      The test represents the matrix row
        trial    (Basis, integer)     As test, but representing matrix column
                       or
                    function          Function evaluated at quadrature nodes
                                      (for linear forms)

    kwargs:
        out              Numpy array  Return array
        axis             int          Axis to take the inner product over
        fast_transform   bool         Use fast transform method if True

    Example:
        Compute mass matrix of Shen's Chebyshev Dirichlet basis:

        >>> from shenfun.chebyshev.bases import ShenDirichletBasis
        >>> SD = ShenDirichletBasis(6)
        >>> B = inner_product((SD, 0), (SD, 0))
        >>> B
        {-2: array([-1.57079633]),
          0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265]),
          2: array([-1.57079633])}

    """
    if isinstance(test, tuple):
        # Bilinear form
        assert trial[0].__module__ == test[0].__module__
        if isinstance(test[1], (int, np.integer)):
            key = ((test[0].__class__, test[1]), (trial[0].__class__, trial[1]))
        elif isinstance(test[1], np.ndarray):
            assert len(test[1]) == 1
            k_test = test[1][(0,)*np.ndim(test[1])]
            k_trial = trial[1][(0,)*np.ndim(trial[1])]
            key = ((test[0].__class__, k_test), (trial[0].__class__, k_trial))
        else:
            raise RuntimeError

        mat = test[0]._get_mat()
        m = mat[key](test, trial)
        return m

    else:
        # Linear form
        if out is None:
            sl = list(trial.shape)
            if isinstance(test, fourier.FourierBase):
                if isinstance(test, fourier.R2CBasis):
                    sl[axis] = sl[axis]//2+1
                out = np.zeros(sl, dtype=np.complex)
            else:
                out = np.zeros_like(trial)
        out = test.scalar_product(trial, out, fast_transform=fast_transform)
        return out


class _func_wrap(object):

    # pylint: disable=too-few-public-methods

    __slots__ = ('_func', '_xfftn', '__doc__', '_input_array', '_output_array')

    def __init__(self, func, xfftn, input_array, output_array):
        object.__setattr__(self, '_xfftn', xfftn)
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
    def xfftn(self):
        return object.__getattribute__(self, '_xfftn')

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
        else:
            return self.output_array

