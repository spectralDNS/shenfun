r"""
This module contains classes for working with the spectral-Galerkin method

There are currently classes for 9 bases and corresponding function spaces

All bases have expansions

   :math:`u(x_j) = \sum_k \hat{u}_k \phi_k`

where j = 0, 1, ..., N and k = indexset(basis), and the indexset differs from
base to base, see function space definitions below. :math:`\phi_k` is the k't basis
function of the basis span(:math:`\phi_k`, k=indexset(basis)).

Chebyshev:
    ChebyshevBasis:
        basis functions: :math:`\phi_k = T_k`

        basis: :math:`span(T_k, k=0,1,..., N)`

    ShenDirichletBasis:
        basis functions:
            :math:`\phi_k = T_k-T_{k+2}`
            
            :math:`\phi_{N-1} = 0.5(T_0+T_1)` for Poisson's equation
            
            :math:`\phi_{N} = 0.5(T_0-T_1)` for Poisson's equation
        
        basis: :math:`span(\phi_k, k=0,1,...,N)`
 
        :math:`u(1)=a, u(-1)=b, \hat{u}_{N-1}=a, \hat{u}_{N}=b`

        Note that there are only N-1 unknown coefficients, :math:`\hat{u}_k`, since
        :math:`\hat{u}_{N-1}` and :math:`\hat{u}_{N}` are determined by boundary conditions.
        Inhomogeneous boundary conditions are possible for the Poisson
        equation, because :math:`\phi_{N}` and :math:`\phi_{N-1}` are in the kernel of
        the Poisson operator.

    ShenNeumannBasis:
        basis function:
            :math:`\phi_k = T_k-\left(\frac{k}{k+2}\right)^2T_{k+2}`

        basis:
            :math:`span(\phi_k, k=1,2,...,N-2)`

        Homogeneous Neumann boundary conditions, :math:`u'(\pm 1) = 0`, and
        zero weighted mean: :math:`\int_{-1}^{1}u(x)w(x)dx = 0`

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = T_k - \frac{2(k+2)}{k+3}T_{k+2} + \frac{k+1}{k+3}T_{k+4}`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-4)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and :math:`u'(\pm 1)=0`

Legendre:
    LegendreBasis:
        basis function: 
            :math:`\phi_k = L_k`                    

        basis:
            :math:`span(L_k, k=0,1,...N)`

    ShenDirichletBasis:
        basis function:
            :math:`\phi_k = L_k-L_{k+2}`             

        basis:
            :math:`span(\phi_k, k=0,1,...,N)`
            :math:`\phi_{N-1} = 0.5(L_0+L_1)`
            :math:`\phi_{N} = 0.5(L_0-L_1)`

        :math:`u(1)=a, u(-1)=b, \hat{u}_{N-1}=a, \hat{u}_{N}=b`

        Note that there are only N-1 unknown coefficients, :math:`\hat{u}_k`, since
        :math:`\hat{u}_{N-1}` and :math:`\hat{u}_{N}` are determined by boundary conditions.

    ShenNeumannBasis:
        basis function:
            :math:`\phi_k = L_k-\frac{k(k+1)}{(k+2)(k+3)}L_{k+2}`

        basis:
            :math:`span(\phi_k, k=1,2,...,N-2)`

        Homogeneous Neumann boundary conditions, :math:`u'(\pm 1) = 0`, and
        zero mean: :math:`\int_{-1}^{1}u(x)dx = 0`

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = L_k - \frac{2(2k+5)}{2k+7}L_{k+2} + \frac{2k+3}{2k+7}L_{k+4}`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-4)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and :math:`u'(\pm 1)=0`

Fourier:
    R2CBasis and C2CBasis:
        basis function:
            :math:`\phi_k = c_k exp(ikx)`

        basis:
            :math:`span(\phi_k, k=-N/2, -N/2+1, ..., N/2)`

        If N is even, then :math:`c_{-N/2}` and :math:`c_{N/2} = 0.5` and :math:`c_k = 1` for
        :math:`k=-N/2+1, ..., N/2-1`. :math:`i` is the imaginary unit.

        If N is odd, then :math:`c_k = 1` for :math:`k=-N/2, ..., N/2`

    R2CBasis and C2CBasis are the same, but R2CBasis is used on real physical
    data and it takes advantage of Hermitian symmetry,
    :math:`\hat{u}_{-k} = conj(\hat{u}_k)`, for :math:`k = 1, ..., N/2`


Each class has methods for moving fast between spectral and physical space, and
for computing the (weighted) scalar product.

"""
import numpy as np
import pyfftw
from mpiFFT4py import work_arrays

#pylint: disable=unused-argument, not-callable, no-self-use, protected-access, too-many-public-methods, missing-docstring

work = work_arrays()


class SpectralBase(object):
    """Abstract base class for all spectral function spaces

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str
               Type of quadrature
 
               GL - Chebyshev-Gauss-Lobatto
         
               GC - Chebyshev-Gauss

               LG - Legendre-Gauss
        padding_factor : float, optional
                         For padding backward transform (for dealiasing)
        domain : 2-tuple of floats, optional
                 The computational domain

    """
    # pylint: disable=method-hidden, too-many-instance-attributes

    def __init__(self, N, quad, padding_factor=1, domain=(-1., 1.)):
        self.N = N
        self.domain = domain
        self.quad = quad
        self._mass = None # Mass matrix (if needed)
        self.axis = 0
        self.xfftn_fwd = None
        self.xfftn_bck = None
        self._xfftn_fwd = None    # pyfftw forward transform function
        self._xfftn_bck = None    # pyfftw backward transform function
        self.padding_factor = np.floor(N*padding_factor)/N

    def points_and_weights(self, N, scaled=False):
        """Return points and weights of quadrature

        Parameters
        ----------
            N : int
                Number of quadrature points
            scaled : bool, optional
                     Whether or not to scale with domain size
        """
        raise NotImplementedError

    def mesh(self, N, axis=0):
        """Return the computational mesh

        All dimensions, except axis, are obtained through broadcasting.

        Parameters
        ---------
            N : int or list/array of ints
                May be a list/array of ints if base is part of a 
                TensorProductSpace with several dimensions
            axis : int, optional
                   The axis of this base in a TensorProductSpace
        """
        N = list(N) if np.ndim(N) else [N]
        x = self.points_and_weights(N[axis], scaled=True)[0]
        X = self.broadcast_to_ndims(x, len(N), axis)
        return X

    def wavenumbers(self, N, axis=0, **kw):
        """Return the wavenumbermesh

        All dimensions, except axis, are obtained through broadcasting.

        Parameters
        ----------
            N : int or list/array of ints
                If N is a float then we have a 1D array. If N is an array, then
                the wavenumber returned is a 1D array broadcasted to the shape 
                of N.
            axis : int, optional
                   The axis of this base in a TensorProductSpace
        """
        N = list(N) if np.ndim(N) else [N]
        assert self.N == N[axis]
        s = self.slice()
        k = np.arange(s.start, s.stop)
        K = self.broadcast_to_ndims(k, len(N), axis)
        return K

    @staticmethod
    def broadcast_to_ndims(x, ndims, axis=0):
        """Return 1D array x as an array of shape ndim

        Parameters
        ----------
            x : 1D array
            ndims : int
                    The number of dimensions to broadcast to
            axis : int
                   The axis over which x is changing

        Note
        ----
        The returned array has shape one in all ndims-1 dimensions apart
        from axis.
        
        Example
        -------
        >>> import numpy as np
        >>> x = np.arange(4)
        >>> y = broadcast_to_ndims(x, 4, axis=2)
        >>> print(y.shape)
        (1, 1, 4, 1)
        """
        s = [np.newaxis]*ndims
        s[axis] = slice(None)
        return x[s]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        """Compute weighted scalar product

        Parameters
        ----------
            input_array : array, optional
                          Function values on quadrature mesh
            output_array : array, optional
                           Expansion coefficients
            fast_transform : bool, optional
                             If True use fast transforms, if False use 
                             Vandermonde type

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

        """
        raise NotImplementedError

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        """Compute forward transform

        Parameters
        ----------
            input_array : array, optional
                          Function values on quadrature mesh
            output_array : array, optional
                           Expansion coefficients
            fast_transform : bool, optional
                             If True use fast transforms, if False use 
                             Vandermonde type

        Note
        ---- 
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

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
        return self.forward.output_array

    def backward(self, input_array=None, output_array=None, fast_transform=True):
        """Compute backward (inverse) transform

        Parameters
        ----------
            input_array : array, optional
                          Function values on quadrature mesh
            output_array : array, optional
                           Expansion coefficients
            fast_transform : bool, optional
                             If True use fast transforms, if False use 
                             Vandermonde type

        Note
        ---- 
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan 

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
        return self.backward.output_array

    def vandermonde(self, x):
        """Return Vandermonde matrix

        Parameters
        ----------
            x : array of floats
                points for evaluation

        """
        raise NotImplementedError

    def get_vandermonde_basis(self, V):
        """Return basis as a Vandermonde matrix

        Parameters
        ----------
            V : 2D array
                Vandermonde matrix

        """
        return V

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivative of basis as a Vandermonde matrix

        Parameters
        ----------
            V : 2D array
                Vandermonde matrix
            k : int
                k'th derivative

        """
        raise NotImplementedError

    def vandermonde_scalar_product(self, input_array, output_array):
        """Naive implementation of scalar product

        Parameters
        ----------
            input_array : array
                          Function values on quadrature mesh
            output_array : array
                           Expansion coefficients

        """
        assert abs(self.padding_factor-1) < 1e-8
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

        Parameters
        ----------
            input_array : array
                          Expansion coefficients
            output_array : array
                           Function values on quadrature mesh

        Note
        ----
        Implemented only for non-padded transforms

        """
        assert abs(self.padding_factor-1) < 1e-8
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

    def vandermonde_evaluate_expansion(self, points, input_array, output_array):
        """Evaluate expansion at certain points, possibly different from
        the quadrature points

        Parameters
        ----------
            points : array
                     Points for evaluation
            input_array : array
                          Expansion coefficients
            output_array : array
                           Function values on points

        """
        assert abs(self.padding_factor-1) < 1e-8
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)

        if output_array.ndim == 1:
            output_array = np.dot(P, input_array, out=output_array)
        else:
            fc = np.moveaxis(input_array, self.axis, -2)
            array = np.dot(P, fc)
            output_array[:] = np.moveaxis(array, 0, self.axis)

        return output_array

    def vandermonde_evaluate_local_expansion(self, P, input_array, output_array):
        """Evaluate expansion at certain points, possibly different from
        the quadrature points

        Parameters
        ----------
            P : 2D array
                Vandermode matrix containing local points only
            input_array : array
                          Expansion coefficients
            output_array : array
                           Function values on points

        """
        fc = np.moveaxis(input_array, self.axis, -2)
        array = np.dot(P, fc)
        output_array[:] = np.moveaxis(array, 0, self.axis)
        return output_array

    def apply_inverse_mass(self, array):
        """Apply inverse mass matrix

        Parameters
        ----------
            array : array (input/output)
                    Expansion coefficients. Overwritten by applying the inverse
                    mass matrix, and returned.

        """
        assert self.N == array.shape[self.axis]
        if self._mass is None:
            B = self.get_mass_matrix()
            self._mass = B((self, 0), (self, 0))

        array = self._mass.solve(array, axis=self.axis)
        return array

    def plan(self, shape, axis, dtype, options):
        """Plan transform

        Allocate work arrays for transforms and set up methods 'forward',
        'backward' and 'scalar_product' with or without padding

        Parameters
        ----------
            shape : array 
                    Local shape of global array
            axis : int
                   This base's axis in global TensorProductSpace
            dtype : numpy.dtype
                    Type of array
            options : dict
                      Options for planning transforms
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

           :math:`f(x_j) = \sum_k f_k T_k(x_j)`  for all j = 0, 1, ..., N

        Parameters
        ----------
            input_array : array
                          Expansion coefficients
            output_array : array
                           Function values on quadrature mesh

        """
        raise NotImplementedError

    def eval(self, x, fk, output_array=None):
        """Evaluate basis at position x, given expansion coefficients fk

        Parameters
        ----------
            x : float or array of floats
            fk : array
                 Expansion coefficients
            output_array : array
                           Function values at points

        """
        if output_array is None:
            output_array = np.zeros(x.shape)
        return self.vandermonde_evaluate_expansion(x, fk, output_array)

    def get_mass_matrix(self):
        """Return mass matrix associated with current basis"""
        raise NotImplementedError

    def slice(self):
        """Return index set of current basis, with N points in physical space"""
        return slice(0, self.N)

    def spectral_shape(self):
        """Return the shape of current basis used to build a SpectralMatrix"""
        s = self.slice()
        return s.stop - s.start

    def domain_factor(self):
        """Return scaling factor for domain"""
        return 1

    def __hash__(self):
        return hash(repr(self.__class__))

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ and
                self.quad == other.quad and self.N == other.N)

    def sl(self, a):
        """Return a list of slices, broadcasted to the shape of a forward
        output array, with 'a' along self.axis
        
        Parameters
        ----------
            a : int or slice object
                This int or slice is used along self.axis of this basis
        """
        s = [slice(None)]*self.forward.output_array.ndim
        s[self.axis] = a
        return s

    def rank(self):
        """Return rank of basis"""
        return 1

    def ndim(self):
        """Return ndim of basis"""
        return 1

    def num_components(self):
        """Return number of components for basis"""
        return 1

    def __len__(self):
        return 1

    def __getitem__(self, i):
        assert i == 0
        return self

    def _get_mat(self):
        raise NotImplementedError

    def is_forward_output(self, u):
        """Return whether or not the array u is of type and shape resulting
        from a forward transform.
        """
        return (np.all(u.shape == self.forward.output_array.shape) and
                u.dtype == self.forward.output_array.dtype)

    def as_function(self, u):
        """Return Numpy array u as a Function."""
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
            su[self.axis] = slice(0, N//2+1)
            padded_array[su] = trunc_array[su]
            su[self.axis] = slice(-(N//2), None)
            padded_array[su] = trunc_array[su]


def inner_product(test, trial, out=None, axis=0, fast_transform=False):
    """Return inner product of linear or bilinear form

    Parameters
    ----------
        test : 2-tuple of (Basis, integer)     
               Basis is any of the classes from

                   shenfun.chebyshev.bases

                   shenfun.legendre.bases

                   shenfun.fourier.bases

               The integer determines the numer of times the basis is 
               differentiated. The test represents the matrix row
        trial : 2-tuple of (Basis, integer)
                Either an basis of argument 1 (trial) or 2 (Function)
                
                    If argument = 1, then a bilinear form is assembled to
                    a matrix. Trial represents matrix column

                    If argument = 2, then a linear form is assembled and the
                    'trial' represents a Function evaluated at quadrature nodes
        out : array, optional 
              Return array
        axis : int
               Axis to take the inner product over
        fast_transform : bool, optional
                         Use fast transform method if True

    Example
    -------
    Compute mass matrix of Shen's Chebyshev Dirichlet basis:

    >>> from shenfun.chebyshev.bases import ShenDirichletBasis
    >>> SD = ShenDirichletBasis(6)
    >>> B = inner_product((SD, 0), (SD, 0))
    >>> B
    {-2: array([-1.57079633]),
      0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265]),
      2: array([-1.57079633])}

    """
    from .fourier import FourierBase, R2CBasis

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
            if isinstance(test, FourierBase):
                if isinstance(test, R2CBasis):
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
        return self.output_array
