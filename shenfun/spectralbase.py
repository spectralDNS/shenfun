r"""
This module contains classes for working with the spectral-Galerkin method

There are currently classes for 9 bases and corresponding function spaces

All bases have expansions

   :math:`u(x) = \sum_{k\in\mathcal{I}}\hat{u}_{k} \phi_k(x)`

where :math:`\mathcal{I}` the index set of the basis, and the index set differs from
base to base, see function space definitions below. :math:`\phi_k` is the k't
basis function in the basis. It is also called a test function, whereas :math:`u(x)`
often is called a trial function.

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

        where :math:`u(1)=a, u(-1)=b`, such that :math:`\hat{u}_{N-1}=a,
        \hat{u}_{N}=b`.

        Note that there are only N-1 unknown coefficients, :math:`\hat{u}_k`,
        since :math:`\hat{u}_{N-1}` and :math:`\hat{u}_{N}` are determined by
        boundary conditions. Inhomogeneous boundary conditions are possible for
        the Poisson equation, because :math:`\phi_{N}` and :math:`\phi_{N-1}`
        are in the kernel of the Poisson operator. For homogeneous boundary
        conditions :math:`\phi_{N-1}` and :math:`\phi_{N}` are simply ignored.

    ShenNeumannBasis:
        basis function:
            :math:`\phi_k = T_k-\left(\frac{k}{k+2}\right)^2T_{k+2}`

        basis:
            :math:`span(\phi_k, k=1,2,...,N-2)`

        Homogeneous Neumann boundary conditions, :math:`u'(\pm 1) = 0`, and
        zero weighted mean: :math:`\int_{-1}^{1}u(x)w(x)dx = 0`.

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = T_k - \frac{2(k+2)}{k+3}T_{k+2} + \frac{k+1}{k+3}T_{k+4}`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-4)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and
        :math:`u'(\pm 1)=0`.

Legendre:
    LegendreBasis:
        basis function:
            :math:`\phi_k = L_k`

        basis:
            :math:`span(L_k, k=0,1,...N)`

    ShenDirichletBasis:
        basis function:
            :math:`\phi_k = L_k-L_{k+2}`

            :math:`\phi_{N-1} = 0.5(L_0+L_1)`, for Poisson's equation

            :math:`\phi_{N} = 0.5(L_0-L_1)`, for Poisson's equation

        basis:
            :math:`span(\phi_k, k=0,1,...,N)`

        where :math:`u(1)=a, u(-1)=b`, such that
        :math:`\hat{u}_{N-1}=a, \hat{u}_{N}=b`

        Note that there are only N-1 unknown coefficients, :math:`\hat{u}_k`,
        since :math:`\hat{u}_{N-1}` and :math:`\hat{u}_{N}` are determined by
        boundary conditions. Inhomogeneous boundary conditions are possible for
        the Poisson equation, because :math:`\phi_{N}` and :math:`\phi_{N-1}`
        are in the kernel of the Poisson operator. For homogeneous boundary
        conditions :math:`\phi_{N-1}` and :math:`\phi_{N}` are simply ignored.

    ShenNeumannBasis:
        basis function:
            :math:`\phi_k = L_k-\frac{k(k+1)}{(k+2)(k+3)}L_{k+2}`

        basis:
            :math:`span(\phi_k, k=1,2,...,N-2)`

        Homogeneous Neumann boundary conditions, :math:`u'(\pm 1) = 0`, and
        zero mean: :math:`\int_{-1}^{1}u(x)dx = 0`.

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = L_k - \frac{2(2k+5)}{2k+7}L_{k+2} + \frac{2k+3}{2k+7}L_{k+4}`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-4)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and
        :math:`u'(\pm 1)=0`.

Fourier:
    R2CBasis and C2CBasis:
        basis function:
            :math:`\phi_k = c_k exp(ikx)`

        basis:
            :math:`span(\phi_k, k=-N/2, -N/2+1, ..., N/2)`

        If N is even, then :math:`c_{-N/2}` and :math:`c_{N/2} = 0.5` and
        :math:`c_k = 1` for :math:`k=-N/2+1, ..., N/2-1`. :math:`i` is the
        imaginary unit.

        If N is odd, then :math:`c_k = 1` for :math:`k=-N/2, ..., N/2`.

    R2CBasis and C2CBasis are the same, but R2CBasis is used on real physical
    data and it takes advantage of Hermitian symmetry,
    :math:`\hat{u}_{-k} = conj(\hat{u}_k)`, for :math:`k = 1, ..., N/2`


Each class has methods for moving fast between spectral and physical space, and
for computing the (weighted) scalar product.

"""
#pylint: disable=unused-argument, not-callable, no-self-use, protected-access, too-many-public-methods, missing-docstring

import importlib
import numpy as np
import pyfftw
from mpi4py_fft import fftw
from .utilities import CachedArrayDict
work = CachedArrayDict()

class SpectralBase(object):
    """Abstract base class for all spectral function spaces

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto or Legendre-Gauss-Lobatto
            - GC - Chebyshev-Gauss
            - LG - Legendre-Gauss

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
        self.axis = 0
        self.xfftn_fwd = None
        self.xfftn_bck = None
        self.padding_factor = np.floor(N*padding_factor)/N
        self._mass = None         # Mass matrix (if needed)
        self._xfftn_fwd = None    # external forward transform function
        self._xfftn_bck = None    # external backward transform function
        self._M = 1.0             # Normalization factor
        self._ndim_tensor = 1     # ndim of belonging TensorProductSpace

    def points_and_weights(self, N=None, map_true_domain=False):
        """Return points and weights of quadrature

        Parameters
        ----------
            N : int, optional
                Number of quadrature points
            map_true_domain : bool, optional
                Whether or not to map points to true domain
        """
        raise NotImplementedError

    def mesh(self, bcast=True, map_true_domain=True):
        """Return the computational mesh

        Parameters
        ----------
            bcast : bool
                Whether or not to broadcast to :meth:`.ndim_tensorspace` dims
            map_true_domain : bool, optional
                Whether or not to map points to true domain
        """
        X = self.points_and_weights(map_true_domain=map_true_domain)[0]
        if bcast is True:
            X = self.broadcast_to_ndims(X)
        return X

    def wavenumbers(self, bcast=True, **kw):
        """Return the wavenumbermesh

        Parameters
        ----------
            bcast : bool
                Whether or not to broadcast to :attr:`.ndim_tensorspace` dims

        """
        s = self.slice()
        k = np.arange(s.start, s.stop)
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def broadcast_to_ndims(self, x):
        """Return 1D array ``x`` as an array of shape according to the
        :class:`.TensorProductSpace` the base (self) belongs to.

        Parameters
        ----------
            x : 1D array

        Note
        ----
        The returned array has shape one in all ndims-1 dimensions apart
        from self.axis.

        Example
        -------
        >>> import numpy as np
        >>> from shenfun import Basis, TensorProductSpace
        >>> from mpi4py import MPI
        >>> K0 = Basis(8, 'F', dtype='D')
        >>> K1 = Basis(8, 'F', dtype='d')
        >>> T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1))
        >>> x = np.arange(4)
        >>> y = K0.broadcast_to_ndims(x)
        >>> print(y.shape)
        (4, 1)
        """
        ndims = self.ndim_tensorspace
        s = [np.newaxis]*ndims
        s[self.axis] = slice(None)
        return x[tuple(s)]

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
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.evaluate_scalar_product(self.scalar_product.input_array,
                                     self.scalar_product.output_array,
                                     fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        return self.scalar_product.output_array

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
        self.scalar_product(input_array, fast_transform=fast_transform)
        self.apply_inverse_mass(self.forward.tmp_array)
        self._truncation_forward(self.forward.tmp_array,
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
                If True use fast transforms (if implemented), if
                False use Vandermonde type

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

        """
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self._padding_backward(self.backward.input_array,
                               self.backward.tmp_array)

        self.evaluate_expansion_all(self.backward.tmp_array,
                                    self.backward.output_array,
                                    fast_transform=fast_transform)

        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def vandermonde(self, x):
        r"""Return Vandermonde matrix based on the primary basis of the family.

        Evaluates basis :math:`\psi_k(x)` for all wavenumbers, and all ``x``.
        Returned Vandermonde matrix is an N x M matrix with N the length of
        ``x`` and M the number of bases.

        .. math::

            \begin{bmatrix}
                \psi_0(x_0) & \psi_1(x_0) & \ldots & \psi_{M-1}(x_0)\\
                \psi_0(x_1) & \psi_1(x_1) & \ldots & \psi_{M-1}(x_1)\\
                \vdots & \ldots \\
                \psi_{0}(x_{N-1}) & \psi_1(x_{N-1}) & \ldots & \psi_{M-1}(x_{N-1})
            \end{bmatrix}

        Parameters
        ----------
            x : array of floats
                points for evaluation

        Note
        ----
        This function returns a matrix of evaluated primary basis functions for
        either family. That is, it is using either pure Chebyshev, Legendre or
        Fourier exponentials. The true Vandermonde matrix of a basis is obtained
        through :meth:`.SpectralBase.evaluate_basis_all`.

        """
        raise NotImplementedError

    def evaluate_basis(self, x, i=0, output_array=None):
        """Evaluate basis ``i`` at points x

        Parameters
        ----------
            x : float or array of floats
            i : int, optional
                Basis number
            output_array : array, optional
                Return result in output_array if provided

        Returns
        -------
            array
                output_array

        """
        raise NotImplementedError

    def evaluate_basis_all(self, x=None):
        """Evaluate basis at ``x`` or all quadrature points

        Parameters
        ----------
            x : float or array of floats, optional
                If not provided use quadrature points of self

        Returns
        -------
            array
                Vandermonde matrix
        """
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        """Evaluate k'th derivative of basis ``i`` at ``x`` or all quadrature points

        Parameters
        ----------
            x : float or array of floats, optional
                If not provided use quadrature points of self
            i : int, optional
                Basis number
            k : int, optional
                k'th derivative
            output_array : array, optional
                return array

        Returns
        -------
            array
                output_array

        """
        raise NotImplementedError

    def evaluate_basis_derivative_all(self, x=None, k=0):
        """Return k'th derivative of basis evaluated at ``x`` or all quadrature
        points as a Vandermonde matrix.

        Parameters
        ----------
            x : float or array of floats, optional
                If not provided use quadrature points of self
            k : int, optional
                k'th derivative

        Returns
        -------
            array
                Vandermonde matrix
        """
        raise NotImplementedError

    def evaluate_scalar_product(self, input_array, output_array,
                                fast_transform=False):
        """Evaluate scalar product

        Parameters
        ----------
            input_array : array, optional
                Function values on quadrature mesh
            output_array : array, optional
                Expansion coefficients
            fast_transform : bool, optional
                If True use fast transforms (if implemented), if
                False use Vandermonde type
        """
        assert fast_transform is False
        self.vandermonde_scalar_product(input_array, output_array)

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

        _, weights = self.points_and_weights()
        P = self.evaluate_basis_all()

        if input_array.ndim == 1:
            output_array[:] = np.dot(input_array*weights, np.conj(P))

        else: # broadcasting
            bc_shape = [np.newaxis,]*input_array.ndim
            bc_shape[self.axis] = slice(None)
            fc = np.moveaxis(input_array*weights[tuple(bc_shape)], self.axis, -1)
            output_array[:] = np.moveaxis(np.dot(fc, np.conj(P)), -1, self.axis)
            #output_array[:] = np.moveaxis(np.tensordot(input_array*weights[bc_shape], np.conj(P), (self.axis, 0)), -1, self.axis)

        assert output_array is self.forward.output_array

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
        P = self.evaluate_basis_all()

        if output_array.ndim == 1:
            output_array = np.dot(P, input_array, out=output_array)
        else:
            fc = np.moveaxis(input_array, self.axis, -2)
            array = np.dot(P, fc)
            output_array[:] = np.moveaxis(array, 0, self.axis)

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
        P = self.evaluate_basis_all(x=points)

        if output_array.ndim == 1:
            output_array = np.dot(P, input_array, out=output_array)
        else:
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

        Allocate work arrays for transforms and set up methods `forward`,
        `backward` and `scalar_product` with or without padding

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

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        if 'builders' in self._xfftn_fwd.__module__:

            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)

            n = shape[axis]
            U = pyfftw.empty_aligned(shape, dtype=dtype)
            xfftn_fwd = plan_fwd(U, n=n, axis=axis, **opts)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, n=n, axis=axis, **opts)
            V.fill(0)
            U.fill(0)

            xfftn_fwd.update_arrays(U, V)
            xfftn_bck.update_arrays(V, U)
            self._M = 1./np.prod(np.take(shape, axis))

        else:
            opts = dict(
                overwrite_input='FFTW_DESTROY_INPUT',
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)
            flags = (fftw.flag_dict[opts['planner_effort']],
                     fftw.flag_dict[opts['overwrite_input']])
            threads = opts['threads']

            n = (shape[axis],)
            U = pyfftw.empty_aligned(shape, dtype=dtype)
            xfftn_fwd = plan_fwd(U, n, (axis,), threads=threads, flags=flags)
            V = xfftn_fwd.output_array

            if np.issubdtype(dtype, np.floating):
                flags = (fftw.flag_dict[opts['planner_effort']],)

            xfftn_bck = plan_bck(V, n, (axis,), threads=threads, flags=flags, output_array=U)
            V.fill(0)
            U.fill(0)
            self._M = xfftn_fwd.get_normalization()

        self.axis = axis
        self._ndim_tensor = U.ndim

        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)

        # scalar_product is not padded, just the forward/backward
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)

    def _get_truncarray(self, shape, dtype):
        shape = list(shape)
        shape[self.axis] = int(np.round(shape[self.axis] / self.padding_factor))
        return pyfftw.empty_aligned(shape, dtype=dtype)

    def get_normalization(self):
        return self._M

    def evaluate_expansion_all(self, input_array, output_array,
                               fast_transform=False):
        r"""Evaluate expansion on entire mesh

        .. math::

            u(x_j) = \sum_{k\in\mathcal{I}} \hat{u}_k T_k(x_j) \quad \text{ for all} \quad j = 0, 1, ..., N

        Parameters
        ----------
            input_array : :math:`\hat{u}_k`
                Expansion coefficients, or instance of :class:`.Function`
            output_array : :math:`u(x_j)`
                Function values on quadrature mesh, instance of :class:`.Array`
            fast_transform : bool, optional
                Whether to use fast transforms (if implemented)

        """
        self.vandermonde_evaluate_expansion_all(input_array, output_array)

    def eval(self, x, u, output_array=None):
        """Evaluate :class:`.Function` ``u`` at position ``x``

        Parameters
        ----------
            x : float or array of floats
            u : array
                Expansion coefficients or instance of :class:`.Function`
            output_array : array, optional
                Function values at points

        Returns
        -------
            array
                output_array

        """
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.input_array.dtype)
        x = self.map_reference_domain(x)
        return self.vandermonde_evaluate_expansion(x, u, output_array)

    def map_reference_domain(self, x):
        """Return true point `x` mapped to reference domain"""
        if not self.domain == self.reference_domain():
            a = self.domain[0]
            c = self.reference_domain()[0]
            x = c + (x-a)*self.domain_factor()
        return x

    def map_true_domain(self, x):
        """Return reference point `x` mapped to true domain"""
        if not self.domain == self.reference_domain():
            a = self.domain[0]
            c = self.reference_domain()[0]
            x = a + (x-c)/self.domain_factor()
        return x

    def reference_domain(self):
        """Return reference domain of basis"""
        raise NotImplementedError

    def slice(self):
        """Return index set of current basis"""
        return slice(0, self.N)

    def shape(self, forward_output=True):
        """Return the shape of current basis

        Parameters
        ----------
            forward_output : bool, optional
                If True then return shape of spectral space (the outcome of a
                forward transform). If False then return shape of physical space
                (the input to a forward transform).
        """
        if forward_output:
            s = self.slice()
            return s.stop - s.start
        return self.N

    def domain_factor(self):
        """Return scaling factor for domain"""
        a, b = self.domain
        c, d = self.reference_domain()
        L = b-a
        R = d-c
        if abs(L-R) < 1e-12:
            return 1
        return R/L

    @property
    def ndim_tensorspace(self):
        """Return ndim of :class:`.TensorProductSpace` if planned, else 1"""
        return self._ndim_tensor

    def __hash__(self):
        return hash(repr(self.__class__))

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ and
                self.quad == other.quad and self.N == other.N)

    def sl(self, a):
        """Return a list of slices, broadcasted to the shape of a forward
        output array, with `a` along self.axis

        Parameters
        ----------
            a : int or slice object
                This int or slice is used along self.axis of this basis
        """
        s = [slice(None)]*self.ndim_tensorspace
        s[self.axis] = a
        return tuple(s)

    def rank(self):
        """Return rank of basis"""
        return 1

    def ndim(self):
        """Return ndim of basis"""
        return 1

    def num_components(self):
        """Return number of components for basis"""
        return 1

    @staticmethod
    def boundary_condition():
        return ''

    @staticmethod
    def family():
        return ''

    def get_mass_matrix(self):
        mat = self._get_mat()
        return mat[((self.__class__, 0), (self.__class__, 0))]

    def _get_mat(self):
        mod = importlib.import_module('shenfun.'+self.family())
        return mod.matrices.mat

    def __len__(self):
        return 1

    def __getitem__(self, i):
        assert i == 0
        return self

    def _truncation_forward(self, padded_array, trunc_array):
        pass

    def _padding_backward(self, trunc_array, padded_array):
        pass

def inner_product(test, trial, out=None, axis=0, fast_transform=False):
    """Return inner product of linear or bilinear form

    Parameters
    ----------
        test : 2-tuple of (Basis, integer)
               Basis is any of the classes from

               - :mod:`.chebyshev.bases`
               - :mod:`.legendre.bases`
               - :mod:`.fourier.bases`

               The integer determines the number of times the basis is
               differentiated. The test represents the matrix row
        trial : 2-tuple of (Basis, integer)
                Either an basis of argument 1 (trial) or 2 (Function)

                - If argument = 1, then a bilinear form is assembled to
                  a matrix. Trial represents matrix column
                - If argument = 2, then a linear form is assembled and the
                  trial represents a Function evaluated at quadrature nodes

        out : array, optional
              Return array
        axis : int, optional
               Axis to take the inner product over
        fast_transform : bool, optional
                         Use fast transform method if True

    Example
    -------
    Compute mass matrix of Shen's Chebyshev Dirichlet basis:

    >>> from shenfun.spectralbase import inner_product
    >>> from shenfun.chebyshev.bases import ShenDirichletBasis
    >>> import six
    >>> SD = ShenDirichletBasis(6)
    >>> B = inner_product((SD, 0), (SD, 0))
    >>> d = {-2: np.array([-np.pi/2]),
    ...       0: np.array([1.5*np.pi, np.pi, np.pi, np.pi]),
    ...       2: np.array([-np.pi/2])}
    >>> [np.all(B[k] == v) for k, v in six.iteritems(d)]
    [True, True, True]
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

class FuncWrap(object):

    # pylint: disable=too-few-public-methods, missing-docstring

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
        return self.func(input_array, output_array, **kw)

class Transform(FuncWrap):

    # pylint: disable=too-few-public-methods

    __slots__ = ('_xfftn', '__doc__', '_input_array', '_output_array',
                 '_tmp_array')

    def __init__(self, func, xfftn, input_array, tmp_array, output_array):
        FuncWrap.__init__(self, func, input_array, output_array)
        object.__setattr__(self, '_xfftn', xfftn)
        object.__setattr__(self, '_tmp_array', tmp_array)

    @property
    def tmp_array(self):
        return object.__getattribute__(self, '_tmp_array')

    @property
    def xfftn(self):
        return object.__getattribute__(self, '_xfftn')
