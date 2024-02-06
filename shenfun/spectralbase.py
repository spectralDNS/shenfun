r"""
This module contains classes for working with global spectral Galerkin methods
in one dimension. All spectral functions have the following expansions on the
real line

   :math:`u(x) = \sum_{k\in\mathcal{I}}\hat{u}_{k} \phi_k(x)`

where :math:`\mathcal{I}` is a index set of the basis, which differs from
space to space. :math:`\phi_k` is the k't basis function and the function
space is the span of these basis functions.

See the [documentation](https://shenfun.readthedocs.io) for more details.

"""
#pylint: disable=unused-argument, not-callable, no-self-use, protected-access, too-many-public-methods, missing-docstring

import importlib
import re
import copy
import importlib
from numbers import Number
import sympy as sp
import numpy as np
from mpi4py_fft import fftw
from shenfun import config
from shenfun.utilities import get_stencil_matrix, n
from .utilities import CachedArrayDict, split
from .coordinates import Coordinates
work = CachedArrayDict()
xp = sp.Symbol('x', real=True)

class SpectralBase:
    """Abstract base class for all spectral function spaces
    """
    # pylint: disable=method-hidden, too-many-instance-attributes
    def __init__(self, N, quad='', padding_factor=1, domain=(-1., 1.), dtype=None,
                 dealias_direct=False, coordinates=None):
        self.N = N
        self.domain = domain
        self.quad = quad
        self.axis = 0
        self.bc = None
        self._bc_space = None
        self._stencil = None
        self._stencil_matrix = {}
        self.padding_factor = padding_factor
        if padding_factor != 1:
            self.padding_factor = np.floor(N*padding_factor)/N if N > 0 else 1
        self.dealias_direct = dealias_direct
        self._mass = None         # Mass matrix (if needed)
        self._dtype = dtype
        self._M = 1.0             # Normalization factor
        self._xfftn_fwd = None    # external forward transform function
        self._xfftn_bck = None    # external backward transform function
        coors = coordinates if coordinates is not None else ((sp.Symbol('x', real=True),),)*2
        self.coors = Coordinates(*coors)
        self.hi = self.coors.hi
        self.sg = self.coors.sg
        self.si = islicedict()
        self.sl = slicedict()
        self._tensorproductspace = None     # link if belonging to TensorProductSpace

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        r"""Return points and weights of quadrature for weighted integral

        .. math::

            \int_{\Omega} f(x) w(x) dx \approx \sum_{i} f(x_i) w_i

        Parameters
        ----------
        N : int, optional
            Number of quadrature points
        map_true_domain : bool, optional
            Whether or not to map points to true domain
        weighted : bool, optional
            Whether to use quadrature weights for a weighted inner product
            (default), or a regular, non-weighted inner product.

        Note
        ----
        The weight of the space is included in the returned quadrature weights.

        """
        raise NotImplementedError

    def mesh(self, bcast=True, map_true_domain=True, kind='quadrature'):
        """Return the computational mesh

        Parameters
        ----------
        bcast : bool
            Whether or not to broadcast to :meth:`.dimensions` if an instance
            of this basis belongs to a :class:`.TensorProductSpace`.
            The returned mesh then has shape one in all ndims-1 dimensions
            apart from self.axis.
        map_true_domain : bool, optional
            Whether or not to map points to true domain
        kind : str, optional

            - 'quadrature' - Use quadrature mesh
            - 'uniform' - Use uniform mesh

        Note
        ----
        For curvilinear problems this function returns the computational mesh.
        For the corresponding Cartesian mesh, use :meth:`.cartesian_mesh`
        """
        N = self.shape(False)
        if self.family() == 'fourier':
            kind = 'quadrature'
        if kind == 'quadrature':
            X = self.points_and_weights(N=N, map_true_domain=map_true_domain)[0]

        else:
            d = self.domain
            X = np.linspace(float(d[0]), float(d[1]), N)
            if map_true_domain is False:
                X = self.map_reference_domain(X)

        if bcast is True:
            X = self.broadcast_to_ndims(X)
        return X

    def cartesian_mesh(self, kind='quadrature'):
        """Return Cartesian mesh

        Parameters
        ----------
        kind : str, optional

            - 'quadrature' - Use quadrature mesh
            - 'uniform' - Use uniform mesh

        Note
        ----
        For Cartesian problems this function returns the same mesh as :meth:`.mesh`
        """
        x = self.mesh(kind=kind)
        if self.coors.is_cartesian:
            return x
        psi = self.coors.psi
        xx = []
        for rvi in self.coors.rv:
            xx.append(sp.lambdify(psi, rvi)(x))
        return xx

    def wavenumbers(self, bcast=True, **kw):
        """Return the wavenumbermesh

        Parameters
        ----------
        bcast : bool
            Whether or not to broadcast to :meth:`.dimensions` if an instance
            of this basis belongs to a :class:`.TensorProductSpace`

        """
        s = self.slice()
        k = np.arange(s.start, s.stop)
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def broadcast_to_ndims(self, x):
        """Return 1D array ``x`` as an array of shape according to the
        :meth:`.dimensions` of the :class:`.TensorProductSpace` class
        that the instance of this class belongs to.

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
        >>> from shenfun import FunctionSpace, TensorProductSpace, comm
        >>> K0 = FunctionSpace(8, 'F', dtype='D')
        >>> K1 = FunctionSpace(8, 'F', dtype='d')
        >>> T = TensorProductSpace(comm, (K0, K1), modify_spaces_inplace=True)
        >>> x = np.arange(4)
        >>> y = K0.broadcast_to_ndims(x)
        >>> print(y.shape)
        (4, 1)
        """
        s = [np.newaxis]*self.dimensions
        s[self.axis] = slice(None)
        return x[tuple(s)]

    def scalar_product(self, input_array=None, output_array=None, kind=None):
        """Compute weighted scalar product

        Parameters
        ----------
        input_array : array, optional
            Function values on quadrature mesh
        output_array : array, optional
            The result of the scalar product
        kind : str, optional
            - 'fast' - use fast transform if implemented
            - 'vandermonde' - use Vandermonde matrix
            - 'recursive' - Use low-memory implementation (only for polynomials)

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

        """
        kind = kind if kind is not None else config['transforms']['kind'][self.family()]
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.scalar_product._input_array = self.get_measured_array(self.scalar_product._input_array)

        self._evaluate_scalar_product(kind=kind)

        self._truncation_forward(self.scalar_product.tmp_array,
                                 self.scalar_product.output_array)
        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        return self.scalar_product.output_array

    def forward(self, input_array=None, output_array=None, kind=None):
        """Compute forward transform

        Parameters
        ----------
        input_array : array, optional
            Function values on quadrature mesh
        output_array : array, optional
            Expansion coefficients
        kind : str, optional
            - 'fast' - use fast transform if implemented
            - 'vandermonde' - Use Vandermonde matrix
            - 'recursive' - Use low-memory implementation (only for polynomials)

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

        """
        kind = kind if kind is not None else config['transforms']['kind'][self.family()]
        self.scalar_product(input_array, kind=kind)
        if self.bc:
            self.bc._add_mass_rhs(self.forward.output_array)
        self.apply_inverse_mass(self.forward.output_array)
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def backward(self, input_array=None, output_array=None, kind=None, mesh=None):
        """Compute backward (inverse) transform

        Parameters
        ----------
        input_array : array, optional
            Expansion coefficients
        output_array : array, optional
            Function values on quadrature mesh
        kind : str, optional
            - 'fast' - Use fast transform on regular quadrature points
            - 'recursive' - Use low-memory implementation (only for polynomials)
            - 'vandermonde' - use Vandermonde on regular quadrature points
        mesh : str or functionspace, optional
            - 'quadrature' - use quadrature mesh of self
            - 'uniform' - use uniform mesh
            - A function space with the same mesh distribution as self

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

        """
        kind = kind if kind is not None else config['transforms']['kind'][self.family()]
        if input_array is not None:
            self.backward.input_array[...] = input_array

        self._padding_backward(self.backward.input_array,
                               self.backward.tmp_array)

        if isinstance(mesh, str):
            assert mesh in ('quadrature', 'uniform')
            if not self.family() == 'fourier' and mesh == 'uniform':
                kind = 'vandermonde' if kind == 'fast' else kind
            mesh = self.mesh(bcast=False, map_true_domain=False, kind=mesh)

        elif isinstance(mesh, SpectralBase):
            if not self.family() == 'fourier':
                kind = 'vandermonde' if kind == 'fast' else kind
            mesh = mesh.mesh(bcast=False, map_true_domain=False)

        self._evaluate_expansion_all(self.backward.tmp_array,
                                     self.backward.output_array,
                                     x=mesh, kind=kind)
        if output_array is not None:
            output_array[...] = self.backward.output_array
            return output_array
        return self.backward.output_array

    def vandermonde(self, x):
        r"""Return Vandermonde matrix based on the primary (orthogonal) basis
        of the family.

        Evaluates basis function :math:`\psi_k(x)` for all wavenumbers, and all
        ``x``. Returned Vandermonde matrix is an N x M matrix with N the length
        of ``x`` and M the number of bases.

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
        This function returns a matrix of evaluated orthogonal basis functions
        for either family. The true Vandermonde matrix of a basis is obtained
        through :meth:`.SpectralBase.evaluate_basis_all`.

        """
        raise NotImplementedError

    def evaluate_basis(self, x, i=0, output_array=None):
        """Evaluate basis function ``i`` at points x

        Parameters
        ----------
        x : float or array of floats
        i : int, optional
            Basis function number
        output_array : array, optional
            Return result in output_array if provided

        Returns
        -------
        array
            output_array

        """
        raise NotImplementedError
        #x = np.atleast_1d(x)
        #if output_array is None:
        #    output_array = np.zeros(x.shape, dtype=self.dtype)
        #X = sp.symbols('x', real=True)
        #f = self.basis_function(i, X)
        #output_array[:] = sp.lambdify(X, f)(x)
        #return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        """Evaluate basis at ``x`` or all quadrature points

        Parameters
        ----------
        x : float or array of floats, optional
            If not provided use quadrature points of self
        argument : int
            Zero for test and 1 for trialfunction

        Returns
        -------
        array
            Vandermonde matrix
        """
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        """Evaluate k'th derivative of basis function ``i`` at ``x`` or all
        quadrature points

        Parameters
        ----------
        x : float or array of floats, optional
            If not provided use quadrature points of self
        i : int, optional
            Basis function number
        k : int, optional
            k'th derivative
        output_array : array, optional
            return array

        Returns
        -------
        array
            output_array

        """
        #warnings.warn('Using slow sympy evaluate_basis_derivative')
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        X = sp.symbols('x', real=True)
        basis = self.basis_function(i=i, x=X).diff(X, k)
        output_array[:] = sp.lambdify(X, basis, 'numpy')(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        """Return k'th derivative of basis evaluated at ``x`` or all quadrature
        points as a Vandermonde matrix.

        Parameters
        ----------
        x : float or array of floats, optional
            If not provided use quadrature points of self
        k : int, optional
            k'th derivative
        argument : int
            Zero for test and 1 for trialfunction

        Returns
        -------
        array
            Vandermonde matrix
        """
        if x is None:
            x = self.mesh(False, False)
        V = np.zeros((x.shape[0], self.N))
        for i in range(self.dim()):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def _evaluate_expansion_all(self, input_array, output_array,
                                x=None, kind=None):
        r"""Evaluate expansion on 'x' or entire mesh

        .. math::

            u(x_j) = \sum_{k\in\mathcal{I}} \hat{u}_k T_k(x_j) \quad \text{ for all} \quad j = 0, 1, ..., N

        Parameters
        ----------
        input_array : :math:`\hat{u}_k`
            Expansion coefficients, or instance of :class:`.Function`
        output_array : :math:`u(x_j)`
            Function values on quadrature mesh, instance of :class:`.Array`
        x : points for evaluation, optional
            If None, use entire quadrature mesh
        kind : str, optional

            - 'fast' - use fast transform if implemented
            - 'vandermonde' - Use Vandermonde matrix
            - 'recursive' - Use low-memory recursive implementation
              (only for polynomials)

        """
        assert kind in ('vandermonde', 'recursive')
        if kind == 'vandermonde':
            P = self.evaluate_basis_all(x=x, argument=1)
            if output_array.ndim == 1:
                output_array = np.dot(P, input_array, out=output_array)
            else:
                fc = np.moveaxis(input_array, self.axis, -2)
                shape = [slice(None)]*input_array.ndim
                N = self.shape(False)
                shape[-2] = slice(0, N)
                array = np.dot(P, fc[tuple(shape)])
                output_array[:] = np.moveaxis(array, 0, self.axis)
        elif kind == 'recursive':
            mod = config['optimization']['mode']
            lib = importlib.import_module('.'.join(('shenfun.optimization', mod, 'transforms')))
            if x is None:
                x = self.mesh(False, False)
            N = self.N
            a = self.get_recursion_matrix(int(N*self.padding_factor)+3, int(N*self.padding_factor)+3).diags('dia').data
            lib.evaluate_expansion_all(input_array, output_array, x, self.axis, a)

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
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        self._evaluate_expansion_all(u, output_array, x, kind='vandermonde')
        return output_array

    def _evaluate_scalar_product(self, kind=None):
        """Evaluate scalar product

        Parameters
        ----------
        kind : str, optional
            - 'fast' - use fast transform if implemented
            - 'vandermonde' - Use Vandermonde matrix
            - 'recursive' - Use low-memory implementation (only for polynomials)

        Note
        ----
        Using internal arrays: ``self.scalar_product.input_array`` and
        ``self.scalar_product.output_array``

        """
        assert kind in ('vandermonde', 'recursive') # fast must be implemented in subclass
        input_array = self.scalar_product.input_array
        output_array = self.scalar_product.tmp_array
        M = self.shape(False)
        xj, weights = self.points_and_weights(M)
        if self.domain_factor() != 1:
            weights /= float(self.domain_factor())
        if kind == 'vandermonde':
            # Slow, memory demanding, Vandermonde type implementation
            P = self.evaluate_basis_all(argument=0)
            if input_array.ndim == 1:
                output_array[slice(0, M)] = np.dot(input_array*weights, np.conj(P))
            else: # broadcasting
                bc_shape = [np.newaxis,]*input_array.ndim
                bc_shape[self.axis] = slice(None)
                fc = np.moveaxis(input_array*weights[tuple(bc_shape)], self.axis, -1)
                output_array[self.sl[slice(0, M)]] = np.moveaxis(np.dot(fc, np.conj(P)), -1, self.axis)
                #output_array[:] = np.moveaxis(np.tensordot(input_array*weights[bc_shape], np.conj(P), (self.axis, 0)), -1, self.axis)
        elif kind == 'recursive':
            # Vandermonde type, but using less memory
            mod = config['optimization']['mode']
            lib = importlib.import_module('.'.join(('shenfun.optimization', mod, 'transforms')))
            if xj is None:
                xj = self.mesh(False, False)
            N = len(xj)
            a = self.get_recursion_matrix(N+3, N+3).diags('dia').data
            lib.scalar_product(input_array, output_array, xj, weights, self.axis, a)

    def apply_inverse_mass(self, array):
        """Apply inverse mass matrix

        Parameters
        ----------
        array : array (input/output)
            Expansion coefficients. Overwritten by applying the inverse
            mass matrix, and returned.

        Returns
        -------
        array
        """
        if self._mass is None:
            B = self.get_mass_matrix()
            self._mass = B((self, 0), (self, 0))
        array = self._mass.solve(array, axis=self.axis)
        return array

    def to_ortho(self, input_array, output_array=None):
        """Project to orthogonal basis

        Parameters
        ----------
        input_array : array
            Expansion coefficients of input basis
        output_array : array, optional
            Expansion coefficients in orthogonal basis

        Returns
        -------
        array
            output_array

        """
        raise NotImplementedError
        #from shenfun import project
        #T = self.get_orthogonal()
        #output_array = project(input_array, T, output_array=output_array,
        #                       use_to_ortho=False)
        #return output_array

    def plan(self, shape, axis, dtype, options):
        """Plan transform

        Allocate work arrays for transforms and set up methods `forward`,
        `backward` and `scalar_product` with or without padding

        Parameters
        ----------
        shape : array
            Local shape of global array
        axis : int
            This base's axis in global :class:`.TensorProductSpace`
        dtype : numpy.dtype
            Type of array
        options : dict
            Options for planning transforms
        """
        # Note - Only needs overloading for fast transforms (Fourier, Chebyshev)
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
            self.scalar_product = Transform(self.scalar_product, None, U, V, trunc_array)
            self.forward = Transform(self.forward, None, U, V, trunc_array)
            self.backward = Transform(self.backward, None, trunc_array, V, U)
        else:
            self.scalar_product = Transform(self.scalar_product, None, U, V, V)
            self.forward = Transform(self.forward, None, U, V, V)
            self.backward = Transform(self.backward, None, V, V, U)

        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def _get_truncarray(self, shape, dtype):
        shape = list(shape) if np.ndim(shape) else [shape]
        shape[self.axis] = int(np.round(shape[self.axis] / self.padding_factor))
        return fftw.aligned(shape, dtype=dtype)

    def get_normalization(self):
        return self._M

    def map_reference_domain(self, x):
        """Return true point `x` mapped to reference domain

        Parameters
        ----------
        x : coordinate or array of points
        """
        if not self.domain == self.reference_domain():
            a = self.domain[0]
            c = self.reference_domain()[0]
            if isinstance(x, (np.ndarray, float)):
                x = float(c) + (x-float(a))*float(self.domain_factor())
            else:
                x = c + (x-a)*self.domain_factor()

        return x

    def map_true_domain(self, x):
        """Return reference point `x` mapped to true domain

        Parameters
        ----------
        x : coordinate or array of points
        """
        if not self.domain == self.reference_domain():
            a = self.domain[0]
            c = self.reference_domain()[0]
            if isinstance(x, (np.ndarray, float)):
                x = float(a) + (x-float(c))/float(self.domain_factor())
            else:
                x = a + (x-c)/self.domain_factor()
        return x

    def map_expression_true_domain(self, f, x=None):
        """Return expression `f` mapped to true domain as a function
        of the reference coordinate

        Parameters
        ----------
        f : Sympy expression
        x : Sympy symbol, optional
            coordinate
        """
        if not self.domain == self.reference_domain():
            f = sp.sympify(f)
            if len(f.free_symbols) == 1:
                if x is None:
                    x = f.free_symbols.pop()
                xm = self.map_true_domain(x)
                f = f.replace(x, xm)
        return f

    def basis_function(self, i=0, x=sp.Symbol('x', real=True)):
        """Return basis function `i`

        Parameters
        ----------
        i : int, optional
            The degree of freedom of the basis function
        x : sympy Symbol, optional
        """
        return self.orthogonal_basis_function(i=i, x=x)

    def orthogonal_basis_function(self, i=0, x=sp.Symbol('x', real=True)):
        """Return the orthogonal basis function `i`

        Parameters
        ----------
        i : int, optional
            The degree of freedom of the basis function
        x : sympy Symbol, optional
        """
        raise NotImplementedError

    def sympy_basis(self, i=0, x=sp.Symbol('x', real=True)):
        raise DeprecationWarning('Use basis_function instead')

    def L2_norm_sq(self, i):
        r"""Return square of L2-norm

        .. math::

            \| \phi_i \|^2_{\omega} = (\phi_i, \phi_i)_{\omega} = \int_{I} \phi_i \overline{\phi}_i \omega dx

        where :math:`\phi_i` is the i'th orthogonal basis function for the
        orthogonal basis in the given family.

        Parameters
        ----------
        i : int
            The number of the orthogonal basis function

        """
        raise NotImplementedError


    def l2_norm_sq(self, i=None):
        r"""Return square of l2-norm

        .. math::

            \| u \|^2_{N,\omega} = (u, u)_{N,\omega} = \sun_{j=0}^{N-1} u(x_j) \overline{u}(x_j) \omega_j

        where :math:`u=\{\phi_i\}_{i=0}^{N-1}` and :math:`\phi_i` is the i'th
        orthogonal basis function in the given family.

        Parameters
        ----------
        i : None or int
            If None then return the square of the l2-norm for all
            i=0, 1, ..., N-1. Else, return for given i.

        """
        raise NotImplementedError

    def sympy_l2_norm_sq(self, i=sp.Symbol('i', integer=True)):
        r"""Return sympy function for square of l2-norm

        .. math::

            \| \phi_i \|^2_{N,\omega} = (\phi_i, \phi_i)_{N,\omega} = \sun_{j=0}^{N-1} \phi_i(x_j) \overline{\phi}_i(x_j) \omega_j

        where :math:`\phi_i` is the i'th orthogonal basis function in the given
        family.

        Parameters
        ----------
        i : Sympy Symbol, optional
        implicit : bool, optional
            Whether to use an unevaluated Sympy function instead of the
            actual value of the l2-norm.

        Returns
        -------
        Sympy Function

        """
        class h(sp.Function):
            @classmethod
            def eval(cls, x):
                if x.is_Number:
                    return self.l2_norm_sq(x)
        return h(i)

    def sympy_L2_norm_sq(self, i=sp.Symbol('i', integer=True)):
        r"""Return sympy function for square of L2-norm

        .. math::

            \| \phi_i \|^2_{N,\omega} = (\phi_i, \phi_i)_{\omega} = \int_{I} \phi_i \overline{\phi}_i \omega dx

        where :math:`\phi_i` is the i'th orthogonal basis function in the given
        family.

        Parameters
        ----------
        i : Sympy Symbol, optional

        Returns
        -------
        Sympy Function

        """
        class h(sp.Function):
            @classmethod
            def eval(cls, x):
                if x.is_Number:
                    return self.L2_norm_sq(x)
        return h(i)

    def weight(self, x=None):
        """Weight of inner product space

        Parameters
        ----------
        x : coordinate
        """
        return 1

    def reference_domain(self):
        raise NotImplementedError

    def domain_length(self):
        return self.domain[1]-self.domain[0]

    def slice(self):
        """Return index set of current space"""
        return slice(0, self.N)

    def dim(self):
        """Return the dimension of ``self`` (the number of degrees of freedom)"""
        s = self.slice()
        return s.stop - s.start

    def dims(self):
        """Return tuple (length one since a basis only has one dim) containing
        self.dim()"""
        return (self.dim(),)

    def shape(self, forward_output=True):
        """Return the allocated shape of arrays used for ``self``

        Parameters
        ----------
        forward_output : bool, optional
            If True then return allocated shape of spectral space (the result of a
            forward transform). If False then return allocated shape of physical space
            (the input to a forward transform).
        """
        if forward_output:
            return self.N
        if self.padding_factor != 1:
            return int(np.floor(self.padding_factor*self.N))
        return self.N

    def domain_factor(self):
        """Return scaling factor for domain

        Note
        ----
        The domain factor is the length of the reference domain over the
        length of the true domain.
        """
        a, b = self.domain
        c, d = self.reference_domain()
        L = b-a
        R = d-c
        if abs(L-R) < 1e-12:
            return 1
        return R/L

    @property
    def dtype(self):
        """Return datatype function space is planned for"""
        if hasattr(self.forward, 'input_array'):
            return self.forward.input_array.dtype
        return self._dtype

    @property
    def dimensions(self):
        """Return the dimensions (the number of bases) of the
        :class:`.TensorProductSpace` class this space is planned for.
        """
        if self.tensorproductspace:
            return self.tensorproductspace.dimensions
        return self.forward.input_array.ndim

    @property
    def tensorproductspace(self):
        """Return the last :class:`.TensorProductSpace` this space has been
        planned for (if planned)

        Note
        ----
        A 1D function space may be part of several :class:`.TensorProductSpace`s,
        but they all need to be of the same global shape.
        """
        return self._tensorproductspace

    @tensorproductspace.setter
    def tensorproductspace(self, T):
        self._tensorproductspace = T

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ and
                self.quad == other.quad and
                self.N == other.N and
                self.axis == other.axis)

    @property
    def is_orthogonal(self):
        return False

    @property
    def is_padded(self):
        if self.padding_factor != 1:
            return True
        return False

    @property
    def is_composite_space(self):
        return 0

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    @property
    def is_boundary_basis(self):
        return False

    @property
    def is_jacobi(self):
        return False

    @property
    def rank(self):
        """Return rank of function space

        Note
        ----
        This is 1 for :class:`.MixedFunctionSpace` and 0 otherwise
        """
        return 0

    @property
    def tensor_rank(self):
        """Return tensor rank of function space

        Note
        ----
        This is the number of free indices in the tensor, 0 for scalar,
        1 for vector etc. It is None for a composite space that is not a
        tensor.
        """
        return 0

    @property
    def ndim(self):
        return 1

    def num_components(self):
        return 1

    @staticmethod
    def boundary_condition():
        return ''

    @staticmethod
    def family():
        return ''

    @staticmethod
    def short_name():
        return ''

    def get_mass_matrix(self):
        mat = self._get_mat()
        dx = self.coors.get_sqrt_det_g()
        key = ((self.__class__, 0), (self.__class__, 0))
        if self.tensorproductspace:
            dx = self.tensorproductspace.coors.get_sqrt_det_g()
            msdict = split(dx)
            #assert len(msdict) == 1
            dx = msdict[0]['xyzrs'[self.axis]]
            if self.axis == self.tensorproductspace.dimensions-1:
                dx *= msdict[0]['coeff']
        if not dx == 1:
            if len(sp.sympify(dx).free_symbols) > 0:
                x0 = dx.free_symbols
                if len(x0) > 1:
                    raise NotImplementedError("Cannot use forward for Curvilinear coordinates with unseparable measure - Use inner with mass matrix for tensor product space")
                x0 = x0.pop()
                x = sp.symbols('x', real=x0.is_real, positive=x0.is_positive)
                dx = dx.subs(x0, x)
                if self.domain != self.reference_domain():
                    xm = self.map_true_domain(x)
                    dx = dx.replace(x, xm)
            key = key + (dx,)

        return mat[key]

    def _get_mat(self):
        mod = importlib.import_module('shenfun.'+self.family())
        return mod.matrices.mat

    def compatible_base(self, space):
        return hash(self) == hash(space)

    def __len__(self):
        return 1

    def __getitem__(self, i):
        assert i == 0
        return self

    def __hash__(self):
        return hash((self.N, self.quad, self.family()))

    def get_bcmass_matrix(self, dx=1):
        msx = 'xyzrs'[self.axis]
        dV = split(dx)
        #assert len(dV) == 1
        dv = dV[0]
        msi = dv[msx]
        if self.axis == self.dimensions-1:
            msi *= dv['coeff']
        return inner_product((self, 0), (self.get_bc_space(), 0), msi)

    def get_measured_weights(self, N=None, measure=1, map_true_domain=False):
        """Return weights times ``measure``

        Parameters
        ----------
        N : integer, optional
            The number of quadrature points
        measure : 1 or ``sympy.Expr``
        """
        if N is None:
            N = self.shape(False)
        xm, wj = self.points_and_weights(N, map_true_domain=map_true_domain)
        if measure == 1:
            return wj

        s = sp.sympify(measure).free_symbols
        if isinstance(measure, Number) or len(s) == 0:
            wj *= float(measure)
            return wj

        assert len(s) == 1
        s = s.pop()
        xj = sp.lambdify(s, measure)(xm)
        wj = wj*xj
        return wj

    def get_measured_array(self, array):
        """Return `array` times Jacobian determinant

        Parameters
        ----------
        array : array

        Note
        ----
        If basis is part of a :class:`.TensorProductSpace`, then the
        array will be measured there. So in that case, just return
        the array unchanged.
        """
        if self.tensorproductspace:
            return array

        measure = self.coors.get_sqrt_det_g()
        if measure == 1:
            return array

        if isinstance(measure, Number):
            array *= measure
            return array

        N = self.shape(False)
        xm = self.points_and_weights(N, map_true_domain=True)[0]
        s = measure.free_symbols
        if len(s) == 0:
            # constant
            array[...] = array*float(measure)
        elif len(s) == 1:
            s = s.pop()
            xj = sp.lambdify(s, measure)(xm)
            array[...] = array*xj
        else:
            raise NotImplementedError
        return array

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False, **kwargs):
        """Return space (otherwise as self) to be used for dealiasing

        Parameters
        ----------
        padding_factor : float, optional
            Create output array of shape padding_factor times non-padded shape
        dealias_direct : bool, optional
            If True, set upper 2/3 of wavenumbers to zero before backward transform.
            Cannot be used together with padding_factor different than 1.
        kwargs : keyword arguments
            Any other keyword arguments used in the creation of the bases.

        Returns
        -------
        :class:`.SpectralBase`
            The space to be used for dealiasing
        """
        if padding_factor == 1 and dealias_direct == False:
            return self

        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=padding_factor,
                 dealias_direct=dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, 'bcs'):
            d['bc'] = copy.deepcopy(self.bcs)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled
        for key in ('alpha', 'beta'):
            if hasattr(self, key):
                d[key] = object.__getattribute__(self, key)
        d.update(kwargs)
        return self.__class__(self.N, **d)

    def get_testspace(self, kind='Galerkin', **kwargs):
        r"""Return appropriate test space

        Parameters
        ----------
        kind : str, optional

            - 'Galerkin' - or 'G'
            - 'Petrov-Galerkin' - or 'PG'

            If kind = 'Galerkin' or 'G', then return self, corresponding to a
            regular Galerkin method.

            If kind = 'Petrov-Galerkin' or 'PG', then return a test space that
            makes use of the testbases `Phi1`, `Phi2`, `Phi3` or `Phi4`, where
            the actual choice is made based on the number of boundary conditions
            in self. One boundary condition uses `Phi1`, two uses `Phi2` etc.

            For Petrov-Galerkin the returned basis corresponds to
            :math:`\{\phi^{(k)}_n\}`, where

            .. math::

                \phi_n^{(k)} = \frac{(1-x^2)^k}{h^{(k)}_{n+k}}\frac{d^k}{dx^k}Q_{n+k}

        kwargs : keyword arguments, optional
            Any other keyword arguments used in the creation of the test bases.
            Only used if kind='Petrov-Galerkin'.

        Note
        ----
        If self is not a Jacobi polynomial basis, then return self,
        corresponding to a regular Galerkin method.

        Returns
        -------
        Instance of :class:`.SpectralBase`
        """
        if kind == 'Galerkin' or kind == 'G':
            return self
        assert kind == 'Petrov-Galerkin' or kind == 'PG'
        d = dict(domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        dim = self.N-self.dim()
        for key in ('alpha', 'beta', 'quad'):
            if hasattr(self, key):
                d[key] = object.__getattribute__(self, key)
        d.update(kwargs)
        mod = importlib.import_module(self.__module__)
        if dim == 0 or not hasattr(mod, 'Phi1') :
            return self.get_unplanned(**d)
        assert dim > 0
        P = getattr(mod, 'Phi'+str(dim))
        return P(self.N+dim, **d)

    def get_refined(self, N, **kwargs):
        """Return space (otherwise as self) with N quadrature points

        Parameters
        ----------
        N : int
            The number of quadrature points for returned space
        kwargs : keyword arguments
            Any other keyword arguments used in the creation of the bases.

        Returns
        -------
        :class:`.SpectralBase`
            A new space with new number of quadrature points, otherwise as self.
        """
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, 'bcs'):
            d['bc'] = copy.deepcopy(self.bcs)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled
        for key in ('alpha', 'beta'):
            if hasattr(self, key):
                d[key] = object.__getattribute__(self, key)
        d.update(kwargs)
        return self.__class__(N, **d)

    def get_unplanned(self, **kwargs):
        """Return unplanned space (otherwise as self)

        Parameters
        ----------
        kwargs : keyword arguments, optional
            Any keyword argument used in the creation of the unplanned
            space. Could be any one of

            - quad
            - domain
            - dtype
            - padding_factor
            - dealias_direct
            - coordinates
            - bcs
            - scaled

            Not all will be applicable for all spaces.

        Returns
        -------
        :class:`.SpectralBase`
            Space not planned for a :class:`.TensorProductSpace`

        """
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, 'bcs'):
            d['bc'] = copy.deepcopy(self.bcs)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled
        for key in ('alpha', 'beta'):
            if hasattr(self, key):
                d[key] = object.__getattribute__(self, key)
        d.update(kwargs)
        return self.__class__(self.N, **d)

    def get_homogeneous(self, **kwargs):
        """Return space (otherwise as self) with homogeneous boundary conditions

        Parameters
        ----------
        kwargs : keyword arguments
            Any keyword arguments used in the creation of the bases.

        Returns
        -------
        :class:`.SpectralBase`
            A new space with homogeneous boundary conditions, otherwise as self.
        """
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        if hasattr(self, '_scaled'):
            d['scaled'] = self._scaled

        for key in ('alpha', 'beta'):
            if hasattr(self, key):
                d[key] = object.__getattribute__(self, key)
        d.update(kwargs)
        return self.__class__(self.N, **d)

    def get_adaptive(self, fun=None, reltol=1e-12, abstol=1e-15):
        """Return space (otherwise as self) with number of quadrature points
        determined by fitting `fun`

        Returns
        -------
        :class:`.SpectralBase`
            A new space with adaptively found number of quadrature points
        """
        from shenfun import Function
        assert isinstance(fun, sp.Expr)
        assert self.N == 0
        T = self.get_refined(5)
        converged = False
        count = 0
        points = np.random.random(8)
        points = float(T.domain[0]) + points*float(T.domain[1]-T.domain[0])
        sym = fun.free_symbols
        assert len(sym) == 1
        x = sym.pop()
        fx = sp.lambdify(x, fun)
        while (not converged) and count < 12:
            T = T.get_refined(int(1.7*T.N))
            u = Function(T, buffer=fun)
            res = T.eval(points, u)
            exact = fx(points)
            energy = np.linalg.norm(res-exact)
            converged = energy**2 < abstol
            count += 1

        # trim trailing zeros (if any)
        trailing_zeros = T.count_trailing_zeros(u, reltol, abstol)
        T = T.get_refined(T.N - trailing_zeros)
        return T

    def get_orthogonal(self, **kwargs):
        """Return orthogonal space (otherwise as self)

        Returns
        -------
        :class:`.SpectralBase`
            The orthogonal space in the same family, and otherwise as self.
        """
        raise NotImplementedError

    def get_recursion_matrix(self, M, N):
        from shenfun.jacobi.recursions import a, pmat
        return pmat(a, 1, self.alpha, self.beta, M, N, self.gn)

    def count_trailing_zeros(self, u, reltol=1e-12, abstol=1e-15):
        assert u.function_space() == self
        assert u.ndim == 1
        a = abs(u[self.slice()])
        ua = (a < reltol*a.max()) | (a < abstol)
        return np.argmin(ua[::-1])

    def _truncation_forward(self, padded_array, trunc_array):
        if not id(trunc_array) == id(padded_array):
            trunc_array.fill(0)
            s = self.sl[self.slice()]
            trunc_array[s] = padded_array[s]

    def _padding_backward(self, trunc_array, padded_array):
        if not id(trunc_array) == id(padded_array):
            padded_array.fill(0)
            s = self.sl[self.slice()]
            padded_array[s] = trunc_array[s]

        elif self.dealias_direct:
            s = self.sl[slice(2*self.N//3, None)]
            padded_array[s] = 0

        else:
            return

        # Fix boundary condition dofs
        if self.bc:
            B = self.get_bc_space()
            sl = B.slice()
            nd = sl.stop - sl.start
            if padded_array.ndim > 1:
                sl = self.sl[sl]
            sp = self.sl[slice(-nd, None)]
            padded_array[sp] = trunc_array[sl]


def getCompositeBase(Orthogonal):
    """Dynamic class factory for Composite base class

    Parameters
    ----------
    Orthogonal : :class:`.SpectralBase`
        Dynamic inheritance, using base class as either one of

        - :class:`.chebyshev.bases.Orthogonal`
        - :class:`.chebyshevu.bases.Orthogonal`
        - :class:`.legendre.bases.Orthogonal`
        - :class:`.ultraspherical.bases.Orthogonal`
        - :class:`.jacobi.bases.Orthogonal`
        - :class:`.laguerre.bases.Orthogonal`
    Returns
    -------
    Either one of the classes

        - :class:`.chebyshev.bases.CompositeBase`
        - :class:`.chebyshevu.bases.CompositeBase`
        - :class:`.legendre.bases.CompositeBase`
        - :class:`.ultraspherical.bases.CompositeBase`
        - :class:`.jacobi.bases.CompositeBase`
        - :class:`.laguerre.bases.CompositeBase`

    """
    class CompositeBase(Orthogonal):
        """Common class for all composite spaces"""

        def __init__(self, *args, **kwargs):
            bc = kwargs.pop('bc', None)
            scaled = kwargs.pop('scaled', None)
            Orthogonal.__init__(self, *args, **kwargs)
            if bc is not None:
                from shenfun.tensorproductspace import BoundaryValues
                if isinstance(bc, dict) and not isinstance(bc, BoundaryConditions):
                    bc = BoundaryConditions(bc, domain=kwargs['domain'])
                if isinstance(bc, (tuple, list)):
                    bc = BoundaryConditions(bc, domain=kwargs['domain'])
                assert isinstance(bc, BoundaryConditions)
                self.bcs = bc
                self.bc = BoundaryValues(self, bc=bc)

            if scaled is not None:
                self._scaled = scaled

        def slice(self):
            return slice(0, self.N-self.bcs.num_bcs())

        def evaluate_basis_all(self, x=None, argument=0):
            V = Orthogonal.evaluate_basis_all(self, x=x, argument=argument)
            return self._composite(V, argument=argument)

        def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
            V = Orthogonal.evaluate_basis_derivative_all(self, x=x, k=k, argument=argument)
            return self._composite(V, argument=argument)

        def _evaluate_expansion_all(self, input_array, output_array, x=None, kind=None):
            if kind == 'fast':
                assert self.family() in ('fourier', 'chebyshev', 'chebyshevu', 'legendre'),\
                    f'Fast method not implemented for {self.family()} family'
            if kind == 'vandermonde':
                SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, kind=kind)
                return
            assert input_array is self.backward.tmp_array
            assert output_array is self.backward.output_array
            input_array[:] = self.to_ortho(input_array, output_array)
            Orthogonal._evaluate_expansion_all(self, input_array, output_array, x, kind)

        def _evaluate_scalar_product(self, kind=None):
            output = self.scalar_product.tmp_array
            if kind == 'fast':
                assert self.family() in ('fourier', 'chebyshev', 'chebyshevu', 'legendre'),\
                    f'Fast method not implemented for {self.family()} family'
            if kind == 'vandermonde':
                SpectralBase._evaluate_scalar_product(self, kind=kind)
                output[self.sl[slice(-(self.N-self.dim()), None)]] = 0
                return
            Orthogonal._evaluate_scalar_product(self, kind)
            K = self.stencil_matrix(self.shape(False))
            w0 = output.copy()
            output = K.matvec(w0, output, axis=self.axis)

        @property
        def is_orthogonal(self):
            return False

        def get_orthogonal(self, **kwargs):
            d = dict(quad=self.quad,
                     domain=self.domain,
                     dtype=self.dtype,
                     padding_factor=self.padding_factor,
                     dealias_direct=self.dealias_direct,
                     coordinates=self.coors.coordinates)
            for kw in ('alpha', 'beta'):
                if hasattr(self, kw):
                    d[kw] = getattr(self, kw)
            d.update(kwargs)
            return Orthogonal(self.N, **d)

        @property
        def has_nonhomogeneous_bcs(self):
            if self.bc is None:
                return False
            return self.bc.has_nonhomogeneous_bcs()

        def is_scaled(self):
            if not hasattr(self, '_scaled'):
                return False
            return self._scaled

        def stencil_matrix(self, N=None):
            """Return stencil matrix in :class:`.SparseMatrix` format

            Parameters
            ----------
            N : int or None, optional
                Shape (N, N) of the stencil matrix. Using self.N if None
            """
            from .matrixbase import SparseMatrix
            if self._stencil is None:
                self._stencil = get_stencil_matrix(self.bcs, self.family(), self.alpha, self.beta, self.gn)
            N = self.N if N is None else N
            if N in self._stencil_matrix:
                return self._stencil_matrix[N]
            d = {}
            k = np.arange(N)
            for i, s in self._stencil.items():
                di = sp.lambdify(n, s)(k[:N-i])
                if isinstance(di, np.ndarray):
                    di[(N-self.bcs.num_bcs()):] = 0
                d[i] = di
            self._stencil_matrix[N] = SparseMatrix(d, (N, N))
            return self._stencil_matrix[N]

        def sympy_stencil(self, i=sp.Symbol('i', integer=True), j=sp.Symbol('j', integer=True), implicit=False):
            """Return stencil matrix as a Sympy matrix

            Parameters
            ----------
            i, j : Sympy symbols
                indices for row and column
            implicit : bool or str, optional
                Whether to use an unevaluated Sympy function instead of the
                actual value of the stencil. This makes the matrix prettier,
                and it can still be evaluated. If implicit is not False, then
                it must be a string of length one. This string represents the
                diagonals of the matrix. If implicit='a' and the stencil matrix
                has two diagonals in the main diagonal and the first upper
                diagonal, then these are called 'a0' and 'a1'.

            Example
            -------
            >>> from shenfun import FunctionSpace
            >>> import sympy as sp
            >>> i, j = sp.symbols('i,j', integer=True)
            >>> D = FunctionSpace(8, 'L', bc=(0, 0), scaled=True)
            >>> D._stencil
            {0: 1/sqrt(4*n + 6), 2: -1/sqrt(4*n + 6)}
            >>> D.sympy_stencil()
            KroneckerDelta(i, j)/sqrt(4*i + 6) - KroneckerDelta(j, i + 2)/sqrt(4*i + 6)
            >>> D.sympy_stencil(implicit='a')
            KroneckerDelta(i, j)*a0(i) + KroneckerDelta(j, i + 2)*a2(i)

            Get the main diagonal

            >>> D.sympy_stencil(implicit=False).subs(j, i)
            1/sqrt(4*i + 6)

            """
            if self._stencil is None:
                self._stencil = get_stencil_matrix(self.bcs, self.family(), self.alpha, self.beta, self.gn)

            def _named_diagonal(s, name, i):
                class h(sp.Function):
                    @classmethod
                    def eval(cls, x):
                        if x.is_Number:
                            return s.subs(i, x)
                hh = h(i)
                hh.__class__.__name__ = name
                return hh

            S = sp.S(0)
            for m, s in self._stencil.items():
                if len(sp.sympify(s).free_symbols) == 1:
                    n = sp.sympify(s).free_symbols.pop()
                    d0 = s.subs(n, i)
                    if implicit is not False:
                        name = implicit if isinstance(implicit, str) else 'a'
                        d0 = _named_diagonal(d0, name+str(m), i)
                else:
                    d0 = s
                S += d0*sp.KroneckerDelta(i+m, j)
            return S

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
                P[:, slice(-(self.N-self.dim()), None)] = self.get_bc_space()._composite(V)
            return P

        def basis_function(self, i=0, x=sp.Symbol('x', real=True)):
            assert i < self.N
            if i < self.dim():
                row = self.stencil_matrix().diags().getrow(i)
                f = 0
                for j, val in zip(row.indices, row.data):
                    f += sp.nsimplify(val)*self.orthogonal_basis_function(i=j, x=x)
            else:
                f = self.get_bc_space().basis_function(i=i-self.dim(), x=x)
            return f

        def to_ortho(self, input_array, output_array=None):
            if output_array is None:
                output_array = np.zeros_like(input_array)
            else:
                output_array.fill(0)
            s = [np.newaxis]*self.dimensions
            for key, val in self.stencil_matrix().items():
                M = self.N if key >= 0 else self.dim()
                s0 = slice(max(0, -key), min(self.dim(), M-max(0, key)))
                Q = s0.stop-s0.start
                s1 = slice(max(0, key), max(0, key)+Q)
                s[self.axis] = slice(0, Q)
                if isinstance(val, Number):
                    output_array[self.sl[s1]] += val*input_array[self.sl[s0]]
                else:
                    output_array[self.sl[s1]] += val[tuple(s)]*input_array[self.sl[s0]]
            if self.has_nonhomogeneous_bcs:
                self.bc._add_to_orthogonal(output_array, input_array)
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
                output_array = self.get_bc_space().evaluate_basis(x, i=i-self.dim(), output_array=output_array)
            return output_array

        def evaluate_basis_derivative(self, x, i=0, k=0, output_array=None):
            x = np.atleast_1d(x)
            if output_array is None:
                output_array = np.zeros(x.shape)
            if i < self.dim():
                row = self.stencil_matrix().diags().getrow(i)
                w0 = np.zeros_like(output_array)
                output_array.fill(0)
                for j, val in zip(row.indices, row.data):
                    output_array[:] += val*Orthogonal.evaluate_basis_derivative(self, x, i=j, k=k, output_array=w0)
            else:
                assert i < self.N
                output_array = self.get_bc_space().evaluate_basis_derivative(x, i=i-self.dim(), k=k, output_array=output_array)
            return output_array

        def eval(self, x, u, output_array=None):
            x = np.atleast_1d(x)
            if output_array is None:
                output_array = np.zeros(x.shape, dtype=self.dtype)
            w = self.to_ortho(u)
            output_array = Orthogonal.eval(self, x, w, output_array)
            return output_array

    return CompositeBase

def getBCGeneric(CompositeBase):
    """Dynamic class factory for boundary spaces

    Parameters
    ----------
    CompositeBase : :class:`.SpectralBase`
        Dynamic inheritance, using base class as either one of

        - :class:`.chebyshev.bases.CompositeBase`
        - :class:`.chebyshevu.bases.CompositeBase`
        - :class:`.legendre.bases.CompositeBase`
        - :class:`.ultraspherical.bases.CompositeBase`
        - :class:`.jacobi.bases.CompositeBase`
        - :class:`.laguerre.bases.CompositeBase`
    Returns
    -------
    Either one of the classes

        - :class:`.chebyshev.bases.BCGeneric`
        - :class:`.chebyshevu.bases.BCGeneric`
        - :class:`.legendre.bases.BCGeneric`
        - :class:`.ultraspherical.bases.BCGeneric`
        - :class:`.jacobi.bases.BCGeneric`
        - :class:`.laguerre.bases.BCGeneric`

    """
    class BCGeneric(CompositeBase):
        """Function space for setting inhomogeneous boundary conditions

        Parameters
        ----------
        N : int
            Number of quadrature points in the homogeneous space.
        bc : dict
            The boundary conditions in dictionary form, see
            :class:`.BoundaryConditions`.
        domain : 2-tuple of numbers, optional
            The domain of the homogeneous space.
        alpha : number, optional
            Parameter of the Jacobi polynomial
        beta : number, optional
            Parameter of the Jacobi polynomial

        """
        def __init__(self, N, bc=None, domain=None, alpha=0, beta=0, **kw):
            CompositeBase.__init__(self, N, bc=bc, domain=domain, alpha=alpha, beta=beta)
            self._stencil_matrix = None

        def stencil_matrix(self, N=None):
            if self._stencil_matrix is None:
                from shenfun.utilities import get_bc_basis
                d = {'alpha': self.alpha, 'beta': self.beta, 'gn': self.gn} if self.is_jacobi else {}
                self._stencil_matrix = np.array(get_bc_basis(self.bcs, self.family(), **d))
            return self._stencil_matrix

        @staticmethod
        def short_name():
            return 'BG'

        @staticmethod
        def boundary_condition():
            return 'Apply'

        @property
        def is_boundary_basis(self):
            return True

        def shape(self, forward_output=True):
            if forward_output:
                return self.stencil_matrix().shape[0]
            else:
                return self.N

        @property
        def dim_ortho(self):
            return self.stencil_matrix().shape[1]

        def slice(self):
            return slice(self.N-self.shape(), self.N)

        def vandermonde(self, x):
            V = np.zeros((x.shape[0], self.dim_ortho))
            for i in range(self.dim_ortho):
                func = self.orthogonal_basis_function(i=i)
                V[:, i] = sp.lambdify(xp, func)(x)
            return V

        def _composite(self, V, argument=1):
            N = self.shape()
            P = np.zeros(V[:, :N].shape)
            P[:] = np.tensordot(V[:, :self.dim_ortho], self.stencil_matrix(), (1, 1))
            return P

        def basis_function(self, i=0, x=xp):
            M = self.stencil_matrix()
            return np.sum(M[i]*np.array([self.orthogonal_basis_function(j, x) for j in range(self.dim_ortho)]))

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

        def get_orthogonal(self, **kwargs):
            d = dict(quad=self.quad,
                     domain=self.domain,
                     dtype=self.dtype,
                     padding_factor=self.padding_factor,
                     dealias_direct=self.dealias_direct,
                     coordinates=self.coors.coordinates)
            for kw in ('alpha', 'beta'):
                if hasattr(self, kw):
                    d[kw] = getattr(self, kw)
            d.update(kwargs)
            base = importlib.import_module('.'.join(('shenfun', self.family().lower())))
            return base.Orthogonal(self.dim_ortho, **d)

    return BCGeneric

class BoundaryConditions(dict):
    """Boundary conditions for :class:`.SpectralBase`.

    Parameters
    ----------
    bc : str, dict or n-tuple
        The dictionary must have keys 'left' and 'right', to describe boundary
        conditions on the left and right boundaries, and another dictionary on
        left or right to describe the condition. For example, specify Dirichlet
        on both ends with::

            {'left': {'D': a}, 'right': {'D': b}}

        for some values `a` and `b`. Specify Neumann as::

            {'left': {'N': a}, 'right': {'N': b}}

        A mixture of 3 conditions::

            {'left': {'N': a}, 'right': {'D': b, 'N': c}}

        Etc. Any combination should be possible, and it should also be possible
        to use higher order derivatives with `N2`, `N3` etc.

        If `bc` is an n-tuple, then we assume the basis function is::

            (None, a) - {'right': {'D': a}}
            (a, None) - {'left': {'D': a}}
            (a, b) - {'left': {'D': a}, 'right': {'D': b}}
            (a, b, c, d) - {'left': {'D': a, 'N': b}, 'right': {'D': c, 'N': d}}
            (a, b, c, d, e, f) - {'left': {'D': a, 'N': b, 'N2': c},
                                  'right': {'D': d, 'N': e, 'N2': f}}
            etc.

        If `bc` is a single string, then we assume the boundary conditions
        are described directly for generic function `u`, like::

            'u(-1)=0&&u(1)=0' - Dirichlet on boundaries x=-1 and x=1
            'u'(-1)=0&&u'(1)=0' - Neumann on boundaries x=-1 and x=1
            'u(-2)=a&&u(2)=b' - Dirichlet with values a and b on boundaries
                x=-2 and x=2.
            etc.

    domain : 2-tuple of numbers, optional
        The domain, if different than (-1, 1).

    """
    def __init__(self, bc, domain=None):
        bcs = {'left': {}, 'right': {}}

        if isinstance(bc, str):
            # Boundary conditions given in single string with boundary conditions separated by &&
            for bci in bc.split('&&'):
                # Check for Dirichlet
                x = re.search(r"u\((.*)\)=(.*)", bci)
                if x:
                    if np.abs(float(x.group(1))-domain[0]) < 1e-8:
                        # left boundary
                        bcs['left']['D'] = float(x.group(2))
                    elif np.abs(float(x.group(1))-domain[1]) < 1e-8:
                        # right boundary
                        bcs['right']['D'] = float(x.group(2))
                    else:
                        raise RuntimeError('Boundary condition not matching domain')
                    continue
                x = re.search(r"u'\((.*)\)=(.*)", bci)
                if x:
                    if np.abs(float(x.group(1))-domain[0]) < 1e-8:
                        # left boundary
                        bcs['left']['N'] = float(x.group(2))
                    elif np.abs(float(x.group(1))-domain[1]) < 1e-8:
                        # right boundary
                        bcs['right']['N'] = float(x.group(2))
                    else:
                        raise RuntimeError('Boundary condition not matching domain')
                    continue
                x = re.search(r"u''\((.*)\)=(.*)", bci)
                if x:
                    if np.abs(float(x.group(1))-domain[0]) < 1e-8:
                        # left boundary
                        bcs['left']['N2'] = float(x.group(2))
                    elif np.abs(float(x.group(1))-domain[1]) < 1e-8:
                        # right boundary
                        bcs['right']['N2'] = float(x.group(2))
                    else:
                        raise RuntimeError('Boundary condition not matching domain')
                    continue
                x = re.search(r"u'''\((.*)\)=(.*)", bci)
                if x:
                    if np.abs(float(x.group(1))-domain[0]) < 1e-8:
                        # left boundary
                        bcs['left']['N3'] = float(x.group(2))
                    elif np.abs(float(x.group(1))-domain[1]) < 1e-8:
                        # right boundary
                        bcs['right']['N3'] = float(x.group(2))
                    else:
                        raise RuntimeError('Boundary condition not matching domain')
                    continue
                x = re.search(r"u''''\((.*)\)=(.*)", bci)
                if x:
                    if np.abs(float(x.group(1))-domain[0]) < 1e-8:
                        # left boundary
                        bcs['left']['N4'] = float(x.group(2))
                    elif np.abs(float(x.group(1))-domain[1]) < 1e-8:
                        # right boundary
                        bcs['right']['N4'] = float(x.group(2))
                    else:
                        raise RuntimeError('Boundary condition not matching domain')
                    continue
                raise RuntimeError(f'Boundary condition {bci} not understood')

        if isinstance(bc, tuple):
            assert len(bc) in (1, 2, 4, 6, 8, 10, 12)
            assert np.all([isinstance(i, (sp.Expr, Number)) or i is None for i in bc])
            if len(bc) == 1: # Laguerre
                bcs['left']['D'] = bc[0]
            elif len(bc) == 2:
                if bc[0] is not None:
                    bcs['left']['D'] = bc[0]
                if bc[1] is not None:
                    bcs['right']['D'] = bc[1]
            elif len(bc) == 4:
                bcs['left'].update({'D': bc[0], 'N': bc[1]})
                bcs['right'].update({'D': bc[2], 'N': bc[3]})
            elif len(bc) == 6:
                bcs['left'].update({'D': bc[0], 'N': bc[1], 'N2': bc[2]})
                bcs['right'].update({'D': bc[3], 'N': bc[4], 'N2': bc[5]})
            elif len(bc) == 8:
                bcs['left'].update({'D': bc[0], 'N': bc[1], 'N2': bc[2], 'N3': bc[3]})
                bcs['right'].update({'D': bc[4], 'N': bc[5], 'N2': bc[6], 'N3': bc[7]})
            elif len(bc) == 10:
                bcs['left'].update({'D': bc[0], 'N': bc[1], 'N2': bc[2], 'N3': bc[3], 'N4': bc[4]})
                bcs['right'].update({'D': bc[5], 'N': bc[6], 'N2': bc[7], 'N3': bc[8], 'N4': bc[9]})
            elif len(bc) == 12:
                bcs['left'].update({'D': bc[0], 'N': bc[1], 'N2': bc[2], 'N3': bc[3], 'N4': bc[4], 'N5': bc[5]})
                bcs['right'].update({'D': bc[6], 'N': bc[7], 'N2': bc[8], 'N3': bc[9], 'N4': bc[10], 'N5': bc[11]})

        elif isinstance(bc, dict):
            if isinstance(bc.get("left", {}) or bc.get("right", {}), (tuple, list)):
                # old form {'left': [('D', 0), ('N', 0)], 'right': [('D', 0)]}
                bc = {k.lower(): list(v) if isinstance(v[0], (tuple, list)) else [v] for k, v in bc.items()}
                for key, val in bc.items():
                    bcs[key] = {v[0]: copy.deepcopy(v[1]) for v in val}
            else:
                #bcs.update(copy.deepcopy(bc))
                bcs.update(bc)

        # Take care of non-standard domain size
        df = 1
        if domain is not None:
            assert isinstance(domain, (tuple, list))
            if np.isfinite(float(domain[0]*domain[1])):
                df = 2/float(domain[1]-domain[0])

        for key, val in bcs.items():
            for bc, v in val.items():
                if bc == 'N':
                    val[bc] /= df
                elif bc == 'N2':
                    val[bc] /= df**2
                elif bc == 'N3':
                    val[bc] /= df**3
        dict.__init__(self, bcs)

    def orderednames(self):
        return ['L'+bci for bci in sorted(self['left'].keys())] + ['R'+bci for bci in sorted(self['right'].keys())]

    def orderedvals(self):
        ls = []
        for lr in ('left', 'right'):
            for key in sorted(self[lr].keys()):
                val = self[lr][key]
                ls.append(val[1] if isinstance(val, (tuple, list)) else val)
        return ls
        #return [self['left'][bci] for bci in sorted(self['left'].keys())] + [self['right'][bci] for bci in sorted(self['right'].keys())]

    def num_bcs(self):
        return len(self.orderedvals())

    def num_derivatives(self):
        n = {'D': 0, 'R': 0, 'N': 1, 'N2': 2, 'N3': 3, 'N4': 4}
        num_diff = 0
        for val in self.values():
            for k in val:
                num_diff += n[k]
        return num_diff

class MixedFunctionSpace:
    """Class for composite bases in 1D

    Parameters
    ----------
    bases : list
        List of instances of :class:`.SpectralBase`

    """
    def __init__(self, bases):
        self.bases = bases
        self.forward = VectorBasisTransform([basis.forward for basis in bases])
        self.backward = VectorBasisTransform([basis.backward for basis in bases])
        self.scalar_product = VectorBasisTransform([basis.scalar_product for basis in bases])

    def dims(self):
        """Return dimensions (degrees of freedom) for MixedFunctionSpace"""
        s = []
        for space in self.flatten():
            s.append(space.dim())
        return s

    def dim(self):
        """Return dimension of ``self`` (degrees of freedom)"""
        s = 0
        for space in self.flatten():
            s += space.dim()
        return s

    def shape(self, forward_output=False):
        """Return shape of arrays for MixedFunctionSpace

        Parameters
        ----------
        forward_output : bool, optional
            If True then return shape of an array that is the result of a
            forward transform. If False then return shape of physical
            space, i.e., the input to a forward transform.
        """
        if forward_output:
            s = []
            for space in self.flatten():
                s.append(space.shape(forward_output))
        else:
            s = self.flatten()[0].shape(forward_output)
            s = (self.num_components(),) + s
        return s

    def num_components(self):
        """Return number of bases in mixed basis"""
        f = self.flatten()
        return len(f)

    def flatten(self):
        s = []
        def _recursiveflatten(l, s):
            if hasattr(l, 'bases'):
                for i in l.bases:
                    _recursiveflatten(i, s)
            else:
                s.append(l)
        _recursiveflatten(self, s)
        return s

    def __getitem__(self, i):
        return self.bases[i]

    def __getattr__(self, name):
        obj = object.__getattribute__(self, 'bases')
        return getattr(obj[0], name)

    def __len__(self):
        return self.bases[0].dimensions

    @property
    def is_composite_space(self):
        return 1

    @property
    def rank(self):
        return 1

    @property
    def tensor_rank(self):
        return None

    @property
    def dimensions(self):
        return self.bases[0].dimensions

    def slice(self):
        return tuple(tuple([i]+[space.slice()]) for i, space in enumerate(self.flatten()))

    def get_diagonal_axes(self):
        return np.array([])

    def _get_ndiag_cum_dofs(self):
        return np.array([0]+np.cumsum(self.dims()).tolist())

    def _get_ndiag_slices(self, j=()):
        return np.array(self.slice())

    def _get_ndiag_slices_and_dims(self, j=()):
        sl = self._get_ndiag_slices(j)
        dims = self._get_ndiag_cum_dofs()
        return sl, dims

    def __len__(self):
        return len(self.bases)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        """Return space (otherwise as self) to be used for dealiasing

        Parameters
        ----------
        padding_factor : number or tuple, optional
            Scale the number of quadrature points in self with this number, or
            tuple of numbers, one for each dimension.
        dealias_direct : bool, optional
            Used only if ``padding_factor=1``. Sets the 1/3 highest frequencies
            to zeros.
        """
        if padding_factor == 1 and dealias_direct is False:
            return self
        if isinstance(padding_factor, Number):
            padding_factor = (padding_factor,)*len(self)
        elif isinstance(padding_factor, (tuple, list, np.ndarray)):
            assert len(padding_factor) == len(self)
        padded_bases = [base.get_dealiased(padding_factor=padding_factor[i],
                                           dealias_direct=dealias_direct)
                        for i, base in enumerate(self.bases)]
        return MixedFunctionSpace(padded_bases)

class VectorBasisTransform:

    __slots__ = ('_transforms',)

    def __init__(self, transforms):
        self._transforms = []
        for transform in transforms:
            if isinstance(transform, VectorBasisTransform):
                self._transforms += transform._transforms
            else:
                self._transforms.append(transform)

    def __getattr__(self, name):
        obj = object.__getattribute__(self, '_transforms')
        if name == '_transforms':
            return obj
        return getattr(obj[0], name)

    def __call__(self, input_array, output_array, **kw):
        for i, transform in enumerate(self._transforms):
            output_array[i] = transform(input_array[i], output_array[i], **kw)
        return output_array


class islicedict(dict):
    """Return a tuple of slices, broadcasted to ``dimensions`` number of
    dimensions, and with integer ``a`` along ``axis``.

    Parameters
    ----------
        axis : int
            The axis the calling basis belongs to in a :class:`.TensorProductSpace`
        dimensions : int
            The number of bases in the :class:`.TensorProductSpace`

    Example
    -------
    >>> from shenfun.spectralbase import islicedict
    >>> s = islicedict(axis=1, dimensions=3)
    >>> print(s[0])
    (slice(None, None, None), 0, slice(None, None, None))

    """
    def __init__(self, axis=0, dimensions=1):
        dict.__init__(self)
        self.axis = axis
        self.dimensions = dimensions

    def __missing__(self, key):
        assert isinstance(key, int)
        s = [slice(None)]*self.dimensions
        s[self.axis] = key
        self[key] = s = tuple(s)
        return s

class slicedict(dict):
    """Return a tuple of slices, broadcasted to ``dimensions`` number of
    dimensions, and with slice ``a`` along ``axis``.

    Parameters
    ----------
        axis : int
            The axis the calling basis belongs to in a :class:`.TensorProductSpace`
        dimensions : int
            The number of bases in the :class:`.TensorProductSpace`

    Example
    -------
    >>> from shenfun.spectralbase import slicedict
    >>> s = slicedict(axis=1, dimensions=3)
    >>> print(s[slice(0, 5)])
    (slice(None, None, None), slice(0, 5, None), slice(None, None, None))

    """
    def __init__(self, axis=0, dimensions=1):
        dict.__init__(self)
        self.axis = axis
        self.dimensions = dimensions

    def __missing__(self, key):
        s = [slice(None)]*self.dimensions
        s[self.axis] = slice(*key)
        self[key] = s = tuple(s)
        return s

    def __keytransform__(self, key):
        assert isinstance(key, slice)
        return key.__reduce__()[1]

    def __getitem__(self, key):
        return dict.__getitem__(self, self.__keytransform__(key))


def inner_product(test, trial, measure=1, assemble=None, kind=None, fixed_resolution=None):
    """Return 1D weighted inner product of bilinear form

    Parameters
    ----------
    test : 2-tuple of (Basis, integer)
        Basis is any of the classes from

        - :mod:`.chebyshev.bases`
        - :mod:`.chebyshevu.bases`
        - :mod:`.legendre.bases`
        - :mod:`.ultraspherical.bases`
        - :mod:`.fourier.bases`
        - :mod:`.laguerre.bases`
        - :mod:`.hermite.bases`
        - :mod:`.jacobi.bases`

        The integer determines the number of times the basis function is
        differentiated. The test represents the matrix row
    trial : 2-tuple of (Basis, integer)
        Like test
    measure: function of coordinate, optional
        The measure is in physical coordinates, not in the reference domain.
    assemble : None or str, optional
        Determines how to perform the integration

        - 'quadrature' (default)
        - 'exact'
        - 'adaptive'

    kind : None or str, optional
        The kind of method used to do quadrature.

        - 'implemented'
        - 'stencil'
        - 'vandermonde'

    Note
    ----
    This function only performs 1D inner products and is unaware of any
    :class:`.TensorProductSpace`

    Example
    -------
    Compute mass matrix of Shen's Chebyshev Dirichlet basis:

    >>> from shenfun.spectralbase import inner_product
    >>> from shenfun.chebyshev.bases import ShenDirichlet
    >>> SD = ShenDirichlet(6)
    >>> B = inner_product((SD, 0), (SD, 0))
    >>> d = {-2: -np.pi/2,
    ...       0: np.array([1.5*np.pi, np.pi, np.pi, np.pi]),
    ...       2: -np.pi/2}
    >>> [np.all(B[k] == v) for k, v in d.items()]
    [True, True, True]
    """
    key = ((test[0].__class__, test[1]), (trial[0].__class__, trial[1]))
    mat = test[0]._get_mat()

    if not isinstance(measure, Number):
        # replace y, z with x if multidimensional
        x0 = measure.free_symbols
        assert len(x0) == 1
        x0 = x0.pop()
        x = sp.symbols('x', real=x0.is_real, positive=x0.is_positive)
        measure = measure.subs(x0, x)
        measure = test[0].map_expression_true_domain(measure, x)
        if measure.is_polynomial():
            measure = sp.simplify(measure)

        if key + (measure,) in mat:
            A = mat[key+(measure,)](test, trial, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)

        else:
            # By mapping measure to true domain, the expression may be split
            B = []
            for dv in split(measure, expand=False):
                sci = dv['coeff']
                msi = dv['x']
                newkey = key + (msi,)
                B.append(mat[newkey](test, trial, scale=sci, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution))
            A = B[0] if len(B) == 1 else np.sum(np.array(B, dtype=object))
    else:
        A = mat[key](test, trial, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        if measure != 1:
            A *= measure

    return A


def get_norm_sq(v, u, method):
    r"""Return square of weighted norm

    .. math::

        `(\phi_i, \phi_i)_w`

    where :math:`\phi_i` is the orthogonal basis for a spectral family.

    Parameters
    ----------
    v : instance of :class:`.SpectralBase`
        Orthogonal test function
    N : instance of :class:`.SpectralBase`
        Orthogonal trial function
    method : str
        Type of integration

        - 'exact'
        - 'quadrature'
    """
    if method == 'quadrature':
        return v.l2_norm_sq()[:min(v.N, u.N)]
    else:
        return np.array([v.L2_norm_sq(sp.S(i)) for i in range(min(v.N, u.N))]).astype(float)


class FuncWrap:

    # pylint: disable=too-few-public-methods, missing-docstring

    __slots__ = ('__doc__', '_func', '_input_array', '_output_array')

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

    __slots__ = ('_xfftn', '_input_array', '_output_array',
                 '_tmp_array', '__doc__')

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
