r"""
This module contains classes for working with the spectral-Galerkin method

There are currently classes for 11 bases and corresponding function spaces

All bases have expansions

   :math:`u(x) = \sum_{k\in\mathcal{I}}\hat{u}_{k} \phi_k(x)`

where :math:`\mathcal{I}` the index set of the basis, and the index set differs from
base to base, see function space definitions below. :math:`\phi_k` is the k't
basis function in the basis. It is also called a test function, whereas :math:`u(x)`
often is called a trial function.

Chebyshev:
    ChebyshevBasis:
        basis functions: :math:`\phi_k = T_k`

        basis: :math:`span(T_k, k=0,1,..., N-1)`

    ShenDirichletBasis:
        basis functions:
            :math:`\phi_k = T_k-T_{k+2}`

            :math:`\phi_{N-2} = 0.5(T_0+T_1)` for Poisson's equation

            :math:`\phi_{N-1} = 0.5(T_0-T_1)` for Poisson's equation

        basis: :math:`span(\phi_k, k=0,1,...,N-1)`

        where :math:`u(1)=a, u(-1)=b`, such that :math:`\hat{u}_{N-2}=a,
        \hat{u}_{N-1}=b`.

        Note that there are only N-2 unknown coefficients, :math:`\hat{u}_k`,
        since :math:`\hat{u}_{N-2}` and :math:`\hat{u}_{N-1}` are determined by
        boundary conditions. Inhomogeneous boundary conditions are possible for
        the Poisson equation, because :math:`\phi_{N-1}` and :math:`\phi_{N-2}`
        are in the kernel of the Poisson operator. For homogeneous boundary
        conditions :math:`\phi_{N-2}` and :math:`\phi_{N-1}` are simply ignored.

    ShenNeumannBasis:
        basis function:
            :math:`\phi_k = T_k-\left(\frac{k}{k+2}\right)^2T_{k+2}`

        basis:
            :math:`span(\phi_k, k=1,2,...,N-3)`

        Homogeneous Neumann boundary conditions, :math:`u'(\pm 1) = 0`, and
        zero weighted mean: :math:`\int_{-1}^{1}u(x)w(x)dx = 0`.

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = T_k - \frac{2(k+2)}{k+3}T_{k+2} + \frac{k+1}{k+3}T_{k+4}`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-5)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and
        :math:`u'(\pm 1)=0`.

Legendre:
    LegendreBasis:
        basis function:
            :math:`\phi_k = L_k`

        basis:
            :math:`span(L_k, k=0,1,...N-1)`

    ShenDirichletBasis:
        basis function:
            :math:`\phi_k = L_k-L_{k+2}`

            :math:`\phi_{N-2} = 0.5(L_0+L_1)`, for Poisson's equation

            :math:`\phi_{N-1} = 0.5(L_0-L_1)`, for Poisson's equation

        basis:
            :math:`span(\phi_k, k=0,1,...,N-1)`

        where :math:`u(1)=a, u(-1)=b`, such that
        :math:`\hat{u}_{N-2}=a, \hat{u}_{N-1}=b`

        Note that there are only N-2 unknown coefficients, :math:`\hat{u}_k`,
        since :math:`\hat{u}_{N-2}` and :math:`\hat{u}_{N-1}` are determined by
        boundary conditions. Inhomogeneous boundary conditions are possible for
        the Poisson equation, because :math:`\phi_{N-1}` and :math:`\phi_{N-2}`
        are in the kernel of the Poisson operator. For homogeneous boundary
        conditions :math:`\phi_{N-2}` and :math:`\phi_{N-1}` are simply ignored.

    ShenNeumannBasis:
        basis function:
            :math:`\phi_k = L_k-\frac{k(k+1)}{(k+2)(k+3)}L_{k+2}`

        basis:
            :math:`span(\phi_k, k=1,2,...,N-3)`

        Homogeneous Neumann boundary conditions, :math:`u'(\pm 1) = 0`, and
        zero mean: :math:`\int_{-1}^{1}u(x)dx = 0`.

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = L_k - \frac{2(2k+5)}{2k+7}L_{k+2} + \frac{2k+3}{2k+7}L_{k+4}`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-5)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and
        :math:`u'(\pm 1)=0`.

Laguerre:
    LaguerreBasis:
        basis function:
            :math:`\phi_k(x) = L_k(x) \cdot \exp(-x)`

        basis:
            :math:`span(L_k, k=0,1,...N-1)`

        where :math:`L_k` is the Laguerre polynomial of order k.

    ShenDirichletBasis:
        basis function:
            :math:`\phi_k = (L_k-L_{k+1})\cdot \exp(-x)`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-2)`

        Homogeneous Dirichlet for domain [0, inf).

Hermite:
    Basis:
        basis function:
            :math:`\phi_k(x) = H_k(x) \cdot \exp(-x^2/2)/(\pi^{0.25}\sqrt{2^k k!})`

        basis:
            :math:`span(\phi_k, k=0,1,...N-1)`

        where :math:`H_k` is the Hermite polynomial of order k.

Jacobi:
    Basis:
        basis function:
            :math:`\phi_k = J_k(\alpha, \beta)`

        basis:
            :math:`span(L_k, k=0,1,...N-1)`

        where :math:`J_k(\alpha, \beta)` is the regular Jacobi polynomial and
        :math:`\alpha > -1` and :math:`\beta > -1`.

    ShenDirichletBasis:
        basis function:
            :math:`\phi_k = j_k(-1, -1)`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-3)`

        where :math:`j_k` is the generalized Jacobi polynomial

    ShenBiharmonicBasis:
        basis function:
            :math:`\phi_k = j_k(-2, -2)`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-5)`

        Homogeneous Dirichlet and Neumann, :math:`u(\pm 1)=0` and
        :math:`u'(\pm 1)=0`.

    ShenOrder6Basis:
        basis function:
            :math:`\phi_k = j_k(-3, -3)`

        basis:
            :math:`span(\phi_k, k=0,1,...,N-7)`

        Homogeneous :math:`u(\pm 1)=u'(\pm 1)=u''(\pm 1)=0`.

Fourier:
    R2CBasis and C2CBasis:
        basis function:
            :math:`\phi_k = exp(ikx)`

        basis:
            :math:`span(\phi_k, k=-N/2, -N/2+1, ..., N/2-1)`

        Note that if N is even, then the Nyquist frequency (-N/2) requires
        special attention for the R2CBasis. We should then have been using
        a Fourier interpolator that symmetrizes by adding and extra
        :math:`\phi_{N/2}` and by multiplying :math:`\phi_{N/2}` and
        :math:`\phi_{-N/2}` with 0.5. This effectively sets the
        Nyquist frequency (-N/2) to zero for odd derivatives. This is
        not done automatically by shenfun. Instead we recommend using
        the :func:`.SpectralBase.mask_nyquist` function that effectively sets the
        Nyquist frequency to zero (if N is even).

    R2CBasis and C2CBasis are the same, but R2CBasis is used on real physical
    data and it takes advantage of Hermitian symmetry,
    :math:`\hat{u}_{-k} = conj(\hat{u}_k)`, for :math:`k = 1, ..., N/2`

Each class has methods for moving (fast) between spectral and physical space, and
for computing the (weighted) scalar product.

"""
#pylint: disable=unused-argument, not-callable, no-self-use, protected-access, too-many-public-methods, missing-docstring

import importlib
import sympy as sp
import numpy as np
from mpi4py_fft import fftw
from .utilities import CachedArrayDict, split
from .coordinates import Coordinates
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
            - LG - Legendre-Gauss or Laguerre-Gauss
            - HG - Hermite-Gauss

        padding_factor : float, optional
            For padding backward transform (for dealiasing)
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        dealias_direct : bool, optional
            If True, then set all upper 2/3 wavenumbers to zero before backward
            transform. Cannot be used if padding_factor is different than 1.
        coordinates: 2-tuple (coordinate, position vector), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

            where theta and rv are the first and second items in the 2-tuple.

    """
    # pylint: disable=method-hidden, too-many-instance-attributes

    def __init__(self, N, quad='', padding_factor=1, domain=(-1., 1.), dtype=None,
                 dealias_direct=False, coordinates=None):
        self.N = N
        self.domain = domain
        self.quad = quad
        self.axis = 0
        self.bc = None
        self.padding_factor = np.floor(N*padding_factor)/N
        self.dealias_direct = dealias_direct
        self._mass = None         # Mass matrix (if needed)
        self._M = 1.0             # Normalization factor
        self._xfftn_fwd = None    # external forward transform function
        self._xfftn_bck = None    # external backward transform function
        self.hi = np.ones(1, dtype=object)  # Integral measure (in addition to weight)
        coors = coordinates if coordinates is not None else ((sp.Symbol('x', real=True),),)*2
        self.coors = Coordinates(*coors)
        self.hi = self.coors.get_scaling_factors()
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
        """
        raise NotImplementedError

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        r"""Return points and weights of quadrature using extended precision
        mpmath if available

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
        If not implemented, or if mpmath/quadpy are not available, then simply
        returns the regular numpy :func:`points_and_weights`.
        """
        return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)

    def mesh(self, bcast=True, map_true_domain=True, uniform=False):
        """Return the computational mesh

        Parameters
        ----------
            bcast : bool
                Whether or not to broadcast to :meth:`.SpectralBase.dimensions`
                if basis belongs to a :class:`.TensorProductSpace`
            map_true_domain : bool, optional
                Whether or not to map points to true domain
            uniform : bool, optional
                Use uniform mesh instead of quadrature if True
        """
        N = self.shape(False)
        if self.family() == 'fourier':
            uniform = False
        if uniform is False:
            X = self.points_and_weights(N=N, map_true_domain=map_true_domain)[0]
        else:
            d = self.domain
            X = np.linspace(d[0], d[1], N)
            if map_true_domain is False:
                X = self.map_reference_domain(X)
        if bcast is True:
            X = self.broadcast_to_ndims(X)
        return X

    def curvilinear_mesh(self, uniform=False):
        """Return curvilinear mesh of basis

        Parameters
        ----------
        uniform : bool, optional
            Use uniform mesh
        """
        x = self.mesh(uniform=uniform)
        psi = self.coors.coordinates[0]
        xx = []
        for rv in self.coors.coordinates[1]:
            xx.append(sp.lambdify(psi, rv)(x))
        return xx

    def wavenumbers(self, bcast=True, **kw):
        """Return the wavenumbermesh

        Parameters
        ----------
            bcast : bool
                Whether or not to broadcast to :meth:`.SpectralBase.dimensions`
                if basis belongs to a :class:`.TensorProductSpace`

        """
        s = self.slice()
        k = np.arange(s.start, s.stop)
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def broadcast_to_ndims(self, x):
        """Return 1D array ``x`` as an array of shape according to the
        :meth:`.dimensions` of the :class:`.TensorProductSpace` class
        that this base (self) belongs to.

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
        s = [np.newaxis]*self.dimensions
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

        self.scalar_product._input_array = self.get_measured_array(self.scalar_product._input_array)

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
                Expansion coefficients
            output_array : array, optional
                Function values on quadrature mesh
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

    def backward_uniform(self, input_array=None, output_array=None):
        """Evaluate function on uniform mesh

        Parameters
        ----------
            input_array : array, optional
                Expansion coefficients
            output_array : array, optional
                Function values on quadrature mesh

        Note
        ----
        If input_array/output_array are not given, then use predefined arrays
        as planned with self.plan

        """
        if self.family() == 'fourier': # Fourier is already using uniform mesh. Use fast transform.
            return self.backward(input_array, output_array)

        if input_array is not None:
            self.backward.input_array[...] = input_array

        self._padding_backward(self.backward.input_array,
                               self.backward.tmp_array)

        self.vandermonde_evaluate_expansion_all(self.backward.tmp_array,
                                                self.backward.output_array,
                                                x=self.mesh(bcast=False, map_true_domain=False, uniform=True))

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
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        X = sp.symbols('x', real=True)
        f = self.sympy_basis(i, X)
        output_array[:] = sp.lambdify(X, f)(x)
        return output_array

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
        #warnings.warn('Using slow sympy evaluate_basis_derivative')
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = np.atleast_1d(x)
        X = sp.symbols('x', real=True)
        basis = self.sympy_basis(i=i, x=X).diff(X, k)
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
        weights = self.points_and_weights(int(self.N*self.padding_factor))[1]
        P = self.evaluate_basis_all(argument=0)
        if input_array.ndim == 1:
            output_array[slice(0, self.N)] = np.dot(input_array*weights, np.conj(P))

        else: # broadcasting
            bc_shape = [np.newaxis,]*input_array.ndim
            bc_shape[self.axis] = slice(None)
            fc = np.moveaxis(input_array*weights[tuple(bc_shape)], self.axis, -1)
            output_array[self.sl[slice(0, self.N)]] = np.moveaxis(np.dot(fc, np.conj(P)), -1, self.axis)
            #output_array[:] = np.moveaxis(np.tensordot(input_array*weights[bc_shape], np.conj(P), (self.axis, 0)), -1, self.axis)

        assert output_array is self.scalar_product.output_array

    def vandermonde_evaluate_expansion_all(self, input_array, output_array, x=None):
        """Naive implementation of evaluate_expansion_all

        Parameters
        ----------
            input_array : array
                Expansion coefficients
            output_array : array
                Function values on quadrature mesh
            x : mesh or None, optional

        """
        P = self.evaluate_basis_all(x=x, argument=1)
        if output_array.ndim == 1:
            output_array = np.dot(P, input_array, out=output_array)
        else:
            fc = np.moveaxis(input_array, self.axis, -2)
            shape = [slice(None)]*input_array.ndim
            shape[-2] = slice(0, self.N)
            array = np.dot(P, fc[tuple(shape)])
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
        P = self.evaluate_basis_all(x=points, argument=1)

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

        Returns
        -------
            array

        """
        #assert self.N == array.shape[self.axis]
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

        if 'builders' in self._xfftn_fwd.__module__: #pragma: no cover

            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
                backward=False,
            )
            opts.update(options)

            n = shape[axis]
            U = fftw.aligned(shape, dtype=dtype)
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
            U = fftw.aligned(shape, dtype=dtype)
            xfftn_fwd = plan_fwd(U, n, (axis,), threads=threads, flags=flags)
            V = xfftn_fwd.output_array

            if np.issubdtype(dtype, np.floating):
                flags = (fftw.flag_dict[opts['planner_effort']],)

            xfftn_bck = plan_bck(V, n, (axis,), threads=threads, flags=flags, output_array=U)
            V.fill(0)
            U.fill(0)
            self._M = xfftn_fwd.get_normalization()

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

        # scalar_product is not padded, just the forward/backward
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    def _get_truncarray(self, shape, dtype):
        shape = list(shape) if np.ndim(shape) else [shape]
        shape[self.axis] = int(np.round(shape[self.axis] / self.padding_factor))
        return fftw.aligned(shape, dtype=dtype)

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
            output_array = np.zeros(x.shape, dtype=self.dtype)
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

    def sympy_basis(self, i=0, x=None):
        """Return basis function `i` as sympy function

        Parameters
        ----------
        i : int, optional
            The degree of freedom of the basis function
        x : sympy Symbol, optional
        """
        raise NotImplementedError

    def sympy_basis_all(self, x=None):
        """Return all basis functions as sympy functions"""
        return np.array([self.sympy_basis(i, x=x) for i in range(self.slice().start, self.slice().stop)])

    def sympy_weight(self, x=None):
        return 1

    def reference_domain(self):
        """Return reference domain of basis"""
        raise NotImplementedError

    def slice(self):
        """Return index set of current basis"""
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
        return int(np.floor(self.padding_factor*self.N))

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
    def dtype(self):
        """Return datatype basis is planned for"""
        return self.forward.input_array.dtype

    @property
    def dimensions(self):
        """Return the dimensions (the number of bases) of the
        :class:`.TensorProductSpace` class this basis is planned for.
        """
        if self.tensorproductspace:
            return self.tensorproductspace.dimensions
        return self.forward.input_array.ndim

    @property
    def tensorproductspace(self):
        """Return the last :class:`.TensorProductSpace` this basis has been
        planned for (if planned)

        Note
        ----
        A basis may be part of several :class:`.TensorProductSpace`s, but they
        all need to be of the same global shape.
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
    def is_composite_space(self):
        return 0

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    @property
    def rank(self):
        """Return tensor rank of basis"""
        return 0

    @property
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
        dx = self.hi.prod()
        key = ((self.__class__, 0), (self.__class__, 0))
        if self.tensorproductspace:
            dx = self.tensorproductspace.hi.prod()
            msdict = split(dx)
            assert len(msdict) == 1
            dx = msdict[0]['xyzrs'[self.axis]]
        if not dx == 1:
            x0 = dx.free_symbols
            if len(x0) > 1:
                raise NotImplementedError("Cannot use forward for Curvilinear coordinates with unseparable measure - Use inner with mass matrix for tensor product space")
            x0 = x0.pop()
            x = sp.symbols('x', real=x0.is_real, positive=x0.is_positive)
            dx = dx.subs(x0, x)
            key = key + (self.domain, dx)

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

    def get_bcmass_matrix(self, hi=1):
        msx = 'xyzrs'[self.axis]
        dV = split(hi)
        assert len(dV) == 1
        dv = dV[0]
        msi = dv[msx]
        return inner_product((self, 0), (self.get_bc_basis(), 0), msi)

    def get_measured_weights(self, N=None, measure=1):
        """Return weights times `measure`

        Parameters
        ----------
        N : integer, optional
            The number of quadrature points
        measure : None or `sympy.Expr`
        """
        if N is None:
            N = self.N
        xm, wj = self.mpmath_points_and_weights(N, map_true_domain=True)
        if measure == 1:
            return wj

        s = measure.free_symbols
        assert len(s) == 1
        s = s.pop()
        xj = sp.lambdify(s, measure)(xm)
        if wj.shape[0] == 1:
            wj = np.broadcast_to(wj, xj.shape).copy()
        wj *= xj
        return wj

    def get_measured_array(self, array):
        """Return weights times `measure`

        Parameters
        ----------
        N : integer, optional
            The number of quadrature points
        measure : None or `sympy.Expr`

        Note
        ----
        If basis is part of a `TensorProductSpace`, then the
        array will be measured there. So in that case, just return
        the array unchanged.
        """
        if self.tensorproductspace:
            return array

        measure = self.hi.prod()
        if measure == 1:
            return array

        xm = self.mpmath_points_and_weights(self.N, map_true_domain=True)[0]
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

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        """Return space (otherwise as self) to be used for dealiasing

        Parameters
        ----------
        padding_factor : float, optional
            Create output array of shape padding_factor time non-padded shape
        dealias_direct : bool, optional
            If True, set upper 2/3 of wavenumbers to zero before backward transform.
            Cannot be used together with padding_factor different than 1.

        Returns
        -------
        SpectralBase
            The space to be used for dealiasing
        """
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates)

    def get_refined(self, N):
        """Return space (otherwise as self) with N quadrature points

        Parameters
        ----------
        N : int
            The number of quadrature points for returned space

        Returns
        -------
        SpectralBase
            A new space with new number of quadrature points, otherwise as self.
        """
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates)

    def _truncation_forward(self, padded_array, trunc_array):
        if not id(trunc_array) == id(padded_array):
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            s = self.sl[slice(0, N)]
            trunc_array[:] = padded_array[s]

    def _padding_backward(self, trunc_array, padded_array):
        if not id(trunc_array) == id(padded_array):
            padded_array.fill(0)
            N = trunc_array.shape[self.axis]
            _sn = self.sl[slice(0, N)]
            padded_array[_sn] = trunc_array[_sn]

        elif self.dealias_direct:
            su = self.sl[slice(2*self.N//3, None)]
            padded_array[su] = 0


class MixedBasis(object):
    """Class for composite bases in 1D

    Parameters
    ----------
    spaces : list
        List of bases

    """
    def __init__(self, bases):
        self.bases = bases
        self.forward = VectorBasisTransform([basis.forward for basis in bases])
        self.backward = VectorBasisTransform([basis.backward for basis in bases])
        self.scalar_product = VectorBasisTransform([basis.scalar_product for basis in bases])

    def dims(self):
        """Return dimensions (degrees of freedom) for MixedBasis"""
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
        """Return shape of arrays for MixedBasis

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
        return None

    @property
    def dimensions(self):
        """Return dimension of scalar space"""
        return self.bases[0].dimensions

class VectorBasisTransform(object):

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


def inner_product(test, trial, measure=1):
    """Return 1D weighted inner product of bilinear form

    Parameters
    ----------
        test : 2-tuple of (Basis, integer)
            Basis is any of the classes from

            - :mod:`.chebyshev.bases`
            - :mod:`.legendre.bases`
            - :mod:`.fourier.bases`
            - :mod:`.laguerre.bases`
            - :mod:`.hermite.bases`
            - :mod:`.jacobi.bases`

            The integer determines the number of times the basis is
            differentiated. The test represents the matrix row
        trial : 2-tuple of (Basis, integer)
            Like test
        measure: function of coordinate, optional

    Note
    ----
    This function only performs 1D inner products and is unaware of any
    :class:`.TensorProductSpace`

    Example
    -------
    Compute mass matrix of Shen's Chebyshev Dirichlet basis:

    >>> from shenfun.spectralbase import inner_product
    >>> from shenfun.chebyshev.bases import ShenDirichletBasis
    >>> SD = ShenDirichletBasis(6)
    >>> B = inner_product((SD, 0), (SD, 0))
    >>> d = {-2: np.array([-np.pi/2]),
    ...       0: np.array([1.5*np.pi, np.pi, np.pi, np.pi]),
    ...       2: np.array([-np.pi/2])}
    >>> [np.all(B[k] == v) for k, v in d.items()]
    [True, True, True]
    """
    assert trial[0].__module__ == test[0].__module__
    key = ((test[0].__class__, test[1]), (trial[0].__class__, trial[1]))
    sc = 1
    if measure != 1:
        x0 = measure.free_symbols
        assert len(x0) == 1
        x0 = x0.pop()
        x = sp.symbols('x', real=x0.is_real, positive=x0.is_positive)
        measure = measure.subs(x0, x)
        if measure.subs(x, 1) < 0:
            sc = -1
            measure *= sc
        key = key + (test[0].domain, measure)

    mat = test[0]._get_mat()
    A = mat[key](test, trial, measure=measure)
    A.scale *= sc
    if not test[0].domain_factor() == 1:
        A.scale *= test[0].domain_factor()**(test[1]+trial[1])
    return A

class FuncWrap(object):

    # pylint: disable=too-few-public-methods, missing-docstring

    __slots__ = ('_func', '_input_array', '_output_array')

    def __init__(self, func, input_array, output_array):
        object.__setattr__(self, '_func', func)
        object.__setattr__(self, '_input_array', input_array)
        object.__setattr__(self, '_output_array', output_array)
        #object.__setattr__(self, '__doc__', func.__doc__)

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
