"""
Module for function spaces of generalized Jacobi type

Note the environment variable

    SHENFUN_JACOBI_MODE

that can be used for extended precision in this module.

SHENFUN_JACOBI_MODE = 'mpmath' will use the extended precision
mpmath module to compute inner products.

The precision can be set using, e.g.,

    from mpmath import mp
    mp.dps = 50

where mp.dps is the number of significant digits.

Note that extended precision is costly, but for some of the
matrices that can be created with the Jacobi bases it is necessary.
Also note the the higher precision is only used for assembling
matrices comuted with :func:`evaluate_basis_derivative_all`.
It has no effect for the matrices that are predifined in the
matrices.py module. Also note that the final matrix will be
in regular double precision. So the higher precision is only used
for the intermediate assembly.

"""

import os
import functools
import numpy as np
import sympy as sp
from scipy.special import eval_jacobi, roots_jacobi #, gamma
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict
from shenfun.forms.arguments import Function
from shenfun.chebyshev.bases import BCBiharmonic, BCDirichlet

try:
    import quadpy
    from mpmath import mp
    mp.dps = 30
    has_quadpy = True
except:
    has_quadpy = False
    mp = None

mode = os.environ.get('SHENFUN_JACOBI_MODE', 'numpy')
mode = mode if has_quadpy else 'numpy'

_x = sp.Symbol('x', real=True)

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

__all__ = ['JacobiBase', 'Orthogonal', 'ShenDirichlet', 'ShenBiharmonic',
           'ShenOrder6', 'mode', 'has_quadpy', 'mp']


class JacobiBase(SpectralBase):
    """Base class for all Jacobi spaces

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss
        alpha : number, optional
            Parameter of the Jacobi polynomial
        beta : number, optional
            Parameter of the Jacobi polynomial
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

    def __init__(self, N, quad="JG", alpha=0, beta=0, domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        SpectralBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.alpha = alpha
        self.beta = beta
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)

    @staticmethod
    def family():
        return 'jacobi'

    def reference_domain(self):
        return (-1, 1)

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              alpha=self.alpha,
                              beta=self.beta)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              alpha=self.alpha,
                              beta=self.beta)

    def get_orthogonal(self):
        return Orthogonal(self.N,
                          quad=self.quad,
                          domain=self.domain,
                          dtype=self.dtype,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates,
                          alpha=0,
                          beta=0)

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, self.alpha, self.beta)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        pw = quadpy.c1.gauss_jacobi(N, self.alpha, self.beta, 'mpmath')
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def jacobi(self, x, alpha, beta, N):
        V = np.zeros((x.shape[0], N))
        if mode == 'numpy':
            for n in range(N):
                V[:, n] = eval_jacobi(n, alpha, beta, x)
        else:
            for n in range(N):
                V[:, n] = sp.lambdify(_x, sp.jacobi(n, alpha, beta, _x), 'mpmath')(x)
        return V

    def derivative_jacobi(self, x, alpha, beta, k=1):
        V = self.jacobi(x, alpha+k, beta+k, self.N)
        if k > 0:
            Vc = np.zeros_like(V)
            for j in range(k, self.N):
                dj = np.prod(np.array([j+alpha+beta+1+i for i in range(k)]))
                #dj = gamma(j+alpha+beta+1+k) / gamma(j+alpha+beta+1)
                Vc[:, j] = (dj/2**k)*V[:, j-k]
            V = Vc
        return V

    def vandermonde(self, x):
        return self.jacobi(x, self.alpha, self.beta, self.shape(False))

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


class Orthogonal(JacobiBase):
    """Function space for regular (orthogonal) Jacobi functions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

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

    def __init__(self, N, quad="JG", alpha=-0.5, beta=-0.5, domain=(-1., 1.),
                 dtype=np.float, padding_factor=1, dealias_direct=False, coordinates=None):
        JacobiBase.__init__(self, N, quad=quad, alpha=alpha, beta=beta, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @property
    def is_orthogonal(self):
        return True

    #def get_orthogonal(self):
    #    return self

    def sympy_basis(self, i=0, x=_x):
        return sp.jacobi(i, self.alpha, self.beta, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array = eval_jacobi(i, self.alpha, self.beta, x, out=output_array)
        else:
            f = self.sympy_basis(i, _x)
            output_array[:] = sp.lambdify(_x, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights(mode=mode)[0]
        #x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)

        if mode == 'numpy':
            dj = np.prod(np.array([i+self.alpha+self.beta+1+j for j in range(k)]))
            output_array[:] = dj/2**k*eval_jacobi(i-k, self.alpha+k, self.beta+k, x)
        else:
            f = sp.jacobi(i, self.alpha, self.beta, _x)
            output_array[:] = sp.lambdify(_x, f.diff(_x, k), 'mpmath')(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights(mode=mode)[0]
        #if x.dtype == 'O':
        #    x = np.array(x, dtype=self.dtype)
        if mode == 'numpy':
            return self.derivative_jacobi(x, self.alpha, self.beta, k)
        else:
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N):
                V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        return self.vandermonde(x)


class ShenDirichlet(JacobiBase):
    """Jacobi function space for Dirichlet boundary conditions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

        bc : tuple of numbers
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
    def __init__(self, N, quad='JG', bc=(0, 0), domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        JacobiBase.__init__(self, N, quad=quad, alpha=-1, beta=-1, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        assert bc in ((0, 0), 'Dirichlet')
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

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
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def is_scaled(self):
        return False

    def slice(self):
        return slice(0, self.N-2)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        N = self.shape(False)
        V = np.zeros((x.shape[0], N))
        for i in range(N-2):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def sympy_basis(self, i=0, x=_x):
        return (1-x**2)*sp.jacobi(i, 1, 1, x)
        #return (1-x)**(-self.alpha)*(1+x)**(-self.beta)*sp.jacobi(i, -self.alpha, -self.beta, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        f = self.sympy_basis(i, _x)
        output_array[:] = sp.lambdify(_x, f.diff(_x, k), mode)(x)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        if mode == 'numpy':
            output_array = (1-x**2)*eval_jacobi(i, -self.alpha, -self.beta, x, out=output_array)
        else:
            f = self.sympy_basis(i, _x)
            output_array[:] = sp.lambdify(_x, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if mode == 'numpy':
            if x is None:
                x = self.mesh(False, False)
            V = np.zeros((x.shape[0], self.N))
            V[:, :-2] = self.jacobi(x, 1, 1, self.N-2)*(1-x**2)[:, np.newaxis]
        else:
            if x is None:
                x = self.mpmath_points_and_weights()[0]
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N-2):
                V[:, i] = self.evaluate_basis(x, i, output_array=V[:, i])
        return V

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, self.alpha+1, self.beta+1)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        pw = quadpy.c1.gauss_jacobi(N, self.alpha+1, self.beta+1, mode)
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def to_ortho(self, input_array, output_array=None):
        assert self.alpha == -1 and self.beta == -1
        if output_array is None:
            output_array = Function(input_array.function_space().get_orthogonal())
        else:
            output_array.fill(0)
        k = self.wavenumbers().astype(np.float)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        z = input_array[s0]*2*(k+1)/(2*k+3)
        output_array[s0] = z
        output_array[s1] -= z
        return output_array

    def _evaluate_scalar_product(self, fast_transform=True):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis


class ShenBiharmonic(JacobiBase):
    """Function space for Biharmonic boundary conditions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

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

    Note
    ----
    The generalized Jacobi function j^{alpha=-2, beta=-2} is used as basis. However,
    inner products are computed without weights, for alpha=beta=0.

    """
    def __init__(self, N, quad='JG', bc=(0, 0, 0, 0), domain=(-1., 1.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None, **kw):
        JacobiBase.__init__(self, N, quad=quad, alpha=-2, beta=-2, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        assert bc in ((0, 0, 0, 0), 'Biharmonic')
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0, x=_x):
        return (1-x**2)**2*sp.jacobi(i, 2, 2, x)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        N = self.shape(False)
        V = np.zeros((x.shape[0], N))
        for i in range(N-4):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        f = self.sympy_basis(i, _x)
        output_array[:] = sp.lambdify(_x, f.diff(_x, k), mode)(x)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array[:] = (1-x**2)**2*eval_jacobi(i, 2, 2, x, out=output_array)
        else:
            X = sp.symbols('x', real=True)
            f = self.sympy_basis(i, X)
            output_array[:] = sp.lambdify(X, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if mode == 'numpy':
            if x is None:
                x = self.mesh(False, False)
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            V[:, :-4] = self.jacobi(x, 2, 2, N-4)*((1-x**2)**2)[:, np.newaxis]
        else:
            if x is None:
                x = self.mpmath_points_and_weights()[0]
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N-4):
                V[:, i] = self.evaluate_basis(x, i, output_array=V[:, i])
        return V

    def _evaluate_scalar_product(self, fast_transform=True):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        pw = quadpy.c1.gauss_jacobi(N, 0, 0, 'mpmath')
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = Function(input_array.function_space().get_orthogonal())
        else:
            output_array.fill(0)
        k = self.wavenumbers().astype(np.float)
        _factor0 = 4*(k+2)*(k+1)/(2*k+5)/(2*k+3)
        _factor1 = (-2*(2*k+5)/(2*k+7))
        _factor2 = ((2*k+3)/(2*k+7))
        s0 = self.sl[slice(0, -4)]
        z = _factor0*input_array[s0]
        output_array[s0] = z
        output_array[self.sl[slice(2, -2)]] += z*_factor1
        output_array[self.sl[slice(4, None)]] += z*_factor2
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis


class ShenOrder6(JacobiBase):
    """Function space for 6th order equation

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

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

    Note
    ----
    The generalized Jacobi function j^{alpha=-3, beta=-3} is used as basis. However,
    inner products are computed without weights, for alpha=beta=0.

    """
    def __init__(self, N, quad='JG', domain=(-1., 1.), dtype=np.float, padding_factor=1, dealias_direct=False,
                 coordinates=None):
        JacobiBase.__init__(self, N, quad=quad, alpha=-3, beta=-3, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.bc = BoundaryValues(self, bc=(0,)*6)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def boundary_condition():
        return '6th order'

    def slice(self):
        return slice(0, self.N-6)

    def sympy_basis(self, i=0, x=_x):
        return (1-x**2)**3*sp.jacobi(i, 3, 3, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        if output_array is None:
            output_array = np.zeros(x.shape)
        f = self.sympy_basis(i, _x)
        output_array[:] = sp.lambdify(_x, f.diff(_x, k), mode)(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        N = self.shape(False)
        V = np.zeros((x.shape[0], N))
        for i in range(N-6):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array[:] = (1-x**2)**3*eval_jacobi(i, 3, 3, x, out=output_array)
        else:
            f = self.sympy_basis(i, _x)
            output_array[:] = sp.lambdify(_x, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if mode == 'numpy':
            if x is None:
                x = self.mesh(False, False)
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            V[:, :-6] = self.jacobi(x, 3, 3, N-6)*((1-x**2)**3)[:, np.newaxis]
        else:
            if x is None:
                x = self.mpmath_points_and_weights()[0]
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N-6):
                V[:, i] = self.evaluate_basis(x, i, output_array=V[:, i])
        return V

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        pw = quadpy.c1.gauss_jacobi(N, 0, 0, 'mpmath')
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def get_orthogonal(self):
        return Orthogonal(self.N, alpha=0, beta=0, dtype=self.dtype, domain=self.domain, coordinates=self.coors.coordinates)

    def _evaluate_scalar_product(self, fast_transform=True):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-6, None)]] = 0

    #def to_ortho(self, input_array, output_array=None):
    #    if output_array is None:
    #        output_array = np.zeros_like(input_array.v)
    #    k = self.wavenumbers().astype(np.float)
    #    _factor0 = 4*(k+2)*(k+1)/(2*k+5)/(2*k+3)
    #    _factor1 = (-2*(2*k+5)/(2*k+7))
    #    _factor2 = ((2*k+3)/(2*k+7))
    #    s0 = self.sl[slice(0, -4)]
    #    z = _factor0*input_array[s0]
    #    output_array[s0] = z
    #    output_array[self.sl[slice(2, -2)]] -= z*_factor1
    #    output_array[self.sl[slice(4, None)]] += z*_factor2
    #    return output_array
