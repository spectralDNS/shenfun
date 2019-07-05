import functools
import numpy as np
import sympy as sp
from scipy.special import eval_jacobi, roots_jacobi #, gamma
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict
from shenfun.utilities import inheritdocstrings

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import


@inheritdocstrings
class JacobiBase(SpectralBase):
    """Base class for all Jacobi bases

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - JG - Jacobi-Gauss

    """

    def __init__(self, N=0, quad="JG", alpha=0, beta=0, domain=(-1., 1.)):
        SpectralBase.__init__(self, N, quad, domain=domain)
        self.alpha = alpha
        self.beta = beta
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)

    @staticmethod
    def family():
        return 'jacobi'

    def reference_domain(self):
        return (-1., 1.)

    def points_and_weights(self, N=None, map_true_domain=False, mode='numpy'):
        if N is None:
            N = self.N
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, self.alpha, self.beta)
        if mode == 'mpmath':
            try:
                import quadpy
                from mpmath import mp
                mp.dps = 30
                pw = quadpy.line_segment.gauss_jacobi(N, self.alpha, self.beta, 'mpmath')
                points = pw.points
                weights = pw.weights
            except:
                pass

        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def jacobi(self, x, alpha, beta, N):
        V = np.zeros((x.shape[0], N))
        for n in range(N):
            V[:, n] = eval_jacobi(n, alpha, beta, x)
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
        return self.jacobi(x, self.alpha, self.beta, self.N)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_jacobi(i, self.alpha, self.beta, x, out=output_array)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
            #x = self.points_and_weights(mode='mpmath')[0]
        if x.dtype == 'O':
            x = np.array(x, dtype=np.float)
        return self.derivative_jacobi(x, self.alpha, self.beta, k)
        #V = np.zeros((x.shape[0], self.N))
        #for i in range(self.N):
        #    V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        #return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
            #x = self.points_and_weights(mode='mpmath')[0]
        dj = np.prod(np.array([i+self.alpha+self.beta+1+j for j in range(k)]))
        return dj/2**k*eval_jacobi(i-k, self.alpha+k, self.beta+k, x)
        #X = sp.symbols('X')
        #f = sp.jacobi(i, self.alpha, self.beta, X)
        #output_array[:] = sp.lambdify(X, f.diff(X, k), 'mpmath')(x)
        #return output_array

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
        return Basis(self.N, alpha=self.alpha, beta=self.beta, domain=self.domain)

@inheritdocstrings
class Basis(JacobiBase):
    """Basis for regular (orthogonal) Jacobi functions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - JG - Jacobi-Gauss

    """

    def __init__(self, N=0, quad="JG", alpha=-0.5, beta=-0.5, domain=(-1., 1.)):
        JacobiBase.__init__(self, N, quad, alpha, beta, domain)
        self.plan(N, 0, np.float, {})

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self):
        return self

    def sympy_basis(self, i=0):
        x = sp.symbols('x')
        return sp.jacobi(i, self.alpha, self.beta, x)


class ShenDirichletBasis(JacobiBase):
    """Jacobi basis for Dirichlet boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - JG - Jacobi-Gauss

        bc : tuple of numbers
             Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
                 The computational domain

    """
    def __init__(self, N=0, quad='JG', bc=(0, 0), domain=(-1., 1.)):
        JacobiBase.__init__(self, N=N, quad=quad, alpha=-1, beta=-1, domain=domain)
        assert bc in ((0, 0), 'Dirichlet')
        from shenfun.tensorproductspace import BoundaryValues
        self.bc = BoundaryValues(self, bc=bc)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    def is_scaled(self):
        return False

    def slice(self):
        return slice(0, self.N-2)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.points_and_weights(mode='mpmath')[0]
        V = np.zeros((x.shape[0], self.N))
        for i in range(self.N-2):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def sympy_basis(self, i=0):
        x = sp.symbols('x')
        return (1-x**2)*sp.jacobi(i, 1, 1, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights(mode='mpmath')[0]
        if output_array is None:
            output_array = np.zeros_like(x)
        X = sp.symbols('X')
        f = (1-X**2)*sp.jacobi(i, 1, 1, X)
        mode = 'numpy'
        if x.dtype == 'O':
            mode = 'mpmath'

        output_array[:] = sp.lambdify(X, f.diff(X, k), mode)(x)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = (1-x**2)*eval_jacobi(i, 1, 1, x, out=output_array)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = np.zeros((x.shape[0], self.N))
        V[:, :-2] = self.jacobi(x, 1, 1, self.N-2)*(1-x**2)[:, np.newaxis]
        return V

    def points_and_weights(self, N=None, map_true_domain=False, mode='numpy'):
        if N is None:
            N = self.N
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if mode == 'mpmath':
            try:
                import quadpy
                from mpmath import mp
                mp.dps = 30
                pw = quadpy.line_segment.gauss_jacobi(N, 0, 0, 'mpmath')
                points = pw.points
                weights = pw.weights
            except:
                pass
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array.v)
        k = self.wavenumbers().astype(np.float)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        z = input_array[s0]*2*(k+1)/(2*k+3)
        output_array[s0] = z
        output_array[s1] -= z
        return output_array

    def get_orthogonal(self):
        return Basis(self.N, alpha=0, beta=0, domain=self.domain)

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)
        output_array[self.sl[slice(-2, None)]] = 0

class ShenBiharmonicBasis(JacobiBase):
    """Basis for Biharmonic boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - JG - Jacobi-Gauss

        domain : 2-tuple of floats, optional
                 The computational domain

    Note
    ----
    The generalized Jacobi function j^{alpha=-2, beta=-2} is used as basis. However,
    inner products are computed without weights, for alpha=beta=0.

    """
    def __init__(self, N=0, quad='JG', domain=(-1., 1.)):
        JacobiBase.__init__(self, N=N, quad=quad, alpha=-2, beta=-2, domain=domain)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0):
        x = sp.symbols('x')
        return (1-x**2)**2*sp.jacobi(i, 2, 2, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights(mode='mpmath')[0]
        if output_array is None:
            output_array = np.zeros_like(x)
        X = sp.symbols('X')
        f = (1-X**2)**2*sp.jacobi(i, 2, 2, X)
        mode = 'numpy'
        if x.dtype == 'O':
            mode = 'mpmath'
        output_array[:] = sp.lambdify(X, f.diff(X, k), mode)(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.points_and_weights(mode='mpmath')[0]
        V = np.zeros((x.shape[0], self.N))
        for i in range(self.N-4):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = (1-x**2)**2*eval_jacobi(i, 2, 2, x, out=output_array)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = np.zeros((x.shape[0], self.N))
        V[:, :-4] = self.jacobi(x, 2, 2, self.N-4)*((1-x**2)**2)[:, np.newaxis]
        return V

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)
        output_array[self.sl[slice(-4, None)]] = 0

    def points_and_weights(self, N=None, map_true_domain=False, mode='numpy'):
        if N is None:
            N = self.N
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if mode == 'mpmath':
            try:
                import quadpy
                from mpmath import mp
                mp.dps = 30
                pw = quadpy.line_segment.gauss_jacobi(N, 0, 0, 'mpmath')
                points = pw.points
                weights = pw.weights
            except:
                pass
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array.v)
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

    def get_orthogonal(self):
        return Basis(self.N, alpha=0, beta=0, domain=self.domain)


class ShenOrder6Basis(JacobiBase):
    """Basis for 6th order equation

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - JG - Jacobi-Gauss

        domain : 2-tuple of floats, optional
                 The computational domain

    Note
    ----
    The generalized Jacobi function j^{alpha=-3, beta=-3} is used as basis. However,
    inner products are computed without weights, for alpha=beta=0.

    """
    def __init__(self, N=0, quad='JG', domain=(-1., 1.)):
        JacobiBase.__init__(self, N=N, quad=quad, alpha=-3, beta=-3, domain=domain)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return '6th order'

    def slice(self):
        return slice(0, self.N-6)

    def sympy_basis(self, i=0):
        x = sp.symbols('x')
        return (1-x**2)**3*sp.jacobi(i, 3, 3, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights(mode='mpmath')[0]
        if output_array is None:
            output_array = np.zeros_like(x)
        X = sp.symbols('X')
        f = (1-X**2)**3*sp.jacobi(i, 3, 3, X)
        mode = 'numpy'
        if x.dtype == 'O':
           mode = 'mpmath'
        output_array[:] = sp.lambdify(X, f.diff(X, k), mode)(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.points_and_weights(mode='mpmath')[0]
        V = np.zeros((x.shape[0], self.N))
        for i in range(self.N-6):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = (1-x**2)**3*eval_jacobi(i, 3, 3, x, out=output_array)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = np.zeros((x.shape[0], self.N))
        V[:, :-6] = self.jacobi(x, 3, 3, self.N-6)*((1-x**2)**3)[:, np.newaxis]
        return V

    def points_and_weights(self, N=None, map_true_domain=False, mode='numpy'):
        if N is None:
            N = self.N
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if mode == 'mpmath':
            try:
                import quadpy
                from mpmath import mp
                mp.dps = 30
                pw = quadpy.line_segment.gauss_jacobi(N, 0, 0, 'mpmath')
                points = pw.points
                weights = pw.weights
            except:
                pass

        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def get_orthogonal(self):
        return Basis(self.N, alpha=0, beta=0, domain=self.domain)

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)
        output_array[self.sl[slice(-6, None)]] = 0

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