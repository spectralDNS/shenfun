import functools
import numpy as np
from numpy.polynomial import hermite
from scipy.special import eval_hermite, factorial
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform
from shenfun.utilities import inheritdocstrings

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import


@inheritdocstrings
class Basis(SpectralBase):
    """Base class for Hermite functions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - HG - Hermite-Gauss

    Note
    ----
    We are using Hermite functions and not the regular Hermite polynomials
    as basis functions. A Hermite function is defined as

    .. math::

        H_k = P_k \cdot \frac{1}{\pi^{0.25} \sqrt{2^n n!}} \exp(-x^2/2)

    where :math:`H_k` and :math:`P_k` are the Hermite function and Hermite
    polynomials of order k, respectively.

    """

    def __init__(self, N=0, quad="HG", bc=(0., 0.)):
        SpectralBase.__init__(self, N, quad, domain=(-np.inf, np.inf))
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def family():
        return 'hermite'

    def reference_domain(self):
        return (-np.inf, np.inf)

    def domain_factor(self):
        return 1

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    def points_and_weights(self, N=None, map_true_domain=False, **kw):
        if N is None:
            N = self.N
        if self.quad == "HG":
            points, weights = hermite.hermgauss(N)
            weights *= np.exp(points**2)
        else:
            raise NotImplementedError

        return points, weights

    @staticmethod
    def factor(i):
        return 1./(np.pi**(0.25)*np.sqrt(2.**i)*np.sqrt(factorial(i)))

    def vandermonde(self, x):
        V = hermite.hermvander(x, self.N-1)
        return V

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array = eval_hermite(i, x, out=output_array)
        output_array *= np.exp(-x**2/2)*self.factor(i)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        M = V.shape[1]
        X = x[:, np.newaxis]
        if k == 1:
            D = np.zeros((M, M))
            D[:-1, :] = hermite.hermder(np.eye(M), 1)
            W = np.dot(V, D)
            W -= V*X
            V = W*np.exp(-X**2/2)
            V *= self.factor(np.arange(M))[np.newaxis, :]

        elif k == 2:
            W = (X**2 - 1 - 2*np.arange(M)[np.newaxis, :])*V
            V = W*self.factor(np.arange(M))[np.newaxis, :]*np.exp(-X**2/2)
            #D = np.zeros((M, M))
            #D[:-1, :] = hermite.hermder(np.eye(M), 1)
            #W = np.dot(V, D)
            #W = -2*X*W
            #D[-2:] = 0
            #D[:-2, :] = hermite.hermder(np.eye(M), 2)
            #W += np.dot(V, D)
            #W += (X**2-1)*V
            #W *= np.exp(-X**2/2)*self.factor(np.arange(M))[np.newaxis, :]
            #V[:] = W

        elif k == 0:
            V *= np.exp(-X**2/2)*self.factor(np.arange(M))[np.newaxis, :]

        else:
            raise NotImplementedError

        return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        V *= self.factor(np.arange(V.shape[1]))[np.newaxis, :]*np.exp(-x**2/2)[:, np.newaxis]
        return V

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        v = eval_hermite(i, x, out=output_array)
        if k == 1:
            D = np.zeros((self.N, self.N))
            D[:-1, :] = hermite.hermder(np.eye(self.N), 1)
            V = np.dot(v, D)
            V -= v*x[:, np.newaxis]
            V *= np.exp(-x**2/2)[:, np.newaxis]*self.factor(i)
            v[:] = V

        elif k == 2:
            W = (x[:, np.newaxis]**2 - 1 - 2*i)*V
            v[:] = W*self.factor(i)*np.exp(-x**2/2)[:, np.newaxis]

        elif k == 0:
            v *= np.exp(-x**2/2)[:, np.newaxis]*self.factor(i)

        else:
            raise NotImplementedError

        return v

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

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = u*self.factor(np.arange(self.N))
        y = hermite.hermval(x, w)
        output_array[:] = y * np.exp(-x**2/2)
        return output_array

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self):
        return self
