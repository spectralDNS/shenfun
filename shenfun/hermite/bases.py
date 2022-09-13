import functools
import sympy as sp
import numpy as np
from numpy.polynomial import hermite
from scipy.special import eval_hermite, factorial
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

bases = ['Orthogonal']
bcbases = []
testbases = []
__all__ = bases

class Orthogonal(SpectralBase):
    r"""Function space for Hermite functions

    The orthogonal basis is the Hermite function

    .. math::

        \phi_k = H_k \frac{1}{\pi^{0.25} \sqrt{2^n n!}} \exp(-x^2/2), \quad k = 0, 1, \ldots, N-1,

    where :math:`\phi_k` and :math:`H_k` are the Hermite function and Hermite
    polynomials of order k, respectively.

    Parameters
    ----------
    N : int, optional
        Number of quadrature points. Should be even for efficiency, but
        this is not required.
    padding_factor : float, optional
        Factor for padding backward transforms. padding_factor=1.5
        corresponds to a 3/2-rule for dealiasing.
    dealias_direct : bool, optional
        True for dealiasing using 2/3-rule. Must be used with
        padding_factor = 1.
    dtype : data-type, optional
        Type of input data in real physical space. Will be overloaded when
        basis is part of a :class:`.TensorProductSpace`.
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    Note
    ----
    We are using Hermite functions and not the regular Hermite polynomials
    as basis functions.

    """
    def __init__(self, N, dtype=float, padding_factor=1, dealias_direct=False,
                 coordinates=None, **kw):
        SpectralBase.__init__(self, N, quad="HG", domain=(-sp.S.Infinity, sp.S.Infinity),
                              dtype=dtype, padding_factor=padding_factor,
                              dealias_direct=dealias_direct, coordinates=coordinates)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def family():
        return 'hermite'

    def reference_domain(self):
        return (-sp.S.Infinity, sp.S.Infinity)

    def domain_factor(self):
        return 1

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'H'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        if self.quad == "HG":
            points, weights = hermite.hermgauss(N)
            if weighted:
                weights *= np.exp(points**2)
        else:
            raise NotImplementedError

        return points, weights

    @staticmethod
    def factor(i):
        return 1./(np.pi**(0.25)*np.sqrt(2.**i)*np.sqrt(factorial(i)))

    def vandermonde(self, x):
        V = hermite.hermvander(x, self.shape(False)-1)
        return V

    def orthogonal_basis_function(self, i=0, x=sp.symbols('x')):
        return sp.hermite(i, x)*sp.exp(-x**2/2)*self.factor(i)

    def L2_norm_sq(self, i):
        return 1

    def l2_norm_sq(self, i=None):
        if i is None:
            return np.ones(self.N)
        return 1

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

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        w = u*self.factor(np.arange(self.N))
        y = hermite.hermval(x, w)
        output_array[:] = y * np.exp(-x**2/2)
        return output_array

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self, **kwargs):
        d = dict(quad=self.quad,
                 domain=self.domain,
                 dtype=self.dtype,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return Orthogonal(self.N, **d)
