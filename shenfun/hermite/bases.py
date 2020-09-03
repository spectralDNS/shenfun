import functools
import sympy as sp
import numpy as np
from numpy.polynomial import hermite
from scipy.special import eval_hermite, factorial
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import


class Orthogonal(SpectralBase):
    r"""Base class for Hermite functions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points. Should be even for efficiency, but
            this is not required.
        quad : str, optional
            Type of quadrature

            - HG - Hermite-Gauss

        domain : 2-tuple of floats, optional
            The computational domain.
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
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    We are using Hermite functions and not the regular Hermite polynomials
    as basis functions. A Hermite function is defined as

    .. math::

        H_k = P_k \cdot \frac{1}{\pi^{0.25} \sqrt{2^n n!}} \exp(-x^2/2)

    where :math:`H_k` and :math:`P_k` are the Hermite function and Hermite
    polynomials of order k, respectively.

    """
    def __init__(self, N, quad="HG", bc=(0., 0.), dtype=np.float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        SpectralBase.__init__(self, N, quad=quad, domain=(-np.inf, np.inf), dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        self.plan(int(N*padding_factor), 0, dtype, {})

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

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates)

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

    def sympy_basis(self, i=0, x=sp.symbols('x')):
        return sp.hermite(i, x)*sp.exp(-x**2/2)*self.factor(i)

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

    def get_orthogonal(self):
        return self
