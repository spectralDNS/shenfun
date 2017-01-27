from __future__ import division
import numpy as np
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import six
from copy import deepcopy
from . import inheritdocstrings

class SparseMatrix(dict):
    """Base class for sparse matrices

    The data is stored as a dictionary, where keys and values are,
    respectively, the offsets and values of the diagonal.

    A tridiagonal matrix of shape N x N could be created as

    >>> d = {-1: 1,
              0: -2,
              1: 1}

    >>> SparseMatrix(d, (N, N))

    In case of variable values, store the entire diagonal
    For an N x N matrix use:

    >>> d = {-1: np.ones(N-1),
              0: -2*np.ones(N),
              1: np.ones(N-1)}

    >>> SparseMatrix(d, (N, N))

    """

    def __init__(self, d, shape):
        dict.__init__(self, d)
        self.shape = shape
        self._diags = None

    def matvec(self, v, c, format='dia'):
        """Matrix vector product

        Returns c = dot(self, v)

        args:
            v    (input)         Numpy array of ndim=1 or 3
            c    (output)        Numpy array of same ndim as v

        kwargs:
            format  ('csr',      Choice for computation
                     'dia',      format = 'csr' or 'dia' uses sparse matrices
                     'python',   from scipy.sparse and their built in matvec.
                     'self',     format = 'python' uses numpy and vectorization
                     'cython')   'self' and 'cython' are keywords reserved for
                                 methods overloaded in subclasses, and may not
                                 be implemented for all matrices.

        """
        assert v.shape == c.shape
        N, M = self.shape
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'python':
                for key, val in six.iteritems(self):
                    if key < 0:

                        for i in range(v.shape[1]):
                            for j in range(v.shape[2]):
                                c[-key:min(N, M-key), i, j] += val*v[:min(M, N+key), i, j]
                    else:
                        for i in range(v.shape[1]):
                            for j in range(v.shape[2]):
                                c[:min(N, M-key), i, j] += val*v[key:min(M, N+key), i, j]

            else:
                if format not in ('csr', 'dia'): # Fallback on 'csr'. Should probably throw warning
                    format = 'csr'
                diags = self.diags(format=format)
                for i in range(v.shape[1]):
                    for j in range(v.shape[2]):
                        c[:N, i, j] = diags.dot(v[:M, i, j])

        else:
            if format == 'python':
                for key, val in six.iteritems(self):
                    if key < 0:
                        c[-key:min(N, M-key)] += val*v[:min(M, N+key)]
                    else:
                        c[:min(N, M-key)] += val*v[key:min(M, N+key)]

            else:
                if format not in ('csr', 'dia'):
                    format = 'csr'
                diags = self.diags(format=format)
                c[:N] = diags.dot(v[:M])

        return c

    def diags(self, format='dia'):
        """Return a regular sparse matrix of specified format

        kwargs:
            format  ('dia', 'csr')

        """
        if self._diags is None:
            self._diags = sp_diags(list(self.values()), list(self.keys()),
                                   shape=self.shape, format=format)

        if self._diags.format != format:
            self._diags = sp_diags(list(self.values()), list(self.keys()),
                                   shape=self.shape, format=format)

        return self._diags

    def __imul__(self, y):
        """self.__imul__(y) <==> self*=y"""
        assert isinstance(y, (np.float, np.int))
        for key in self:
            # Check if symmetric
            if key < 0 and (-key) in self:
                if id(self[key]) == id(self[-key]):
                    continue
            self[key] *= y

        return self

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(y, (np.float, np.int))
        for key in f:
            # Check if symmetric
            if key < 0 and (-key) in f:
                if id(f[key]) == id(f[-key]):
                    continue
            f[key] *= y
        return f

    def __rmul__(self, y):
        """Returns copy of self.__rmul__(y) <==> y*self"""
        return self.__mul__(y)

    def __div__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(y, (np.float, np.int))
        for key in f:
            # Check if symmetric
            if key < 0 and (-key) in f:
                if id(f[key]) == id(f[-key]):
                    continue
            f[key] /= y
        return f

    def __truediv__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        return self.__div__(y)

    def __add__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in f:
                # Check if symmetric and make copy if necessary
                if -key in f:
                    if id(f[key]) == id(f[-key]):
                        f[-key] = deepcopy(f[key])
                f[key] += val
            else:
                f[key] = val

        return f

    def __iadd__(self, d):
        """self.__iadd__(d) <==> self += d"""
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in self:
                # Check if symmetric and make copy if necessary
                if -key in self:
                    if id(self[key]) == id(self[-key]):
                        self[-key] = deepcopy(self[key])
                self[key] += val
            else:
                self[key] = val

        return self

    def __sub__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in f:
                # Check if symmetric and make copy if necessary
                if -key in f:
                    if id(f[key]) == id(f[-key]):
                        f[-key] = deepcopy(f[key])
                f[key] -= val
            else:
                f[key] = -val

        return f

    def __isub__(self, d):
        """self.__isub__(d) <==> self -= d"""
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in self:
                # Check if symmetric and make copy if necessary
                if -key in self:
                    if id(self[key]) == id(self[-key]):
                        self[-key] = deepcopy(self[key])
                self[key] -= val
            else:
                self[key] = -val

        return self

    def solve(self, b, u=None):
        """Solve matrix system Au = b

        where A is the current matrix (self)

        args:
            b    (input/output)    Vector of right hand side on entry,
                                   solution on exit.

        Vectors may be one- or multidimensional. Solve along first axis.

        """
        assert self.shape[0] == self.shape[1]
        assert self.shape[0] == b.shape[0]
        u = np.zeros_like(b)
        if u is None:
            us = np.zeros_like(b)
        else:
            assert u.shape == b.shape
            us = u

        if b.ndim == 1:
            us[:] = spsolve(self.diags(), b)
        else:
            #for i in np.ndindex(b.shape[1:]):
                #u[(slice(None),)+i] = spsolve(self.diags(), b[(slice(None),)+i])

            N = b.shape[0]
            P = np.prod(b.shape[1:])
            us[:] = spsolve(self.diags(), b.reshape((N, P))).reshape(b.shape)

        if u is None:
            b[:] = us
            return b
        else:
            return u


@inheritdocstrings
class ShenMatrix(SparseMatrix):
    """Base class for Shen matrices

    args:
        d                            Dictionary, where keys are the diagonal
                                     offsets and values the diagonals
        N      integer               Length of main diagonal
        trial  (basis, derivative)   tuple, where basis is an instance of
                                     one of
                                         - chebyshev.ChebyshevBasis
                                         - chebyshev.ShenDirichletBasis
                                         - chebyshev.ShenBiharmonicBasis
                                         - chebyshev.ShenNeumannBasis
                                     derivative is an integer, and represents
                                     the number of times the trial function
                                     should be differentiated
        test   (basis, derivative)   As trial
        scale  float                 Scale matrix with this constant


    Shen matrices are assumed to be sparse diagonal. The matrices are
    scalar products of trial and test functions from one of six function
    spaces

    Chebyshev basis and space of first kind

        T_k,
        span(T_k) for k = 0, 1, ..., N

    For homogeneous Dirichlet boundary conditions:

        phi_k = T_k - T_{k+2},
        span(phi_k) for k = 0, 1, ..., N-2

    For homogeneous Neumann boundary conditions:

        phi_k = T_k - (k/(k+2))**2T_{k+2},
        span(phi_k) for k = 1, 2, ..., N-2

    For Biharmonic basis with both homogeneous Dirichlet
    and Neumann:

        psi_k = T_k - 2(k+2)/(k+3)*T_{k+2} + (k+1)/(k+3)*T_{k+4},
        span(psi_k) for k = 0, 1, ..., N-4

    The scalar product is computed as a weighted inner product with
    w=1/sqrt(1-x**2) the weights.

    For Legendre basis

        L_k
        span(L_k, k=0,1,...,N)

    For Legendre with homogeneous Dirichlet boundary conditions

        xi_k = L_k-L_{k+2}
        span(xi_k, k=0,1,...,N-2)

    Examples:

    Mass matrix for Chebyshev Dirichlet basis:

        (phi_k, phi_j)_w = int_{-1}^{1} phi_k(x) phi_j(x) w(x) dx

    Stiffness matrix for Chebyshev Dirichlet basis:

        (phi_k'', phi_j)_w = int_{-1}^{1} phi_k''(x) phi_j(x) w(x) dx

    etc.

    The matrices can be automatically created using, e.g., for the mass
    matrix of the Dirichlet space

      M = ShenMatrix({}, 16, (ShenDirichletBasis(), 0), (ShenDirichletBasis(), 0))

    where the first (ShenDirichletBasis, 0) represents the trial function and
    the second the test function. The stiffness matrix can be obtained as

      A = ShenMatrix({}, 16, (ShenDirichletBasis(), 2), (ShenDirichletBasis(), 0))

    where (ShenDirichletBasis, 2) signals that we use the second derivative
    of this trial function.

    The automatically created matrices may be overloaded with more exactly
    computed diagonals.

    Note that matrices with the Neumann basis are stored using index space
    k = 0, 1, ..., N-2, i.e., including the zero index for a nonzero average
    value.

    """
    def __init__(self, d, N, trial, test, scale=1.0):
        self.trialfunction, self.trial_derivative = trial
        self.testfunction, self.test_derivative = test
        self.N = N
        self.scale = scale
        shape = self.get_shape()
        if d == {}:
            D = self.get_dense_matrix()[:shape[0], :shape[1]]
            d = extract_diagonal_matrix(D)
        SparseMatrix.__init__(self, d, shape)
        if not round(scale-1.0, 8) == 0:
            self *= scale

    def get_shape(self):
        """Return shape of matrix"""
        return (self.testfunction.get_shape(self.N),
                self.trialfunction.get_shape(self.N))

    def get_ck(self, N, quad):
        ck = np.ones(N, int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2
        return ck

    def get_dense_matrix(self):
        """Return dense matrix automatically computed from basis"""
        N = self.N
        x, w = self.testfunction.points_and_weights(N, self.testfunction.quad)
        V = self.testfunction.vandermonde(x, N)
        test = self.testfunction.get_vandermonde_basis_derivative(V, self.test_derivative)
        trial = self.trialfunction.get_vandermonde_basis_derivative(V, self.trial_derivative)
        return np.dot(w*test.T, trial)

    def test(self):
        """Test for matrix.

        Test that automatically created matrix is the same as the one created

        """
        N, M = self.shape
        D = self.get_dense_matrix()[:N, :M]
        Dsp = extract_diagonal_matrix(D)
        Dsp *= self.scale
        for key, val in six.iteritems(self):
            assert np.allclose(val, Dsp[key])

    def solve(self, b, u=None):
        from .chebyshev.bases import ShenNeumannBasis
        assert self.shape[0] == self.shape[1]
        s = self.testfunction.slice(b.shape[0])
        bs = b[s]
        if u is None:
            us = np.zeros_like(b[s])
        else:
            assert u.shape == b.shape
            us = u[s]

        if isinstance(self.testfunction, ShenNeumannBasis):
            # Handle level by using Dirichlet for dof=0
            A = self.diags().toarray()
            A[0] = 0
            A[0,0] = 1
            b[0] = self.testfunction.mean
            if b.ndim == 1:
                us[:] = solve(A, bs)
            else:
                N = bs.shape[0]
                P = np.prod(bs.shape[1:])
                us[:] = solve(A, bs.reshape((N, P))).reshape(bs.shape)

        else:
            if b.ndim == 1:
                us[:] = spsolve(self.diags(), bs)
            else:
                #for i in np.ndindex(b.shape[1:]):
                    #u[(s,)+i] = spsolve(self.diags(), b[(s,)+i])
                N = bs.shape[0]
                P = np.prod(bs.shape[1:])
                us[:] = spsolve(self.diags(), bs.reshape((N, P))).reshape(bs.shape)

        if u is None:
            b[s] = us
            return b
        else:
            return u


def extract_diagonal_matrix(M, tol=1e-8):
    """Return matrix with essentially zero diagonals nulled out
    """
    d = {}
    for i in range(M.shape[1]):
        u = M.diagonal(i).copy()
        if abs(u).max() > tol:
            d[i] = u

    for i in range(1, M.shape[0]):
        l = M.diagonal(-i).copy()
        if abs(l).max() > tol:
            d[-i] = l

    return SparseMatrix(d, M.shape)
