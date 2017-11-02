from __future__ import division
import numpy as np
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as lasolve
import six
from copy import deepcopy
from numbers import Number
from .utilities import inheritdocstrings

__all__=['SparseMatrix', 'SpectralMatrix', 'extract_diagonal_matrix']

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

    def matvec(self, v, c, format='dia', axis=0):
        """Matrix vector product

        Returns c = dot(self, v)

        args:
            v    (input)         Numpy array of ndim>=1
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

        # Roll relevant axis to first
        if axis > 0:
            v = np.moveaxis(v, axis, 0)
            c = np.moveaxis(c, axis, 0)

        if format == 'python':
            for key, val in six.iteritems(self):
                if np.ndim(val) > 0: # broadcasting
                    val = val[(slice(None), ) + (np.newaxis,)*(v.ndim-1)]
                if key < 0:
                    c[-key:min(N, M-key)] += val*v[:min(M, N+key)]
                else:
                    c[:min(N, M-key)] += val*v[key:min(M, N+key)]

        else:
            if format not in ('csr', 'dia'): # Fallback on 'csr'. Should probably throw warning
                format = 'csr'
            diags = self.diags(format=format)
            P = int(np.prod(v.shape[1:]))
            c[:N] = diags.dot(v[:M].reshape(M, P)).reshape(c[:N].shape)

        if axis > 0:
            c = np.moveaxis(c, 0, axis)
            v = np.moveaxis(v, 0, axis)

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
        assert isinstance(y, Number)
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
        assert isinstance(y, Number)
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
        assert isinstance(y, Number)
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
        #assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in f:
                # Check if symmetric and make copy if necessary
                if -key in f:
                    if id(f[key]) == id(f[-key]):
                        f[-key] = deepcopy(f[key])
                f[key] = f[key] + val
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
        """Return copy of self.__sub__(y) <==> self-d"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in f:
                # Check if symmetric and make copy if necessary
                if -key in f:
                    if id(f[key]) == id(f[-key]):
                        f[-key] = deepcopy(f[key])
                f[key] = f[key] - val
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

    def get_key(self):
        return self.__hash__()

    def solve(self, b, u=None, axis=0):
        """Solve matrix system Au = b

        where A is the current matrix (self)

        args:
            b    (input/output)    Vector of right hand side on entry.
                                   Solution on exit unless u is provided.
            u    (output)          Optional output vector

        Vectors may be one- or multidimensional.

        """
        assert self.shape[0] == self.shape[1]
        assert self.shape[0] == b.shape[axis]

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Roll relevant axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        if b.ndim == 1:
            u[:] = spsolve(self.diags(), b)
        else:
            N = b.shape[0]
            P = np.prod(b.shape[1:])
            u[:] = spsolve(self.diags(), b.reshape((N, P))).reshape(u.shape)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)
        return u


@inheritdocstrings
class SpectralMatrix(SparseMatrix):
    """Base class for inner product matrices

    args:
        d                            Dictionary, where keys are the diagonal
                                     offsets and values the diagonals
        trial  (basis, derivative)   tuple, where basis is an instance of
                                     a class for one of the bases in
                                         - shenfun.legendre.bases
                                         - shenfun.chebyshev.bases
                                         - shenfun.fourier.bases
                                     derivative is an integer, and represents
                                     the number of times the trial function
                                     should be differentiated. Representing
                                     matrix column.
        test   (basis, derivative)   As trial, but representing matrix row.
        scale  float                 Scale matrix with this constant


    The matrices are assumed to be sparse diagonal. The matrices are
    inner products of trial and test functions from one of eight function
    spaces

    Chebyshev basis:

      Chebyshev basis of first kind

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

    Legendre basis:

      Regular Legendre

        L_k
        span(L_k, k=0,1,...,N)

      Dirichlet boundary conditions

        xi_k = L_k-L_{k+2}
        span(xi_k, k=0,1,...,N-2)

      Homogeneous Neumann boundary conditions:

        phi_k = L_k - k*(k+1)/(k+2)/(k+3)L_{k+2},
        span(phi_k) for k = 1, 2, ..., N-2

      Both homogeneous Dirichlet and Neumann:

        psi_k = L_k -2*(2*k+5)/(2*k+7)*L_{k+2} + (2*k+3)/(2*k+7)*L_{k+4},
        span(psi_k) for k = 0, 1, ..., N-4

    Fourier basis:

        F_k = exp(ikx)
        span(F_k, k=-N/2, -N/2+1, ..., N/2-1)

    Examples:

    Mass matrix for Chebyshev Dirichlet basis:

        (phi_k, phi_j)_w = int_{-1}^{1} phi_k(x) phi_j(x) w(x) dx

    Stiffness matrix for Chebyshev Dirichlet basis:

        (phi_k'', phi_j)_w = int_{-1}^{1} phi_k''(x) phi_j(x) w(x) dx

    etc.

    The matrices can be automatically created using, e.g., for the mass
    matrix of the Dirichlet space

      M = SpectralMatrix({}, (ShenDirichletBasis(18), 0), (ShenDirichletBasis(18), 0))

    where the first (ShenDirichletBasis(18), 0) represents the trial function and
    the second the test function. The stiffness matrix can be obtained as

      A = SpectralMatrix({}, (ShenDirichletBasis(18), 0), (ShenDirichletBasis(18), 2))

    where (ShenDirichletBasis(18), 2) signals that we use the second derivative
    of this trial function. The number (here 18) is the number of quadrature
    points used for the basis.

    The automatically created matrices may be overloaded with more exactly
    computed diagonals.

    Note that matrices with the Neumann basis are stored using index space
    k = 0, 1, ..., N-2, i.e., including the zero index for a nonzero average
    value.

    """
    def __init__(self, d, test, trial, scale=1.0):
        if isinstance(test[1], (int, np.integer)):
            k_test, k_trial = test[1], trial[1]
        elif isinstance(test[1], np.ndarray):
            assert len(test[1]) == 1
            k_test = test[1][(0,)*np.ndim(test[1])]
            k_trial = trial[1][(0,)*np.ndim(trial[1])]
        else:
            raise RuntimeError

        self.testfunction = (test[0], k_test)
        self.trialfunction = (trial[0], k_trial)

        self.scale = scale
        shape = self.spectral_shape()
        if d == {}:
            D = self.get_dense_matrix()[:shape[0], :shape[1]]
            d = extract_diagonal_matrix(D)
        SparseMatrix.__init__(self, d, shape)
        #if not round(scale-1.0, 8) == 0:
            #self *= scale

    def spectral_shape(self):
        """Return shape of matrix"""
        return (self.testfunction[0].spectral_shape(),
                self.trialfunction[0].spectral_shape())

    def get_dense_matrix(self):
        """Return dense matrix automatically computed from basis"""
        N = self.testfunction[0].N
        x, w = self.testfunction[0].points_and_weights(N)
        V = self.testfunction[0].vandermonde(x)
        test = self.testfunction[0].get_vandermonde_basis_derivative(V, self.testfunction[1])
        trial = self.trialfunction[0].get_vandermonde_basis_derivative(V, self.trialfunction[1])
        return np.dot(w*test.T, np.conj(trial))

    def test_sanity(self):
        """Sanity test for matrix.

        Test that automatically created matrix agrees with overloaded one

        """
        N, M = self.shape
        D = self.get_dense_matrix()[:N, :M]
        Dsp = extract_diagonal_matrix(D)
        Dsp *= self.scale
        for key, val in six.iteritems(self):
            assert np.allclose(val, Dsp[key])

    def matvec(self, v, c, format='csr', axis=0):
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        if self.testfunction[0].__class__.__name__ == 'ShenNeumannBasis':
            ss = [slice(None)]*len(v.shape)
            ss[axis] = 0
            c[ss] = 0
        return c

    def solve(self, b, u=None, axis=0):
        """Solve self u = b and return u

        The matrix self must be square

        args:
            u   (output)    Array
            b   (input)     Array

        """
        from shenfun import solve as default_solve
        u = default_solve(self, b, u, axis=axis)
        return u

    def __hash__(self):
        return hash(((self.testfunction[0].__class__, self.testfunction[1]),
                     (self.trialfunction[0].__class__, self.trialfunction[1])))

    def get_key(self):
        if self.__class__.__name__.startswith('_'):
            return self.__hash__()
        else:
            return self.__class__.__name__

    def __imul__(self, y):
        """self.__imul__(y) <==> self*=y"""
        assert isinstance(y, Number)
        self.scale *= y
        return self

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        assert isinstance(y, Number)
        f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                           self.trialfunction, self.scale*y)
        return f

    def __rmul__(self, y):
        """Returns copy of self.__rmul__(y) <==> y*self"""
        return self.__mul__(y)

    def __div__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        assert isinstance(y, Number)
        f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                           self.trialfunction, self.scale/y)
        return f

    def __truediv__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        return self.__div__(y)

    def __add__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        assert isinstance(d, dict)
        # Check is the same matrix
        if self.__hash__() == d.__hash__():
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale+d.scale)
        else:
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, 1.0)
            for key, val in six.iteritems(d):
                if key in f:
                    # Check if symmetric and make copy if necessary
                    if -key in f:
                        if id(f[key]) == id(f[-key]):
                            f[-key] = deepcopy(f[key])
                    f[key] = self.scale*f[key] + d.scale*val
                else:
                    f[key] = d.scale*val

        return f

    def __iadd__(self, d):
        """self.__iadd__(d) <==> self += d"""
        assert isinstance(d, dict)
        assert d.shape == self.shape
        if self.__hash__() == d.__hash__():
            self.scale += d.scale
        else:
            for key, val in six.iteritems(d):
                if key in self:
                    # Check if symmetric and make copy if necessary
                    if -key in self:
                        if id(self[key]) == id(self[-key]):
                            self[-key] = deepcopy(self[key])
                    self[key] += d.scale*val/self.scale
                else:
                    self[key] = d.scale*val/self.scale

        return self

    def __sub__(self, d):
        """Return copy of self.__sub__(y) <==> self-d"""
        assert isinstance(d, dict)
        # Check is the same matrix
        if self.__hash__() == d.__hash__():
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale-d.scale)
        else:
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, 1.0)
            for key, val in six.iteritems(d):
                if key in f:
                    # Check if symmetric and make copy if necessary
                    if -key in f:
                        if id(f[key]) == id(f[-key]):
                            f[-key] = deepcopy(f[key])
                    f[key] = self.scale*f[key] - d.scale*val
                else:
                    f[key] = -d.scale*val

        return f


    def __isub__(self, d):
        """self.__isub__(d) <==> self -= d"""
        assert isinstance(d, dict)
        assert d.shape == self.shape
        if self.__hash__() == d.__hash__():
            self.scale -= d.scale
        else:
            for key, val in six.iteritems(d):
                if key in self:
                    # Check if symmetric and make copy if necessary
                    if -key in self:
                        if id(self[key]) == id(self[-key]):
                            self[-key] = deepcopy(self[key])
                    self[key] -= d.scale*val/self.scale
                else:
                    self[key] = -d.scale*val/self.scale

        return self


def extract_diagonal_matrix(M, abstol=1e-8, reltol=1e-12):
    """Return SparseMatrix version of M

    args:
        M                    Dense matrix. Numpy array of ndim=2

    kwargs:
        abstol     (float)   Tolerance. Only diagonals with max(|d|) < tol
                             are kept in the returned SparseMatrix
        reltol     (float)   Relative tolerance. Only diagonals with
                             max(|d|)/max(|M|) > reltol are kept in
                             the returned SparseMatrix
    """
    d = {}
    relmax = abs(M).max()
    for i in range(M.shape[1]):
        u = M.diagonal(i).copy()
        if abs(u).max() > abstol and abs(u).max()/relmax > reltol :
            d[i] = u

    for i in range(1, M.shape[0]):
        l = M.diagonal(-i).copy()
        if abs(l).max() > abstol and abs(l).max()/relmax > reltol:
            d[-i] = l

    return SparseMatrix(d, M.shape)

