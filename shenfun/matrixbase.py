r"""
This module contains classes for working with sparse matrices
"""
from __future__ import division
from copy import deepcopy
from numbers import Number
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve
import numpy as np
import six
from .utilities import inheritdocstrings

__all__ = ['SparseMatrix', 'SpectralMatrix', 'extract_diagonal_matrix', 'check_sanity', 'get_dense_matrix']

class SparseMatrix(dict):
    r"""Base class for sparse matrices

    The data is stored as a dictionary, where keys and values are, respectively,
    the offsets and values of the diagonals. In addition, each matrix is stored
    with a coefficient that is used as a scalar multiple of the matrix.

    Parameters
    ----------
        d : dict
            Dictionary, where keys are the diagonal offsets and values the
            diagonals
        shape : two-tuple of ints
        scale : float
                Scale matrix with this constant or array of constants

    Examples
    --------
    A tridiagonal matrix of shape N x N could be created as

    >>> from shenfun import SparseMatrix
    >>> import numpy as np
    >>> N = 4
    >>> d = {-1: 1, 0: -2, 1: 1}
    >>> SparseMatrix(d, (N, N))
    {-1: 1, 0: -2, 1: 1}

    In case of variable values, store the entire diagonal. For an N x N
    matrix use

    >>> d = {-1: np.ones(N-1),
    ...       0: -2*np.ones(N),
    ...       1: np.ones(N-1)}
    >>> SparseMatrix(d, (N, N))
    {-1: array([1., 1., 1.]), 0: array([-2., -2., -2., -2.]), 1: array([1., 1., 1.])}
    """
    # pylint: disable=redefined-builtin, missing-docstring

    def __init__(self, d, shape, scale=1.0):
        dict.__init__(self, d)
        self.shape = shape
        self._diags = None
        self.scale = scale

    def matvec(self, v, c, format='dia', axis=0):
        """Matrix vector product

        Returns c = dot(self, v)

        Parameters
        ----------
            v : array
                Numpy input array of ndim>=1
            c : array
                Numpy output array of same ndim as v
            format : str, optional
                     Choice for computation

                     - csr - Compressed sparse row format
                     - dia - Sparse matrix with DIAgonal storage
                     - python - Use numpy and vectorization
                     - self - To be implemented in subclass
                     - cython - Cython implementation that may be implemented in subclass
            axis : int, optional
                   The axis over which to take the matrix vector product

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

        c *= self.scale
        return c

    def diags(self, format='dia'):
        """Return a regular sparse matrix of specified format

        Parameters
        ----------
            format : str, optional
                     Choice of matrix type (see scipy.sparse.diags)

                     - dia - Sparse matrix with DIAgonal storage
                     - csr - Compressed sparse row

        Note
        ----
        This method does not return the matrix scaled by self.scale. Make sure
        to include the scale if the returned matrix is to be used in further
        calculations

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
        self.scale *= y
        return self
#        for key in self:
#            # Check if symmetric
#            if key < 0 and (-key) in self:
#                if id(self[key]) == id(self[-key]):
#                    continue
#            self[key] *= y
#
#        return self

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        if isinstance(y, Number):
            return SparseMatrix(deepcopy(dict(self)), self.shape,
                                scale=self.scale*y)
        elif isinstance(y, np.ndarray):
            c = np.zeros_like(y)
            c = self.matvec(y, c)
            return c

    def __rmul__(self, y):
        """Returns copy of self.__rmul__(y) <==> y*self"""
        return self.__mul__(y)

    def __div__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        if isinstance(y, Number):
            return SparseMatrix(deepcopy(dict(self)), self.shape,
                                scale=self.scale/y)
        elif isinstance(y, np.ndarray):
            b = np.zeros_like(y)
            b = self.solve(y, b)
            return b
        else:
            raise NotImplementedError

    def __truediv__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        return self.__div__(y)

    def __add__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        if self.__hash__() == d.__hash__():
            f = SparseMatrix(deepcopy(dict(self)), self.shape,
                             self.scale+d.scale)
        else:
            f = SparseMatrix(deepcopy(dict(self)), self.shape)
            assert isinstance(d, dict)
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
            f = SparseMatrix(deepcopy(dict(self)), self.shape,
                             self.scale-d.scale)
        else:
            f = SparseMatrix(deepcopy(dict(self)), self.shape, 1.0)
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

    def __neg__(self):
        self.scale *= -1
        return self

    def __hash__(self):
        return hash(frozenset(self))

    def get_key(self):
        return self.__hash__()

    def scale_array(self, c):
        if isinstance(self.scale, Number):
            if self.scale not in (1.0, 1):
                c *= self.scale
        else:
            c *= self.scale

    def solve(self, b, u=None, axis=0):
        """Solve matrix system Au = b

        where A is the current matrix (self)

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array, optional
                Output array
            axis : int, optional
                   The axis over which to solve for if b and u are multi-
                   dimensional

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
        u /= self.scale
        return u


@inheritdocstrings
class SpectralMatrix(SparseMatrix):
    r"""Base class for inner product matrices

    Parameters
    ----------
        d : dict
            Dictionary, where keys are the diagonal offsets and values the
            diagonals
        trial : 2-tuple of (basis, int)
                The basis is an instance of a class for one of the bases in

                - shenfun.legendre.bases
                - shenfun.chebyshev.bases
                - shenfun.fourier.bases

                The int represents the number of times the trial function
                should be differentiated. Representing matrix column.
        test : 2-tuple of (basis, int)
               As trial, but representing matrix row.
        scale : float
                Scale matrix with this constant or array of constants

    The matrices are assumed to be sparse diagonal. The matrices are
    inner products of trial and test functions from one of 9 function
    spaces

    Denoting the :math:`k`'th basis function as :math:`\phi_k` and basis
    as V we get the following:

    Chebyshev basis:

        Chebyshev basis of first kind

        .. math::

            \phi_k &= T_k \\
            V &= span\{\phi_k\}_{k=0}^{N}

        For homogeneous Dirichlet boundary conditions:

        .. math::

            \phi_k &= T_k - T_{k+2} \\
            V &= span\{\phi_k\}_{k=0}^{N-2}

        For homogeneous Neumann boundary conditions:

        .. math::

            \phi_k &= T_k - \left(\frac{k}{k+2}\right)^2T_{k+2} \\
            V &= span\{\phi_k\}_{k=1}^{N-2}

        For Biharmonic basis with both homogeneous Dirichlet and Neumann:

        .. math::

            \phi_k &= T_k - 2 \frac{k+2}{k+3} T_{k+2} + \frac{k+1}{k+3} T_{k+4} \\
            V &= span\{\phi_k\}_{k=0}^{N-4}

        The scalar product is computed as a weighted inner product with
        :math:`w=1/\sqrt{1-x^2}` the weights.

    Legendre basis:

        Regular Legendre

        .. math::

            \phi_k &= L_k \\
            V &= span\{\phi_k\}_{k=0}^{N}

        Dirichlet boundary conditions

        .. math::

            \phi_k &= L_k-L_{k+2} \\
            V &= span\{\phi_k\}_{k=0}^{N-2}

        Homogeneous Neumann boundary conditions:

        .. math::

            \phi_k &= L_k - \frac{k(k+1)}{(k+2)(k+3)}L_{k+2} \\
            V &= span\{\phi_k\}_{k=1}^{N-2}

        Both homogeneous Dirichlet and Neumann:

        .. math::

            \psi_k &= L_k - 2 \frac{2k+5}{2k+7} L_{k+2} + \frac{2k+3}{2k+7} L_{k+4} \\
            V &= span\{\phi_k\}_{k=0}^{N-4}

    Fourier basis:

        .. math::

            \phi_k &= exp(ikx) \\
            V &= span\{\phi_k\}_{k=-N/2}^{N/2-1}

    Examples
    --------

    Mass matrix for Chebyshev Dirichlet basis:

    .. math::

        (\phi_k, \phi_j)_w = \int_{-1}^{1} \phi_k(x) \phi_j(x) w(x) dx

    Stiffness matrix for Chebyshev Dirichlet basis:

    .. math::

        (\phi_k'', \phi_j)_w = \int_{-1}^{1} \phi_k''(x) \phi_j(x) w(x) dx

    The matrices can be automatically created using, e.g., for the mass
    matrix of the Dirichlet space::

        SD = ShenDirichletBasis
        N = 16
        M = SpectralMatrix({}, (SD(N), 0), (SD(N), 0))

    where the first (SD(N), 0) represents the test function and
    the second the trial function. The stiffness matrix can be obtained as::

        A = SpectralMatrix({}, (SD(N), 0), (SD(N), 2))

    where (SD(N), 2) signals that we use the second derivative of this trial
    function. The number N is the number of quadrature points used for the
    basis.

    The automatically created matrices may be overloaded with more exactly
    computed diagonals.

    Note that matrices with the Neumann basis are stored using index space
    :math:`k = 0, 1, ..., N-2`, i.e., including the zero index for a nonzero
    average value.

    """
    def __init__(self, d, test, trial, scale=1.0):
        assert isinstance(test[1], (int, np.integer))
        assert isinstance(trial[1], (int, np.integer))
        self.testfunction = test
        self.trialfunction = trial
        shape = (test[0].shape(), trial[0].shape())
        if d == {}:
            D = get_dense_matrix(test, trial)[:shape[0], :shape[1]]
            d = extract_diagonal_matrix(D)
        SparseMatrix.__init__(self, d, shape, scale)
        if shape[0] == shape[1]:
            if test[0].__class__.__name__ == 'ShenNeumannBasis':
                from shenfun.la import NeumannSolve
                self.solver = NeumannSolve(self, test[0])
            else:
                from shenfun.la import Solve
                self.solver = Solve(self, test[0])

    def matvec(self, v, c, format='csr', axis=0):
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        if self.testfunction[0].__class__.__name__ == 'ShenNeumannBasis':
            ss = [slice(None)]*len(v.shape)
            ss[axis] = 0
            c[tuple(ss)] = 0
        return c

    def solve(self, b, u=None, axis=0):
        """Solve matrix system Au = b

        where A is the current matrix (self)

        Parameters
        ----------
            b : array
                Array of right hand side on entry and solution on exit unless
                u is provided.
            u : array, optional
                Output array
            axis : int, optional
                   The axis over which to solve for if b and u are multidimensional

        Vectors may be one- or multidimensional.
        """
        u = self.solver(b, u=u, axis=axis)
        return u

    def __hash__(self):
        return hash(((self.testfunction[0].__class__, self.testfunction[1]),
                     (self.trialfunction[0].__class__, self.trialfunction[1])))

    def get_key(self):
        if self.__class__.__name__.startswith('_'):
            return self.__hash__()
        return self.__class__.__name__

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        if isinstance(y, Number):
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale*y)
        elif isinstance(y, np.ndarray):
            f = SparseMatrix.__mul__(self, y)

        return f

    def __div__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        if isinstance(y, Number):
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale/y)
        elif isinstance(y, np.ndarray):
            f = SparseMatrix.__div__(self, y)

        return f

    def __add__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        assert isinstance(d, dict)
        # Check is the same matrix
        if self.__hash__() == d.__hash__():
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale+d.scale)
        else:
            f = SparseMatrix.__add__(self, d)
        return f

    def __sub__(self, d):
        """Return copy of self.__sub__(y) <==> self-d"""
        assert isinstance(d, dict)
        # Check is the same matrix
        if self.__hash__() == d.__hash__():
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale-d.scale)
        else:
            f = SparseMatrix.__sub__(self, d)
        return f


def check_sanity(A, test, trial):
    """Sanity check for matrix.

    Test that automatically created matrix agrees with overloaded one

    Parameters
    ----------
        A : matrix
        test : 2-tuple of (basis, int)
                The basis is an instance of a class for one of the bases in

                - shenfun.legendre.bases
                - shenfun.chebyshev.bases
                - shenfun.fourier.bases

                The int represents the number of times the test function
                should be differentiated. Representing matrix row.
        trial : 2-tuple of (basis, int)
                As test, but representing matrix column.
    """
    N, M = A.shape
    D = get_dense_matrix(test, trial)[:N, :M]
    Dsp = extract_diagonal_matrix(D)
    Dsp *= A.scale
    for key, val in six.iteritems(A):
        assert np.allclose(val, Dsp[key])


def get_dense_matrix(test, trial):
    """Return dense matrix automatically computed from basis

    Parameters
    ----------
        test : 2-tuple of (basis, int)
                The basis is an instance of a class for one of the bases in

                - shenfun.legendre.bases
                - shenfun.chebyshev.bases
                - shenfun.fourier.bases

                The int represents the number of times the test function
                should be differentiated. Representing matrix row.
        trial : 2-tuple of (basis, int)
                As test, but representing matrix column.
    """
    N = test[0].N
    _, w = test[0].points_and_weights(N)
    v = test[0].evaluate_basis_derivative_all(k=test[1])
    u = trial[0].evaluate_basis_derivative_all(k=trial[1])
    return np.dot(w*v.T, np.conj(u))


def extract_diagonal_matrix(M, abstol=1e-8, reltol=1e-12):
    """Return SparseMatrix version of M

    Parameters
    ----------
        M : Numpy array of ndim=2
        abstol : float
                 Tolerance. Only diagonals with max(:math:`|d|`) < tol are
                 kept in the returned SparseMatrix, where :math:`d` is the
                 diagonal
        reltol : float
                 Relative tolerance. Only diagonals with
                 max(:math:`|d|`)/max(:math:`|M|`) > reltol are kept in the
                 returned SparseMatrix

    """
    d = {}
    relmax = abs(M).max()
    for i in range(M.shape[1]):
        u = M.diagonal(i).copy()
        if abs(u).max() > abstol and abs(u).max()/relmax > reltol:
            d[i] = u

    for i in range(1, M.shape[0]):
        l = M.diagonal(-i).copy()
        if abs(l).max() > abstol and abs(l).max()/relmax > reltol:
            d[-i] = l

    return SparseMatrix(d, M.shape)
