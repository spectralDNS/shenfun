r"""
This module contains classes for working with sparse matrices.

"""
from __future__ import division
import functools
from copy import copy, deepcopy
from collections.abc import Mapping, MutableMapping
from numbers import Number
import numpy as np
import sympy as sp
from scipy.sparse import bmat, spmatrix, dia_matrix, csr_matrix, kron, \
    diags as sp_diags
from scipy.integrate import quad
from mpi4py import MPI
from shenfun.config import config
from .utilities import integrate_sympy

__all__ = ['SparseMatrix', 'SpectralMatrix', 'extract_diagonal_matrix',
           'extract_bc_matrices', 'check_sanity', 'assemble_sympy',
           'TPMatrix', 'BlockMatrix', 'BlockMatrices', 'Identity',
           'get_simplified_tpmatrices', 'ScipyMatrix', 'SpectralMatDict']

comm = MPI.COMM_WORLD

class SparseMatrix(MutableMapping):
    r"""Base class for sparse matrices.

    The data is stored as a dictionary, where keys and values are, respectively,
    the offsets and values of the diagonals. In addition, each matrix is stored
    with a coefficient that is used as a scalar multiple of the matrix.

    Parameters
    ----------
    d : dict
        Dictionary, where keys are the diagonal offsets and values the
        diagonals
    shape : two-tuple of ints
    scale : number, optional
        Scale matrix with this number

    Note
    ----
    The matrix format and storage is similar to Scipy's `dia_matrix`. The format is
    chosen because spectral matrices often are computed by hand and presented
    in the literature as banded matrices.
    Note that a SparseMatrix can easily be transformed to any of Scipy's formats
    using the `diags` method. However, Scipy's matrices are not implemented to
    act along different axes of multidimensional arrays, which is required
    for tensor product matrices, see :class:`.TPMatrix`. Hence the need for
    this SparseMatrix class.

    Examples
    --------
    A tridiagonal matrix of shape N x N could be created as

    >>> from shenfun import SparseMatrix
    >>> import numpy as np
    >>> N = 4
    >>> d = {-1: 1, 0: -2, 1: 1}
    >>> S = SparseMatrix(d, (N, N))
    >>> dict(S)
    {-1: 1, 0: -2, 1: 1}

    In case of variable values, store the entire diagonal. For an N x N
    matrix use

    >>> d = {-1: np.ones(N-1),
    ...       0: -2*np.ones(N),
    ...       1: np.ones(N-1)}
    >>> S = SparseMatrix(d, (N, N))
    >>> dict(S)
    {-1: array([1., 1., 1.]), 0: array([-2., -2., -2., -2.]), 1: array([1., 1., 1.])}

    """
    # pylint: disable=redefined-builtin, missing-docstring
    def __init__(self, d, shape, scale=1):
        # sort d before storing
        sorted_dict = sorted(d.items())
        self._storage = {si[0]: si[1] for si in sorted_dict}
        self.shape = shape
        self._diags = dia_matrix((1, 1))
        self.scale = scale
        self._matvec_methods = []
        self.solver = None

    def matvec(self, v, c, format=None, axis=0):
        """Matrix vector product

        Returns c = dot(self, v)

        Parameters
        ----------
        v : array
            Numpy input array of ndim>=1
        c : array
            Numpy output array of same shape as v
        format : str, optional
             Choice for computation

             - csr - Compressed sparse row format
             - dia - Sparse matrix with DIAgonal storage
             - python - Use numpy and vectorization
             - self - To be implemented in subclass
             - cython - Cython implementation that may be implemented in subclass
             - numba - Numba implementation that may be implemented in subclass

             Using ``config['matrix']['sparse']['matvec']`` setting if format is None

        axis : int, optional
            The axis over which to take the matrix vector product

        """
        format = config['matrix']['sparse']['matvec'] if format is None else format
        N, M = self.shape
        c.fill(0)

        # Roll relevant axis to first
        if axis > 0:
            v = np.moveaxis(v, axis, 0)
            c = np.moveaxis(c, axis, 0)

        if format == 'python':
            for key, val in self.items():
                if np.ndim(val) > 0: # broadcasting
                    val = val[(slice(None), ) + (np.newaxis,)*(v.ndim-1)]
                if key < 0:
                    c[-key:min(N, M-key)] += val*v[:min(M, N+key)]
                else:
                    c[:min(N, M-key)] += val*v[key:min(M, N+key)]
            c *= self.scale

        else:
            diags = self.diags(format=format)
            P = int(np.prod(v.shape[1:]))
            y = diags.dot(v[:M].reshape(M, P)).squeeze()
            d = tuple([slice(0, m) for m in y.shape])
            c[d] = y.reshape(c[d].shape)

        if axis > 0:
            c = np.moveaxis(c, 0, axis)
            v = np.moveaxis(v, 0, axis)

        return c

    def diags(self, format=None, scaled=True):
        """Return a regular sparse matrix of specified format

        Parameters
        ----------
        format : str, optional
            Choice of matrix type (see scipy.sparse.diags)

            - dia - Sparse matrix with DIAgonal storage
            - csr - Compressed sparse row
            - csc - Compressed sparse column

            Using ``config['matrix']['sparse']['diags']`` setting if format is None

        scaled : bool, optional
            Return matrix scaled by the constant self.scale if True

        Note
        ----
        This method returns the matrix scaled by self.scale if keyword scaled
        is True.

        """
        format = config['matrix']['sparse']['diags'] if format is None else format
        self.sort()
        self._diags = sp_diags(list(self.values()),
                               list(self.keys()),
                               shape=self.shape, format=format)
        scale = self.scale
        if isinstance(scale, np.ndarray):
            scale = np.atleast_1d(scale).item()
        return self._diags*scale if scaled else self._diags

    def sort(self):
        self._storage = {si[0]: si[1] for si in sorted(self.items())}

    def __getitem__(self, key):
        v = self._storage[key]
        if hasattr(v, '__call__'):
            return v(key)
        return v

    def __delitem__(self, key):
        del self._storage[key]

    def __setitem__(self, key, val):
        self._storage[key] = val

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def __eq__(self, a):
        if self.shape != a.shape:
            return False
        if not self.same_keys(a):
            return False
        d0 = self.diags('csr', False).data
        a0 = a.diags('csr', False).data
        if d0.shape[0] != a0.shape[0]:
            return False
        if not np.linalg.norm(d0-a0) < 1e-8:
            return False
        return True

    def __neq__(self, a):
        return not self.__eq__(a)

    def __imul__(self, y):
        """self.__imul__(y) <==> self*=y"""
        assert isinstance(y, Number)
        self.scale *= y
        return self

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        if isinstance(y, Number):
            c = self.copy()
            c.scale *= y
            return c
        elif isinstance(y, np.ndarray):
            c = np.empty_like(y)
            c = self.matvec(y, c)
            return c
        elif isinstance(y, SparseMatrix):
            return self.diags('csc')*y.diags('csc')
        raise RuntimeError

    def __rmul__(self, y):
        """Returns copy of self.__rmul__(y) <==> y*self"""
        return self.__mul__(y)

    def __div__(self, y):
        """Returns elementwise division if `y` is a Number, or a linear algebra
        solve if `y` is an array.

        Parameters
        ----------
        y : Number or array

        """
        if isinstance(y, Number):
            assert abs(y) > 1e-8
            c = self.copy()
            c.scale /= y
            return c
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

        if abs(self.scale) < 1e-15 and abs(d.scale) < 1e-15:
            f = SparseMatrix({0: 0}, self.shape)

        elif abs(self.scale) < 1e-15:
            f = SparseMatrix(deepcopy(dict(d)), d.shape, d.scale)

        elif abs(d.scale) < 1e-15:
            f = self.copy()

        else:
            assert isinstance(d, Mapping)
            f = SparseMatrix(deepcopy(dict(self)), self.shape, self.scale)
            f.incorporate_scale()
            d.incorporate_scale()
            for key, val in d.items():
                if key in f:
                    f[key] = f[key] + val
                else:
                    f[key] = val
        return f

    def __iadd__(self, d):
        """self.__iadd__(d) <==> self += d"""
        assert isinstance(d, Mapping)
        assert d.shape == self.shape

        if abs(d.scale) < 1e-16:
            return self

        elif abs(self.scale) < 1e-16:
            self.clear()
            for key, val in d.items():
                self[key] = val
            self.scale = d.scale
            return self

        self.incorporate_scale()
        d.incorporate_scale()
        for key, val in d.items():
            if key in self:
                self[key] = self[key] + val
            else:
                self[key] = val
        return self

    def __sub__(self, d):
        """Return copy of self.__sub__(d) <==> self-d"""
        assert isinstance(d, Mapping)

        if abs(self.scale) < 1e-15 and abs(d.scale) < 1e-15:
            f = SparseMatrix({0: 0}, self.shape)

        elif abs(self.scale) < 1e-15:
            f = SparseMatrix(deepcopy(dict(d)), d.shape, -d.scale)

        elif abs(d.scale) < 1e-15:
            f = self.copy()

        else:
            f = SparseMatrix(deepcopy(dict(self)), self.shape, self.scale)
            f.incorporate_scale()
            d.incorporate_scale()
            for key, val in d.items():
                if key in f:
                    f[key] = f[key] - val
                else:
                    f[key] = -val

        return f

    def __isub__(self, d):
        """self.__isub__(d) <==> self -= d"""
        assert isinstance(d, Mapping)
        assert d.shape == self.shape

        if abs(d.scale) < 1e-16:
            return self

        elif abs(self.scale) < 1e-16:
            self.clear()
            for key, val in d.items():
                self[key] = val
            self.scale = -d.scale
            return self

        self.incorporate_scale()
        d.incorporate_scale()
        for key, val in d.items():
            if key in self:
                self[key] = self[key] - val
            else:
                self[key] = -val
        return self

    def copy(self):
        """Return SparseMatrix deep copy of self"""
        return self.__deepcopy__()

    def __copy__(self):
        if self.__class__.__name__ == 'Identity':
            return self
        return SparseMatrix(copy(dict(self)), self.shape, self.scale)

    def __deepcopy__(self, memo=None, _nil=[]):
        if self.__class__.__name__ == 'Identity':
            return Identity(self.shape, self.scale)
        return SparseMatrix(deepcopy(dict(self)), self.shape, self.scale)

    def __neg__(self):
        """self.__neg__() <==> -self"""
        A = self.copy()
        A.scale = self.scale*-1
        return A

    def __hash__(self):
        return hash(frozenset(self))

    def get_key(self):
        return self.__hash__()

    def same_keys(self, a):
        return self.__hash__() == a.__hash__()

    def scale_array(self, c, sc):
        assert isinstance(sc, Number)
        if abs(sc-1) > 1e-8:
            c *= sc

    def incorporate_scale(self):
        """Modifies matrix such that self.scale = 1"""
        if abs(self.scale-1) < 1e-8:
            return
        if hasattr(self, '_keyscale'):
            self._keyscale *= self.scale
        else:
            for key, val in self.items():
                self[key] = val*self.scale
        self.scale = 1

    def sorted_keys(self):
        return np.sort(np.array(list(self.keys())))

    def solve(self, b, u=None, axis=0, constraints=()):
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
        constraints : tuple of 2-tuples
            The 2-tuples represent (row, val)
            The constraint indents the matrix row and sets b[row] = val

        Note
        ----
        Vectors may be one- or multidimensional.

        """
        if self.solver is None:
            self.solver = self.get_solver()(self)
        u = self.solver(b, u=u, axis=axis, constraints=constraints)
        return u

    def get_solver(self):
        """Return appropriate solver for self

        Note
        ----
        Fall back on generic Solve, which is using Scipy sparse
        matrices with splu/spsolve. This is still pretty fast.
        """
        from .la import (Solve, TDMA, TDMA_O, FDMA, TwoDMA, ThreeDMA, PDMA,
            DiagMA, HeptaDMA)
        if self.scale == 0:
            return Solve
            
        if len(self) == 1:
            if list(self.keys())[0] == 0:
                return DiagMA
        elif len(self) == 2:
            if np.all(self.sorted_keys() == (0, 2)):
                return TwoDMA
        elif len(self) == 3:
            if np.all(self.sorted_keys() == (-2, 0, 2)):
                return TDMA
            elif np.all(self.sorted_keys() == (-1, 0, 1)) and self.issymmetric:
                return TDMA_O
            elif np.all(self.sorted_keys() == (0, 2, 4)):
                return ThreeDMA
        elif len(self) == 4:
            if np.all(self.sorted_keys() == (-2, 0, 2, 4)):
                return FDMA
        elif len(self) == 5:
            if np.all(self.sorted_keys() == (-4, -2, 0, 2, 4)):
                return PDMA
        elif len(self) == 7:
            if np.all(self.sorted_keys() == (-4, -2, 0, 2, 4, 6, 8)):
                return HeptaDMA
        return Solve

    def isdiagonal(self):
        if len(self) == 1:
            if 0 in self:
                return True
        return False

    def isidentity(self):
        if not len(self) == 1:
            return False
        if 0 not in self:
            return False
        d = self[0]
        if np.all(d == 1):
            return True
        return False

    @property
    def issymmetric(self):
        #M = self.diags()
        #return (abs(M-M.T) > 1e-8).nnz == 0 # too expensive
        if np.sum(np.array(list(self.keys()))) != 0:
            return False
        for key, val in self.items():
            if key <= 0:
                continue
            if not np.all(abs(val-self[-key]) < 1e-16):
                return False
        return True

    def simplify_diagonal_matrices(self):
        if self.isdiagonal():
            self.scale = self.scale*self[0]
            self[0] = 1

    def clean_diagonals(self, reltol=1e-8):
        """Eliminate essentially zerovalued diagonals

        Parameters
        ----------
        reltol : number
            Relative tolerance
        """
        a = self * np.ones(self.shape[1])
        relmax = abs(a).max() / self.shape[1]
        if relmax == 0:
            relmax = 1
        list_keys = []
        for key, val in self.items():
            if abs(np.linalg.norm(val))/relmax < reltol:
                list_keys.append(key)
        for key in list_keys:
            del self[key]
        return self

    def is_bc_matrix(self):
        return False


class SpectralMatrix(SparseMatrix):
    r"""Base class for inner product matrices.

    Parameters
    ----------
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
        - :mod:`.chebyshevu.bases`
        - :mod:`.ultraspherical.bases`
        - :mod:`.fourier.bases`
        - :mod:`.laguerre.bases`
        - :mod:`.hermite.bases`
        - :mod:`.jacobi.bases`

        The int represents the number of times the test function
        should be differentiated. Representing matrix column.
    trial : 2-tuple of (basis, int)
        As test, but representing matrix column.
    scale : number, optional
        Scale matrix with this number
    measure : number or Sympy expression, optional
        A function of the reference coordinate.
    assemble : None or str, optional
        Determines how to perform the integration,

        - 'quadrature' (default)
        - 'exact'
        - 'adaptive'

        Exact and adaptive should result in the same matrix. Exact computes the
        integral using `Sympy integrate <https://docs.sympy.org/latest/modules/integrals/integrals.html>`_,
        whereas adaptive makes use of adaptive quadrature through `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quadrature.html>`_.
    kind : None or str, optional
        Alternative kinds of methods.

        - 'implemented' - Hardcoded implementations
        - 'stencil' - Use orthogonal bases and stencil-matrices
        - 'vandermonde' - Use Vandermonde matrix

        The default is to first try to look for implemented kind, and if that
        fails try first 'stencil' and then finally fall back on vandermonde.
        Vandermonde creates a dense matrix of size NxN, so it should be avoided
        (e.g., by implementing the matrix) for large N.
    fixed_resolution : None or str, optional
        A fixed number of quadrature points used to compute the matrix.
        If 'fixed_resolution' is set, then assemble is set to 'quadrature' and
        kind is set to 'vandermonde'.

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

    >>> from shenfun import FunctionSpace, SpectralMatrix
    >>> SD = FunctionSpace(16, 'C', bc=(0, 0))
    >>> M = SpectralMatrix((SD, 0), (SD, 0))

    where the first (SD, 0) represents the test function and
    the second the trial function. The stiffness matrix can be obtained as::

    >>> A = SpectralMatrix((SD, 0), (SD, 2))

    where (SD, 2) signals that we use the second derivative of this trial
    function. A more natural way to do the same thing is to

    >>> from shenfun import TrialFunction, TestFunction, inner, Dx
    >>> u = TrialFunction(SD)
    >>> v = TestFunction(SD)
    >>> A = inner(v, Dx(u, 0, 2))

    where :func:`Dx` is a partial (or ordinary) derivative.

    """
    def __init__(self, test, trial, scale=1.0, measure=1, assemble=None,
                 kind=None, fixed_resolution=None):
        assert isinstance(test[1], (int, np.integer))
        assert isinstance(trial[1], (int, np.integer))
        self.testfunction = test
        self.trialfunction = trial
        self.measure = measure
        shape = (test[0].dim(), trial[0].dim())
        if isinstance(measure, Number):
            scale *= measure
            self.measure = measure = 1

        if assemble is None or fixed_resolution is not None:
            assemble = 'quadrature'
        d = {}
        _assembly_method = assemble
        if assemble == 'exact':
            d = self.assemble(assemble) # Look for implemented exact matrix
            if d is None:
                d = _get_matrix(test, trial, measure, assemble=assemble)
        elif assemble == 'adaptive':
            d = _get_matrix(test, trial, measure, assemble=assemble)
        else:
            if fixed_resolution is not None:
                kind = 'vandermonde'
            if kind is None:
                # If nothing is specified then
                # 1. Check for specific implementation
                # 2. Try to use stencil-method. This may fail because it does not
                #    cover all options, so
                # 3. Fall back on Vandermonde quadrature in case the above fails.
                d = self.assemble(assemble)
                if d is not None:
                    _assembly_method += '_implemented'
                if d is None:
                    if test[0].family() == 'fourier':
                        kind = 'vandermonde'
                    else:
                        if test[0].family() != trial[0].family():
                            kind = 'vandermonde'
                        elif test[0].short_name() in ('P1', 'P2', 'P3', 'P4'):
                            try:
                                d = assemble_phi(test, trial, measure)
                                _assembly_method += '_phi'
                            except AssertionError:
                                kind = 'vandermonde'
                        else:
                            if test[0].is_jacobi and sp.sympify(measure).is_polynomial() and not (test[0].is_orthogonal and trial[0].is_orthogonal):
                                d = assemble_stencil(test, trial, measure)
                                _assembly_method += '_stencil'
                            else:
                                kind = 'vandermonde'

            if kind is not None:
                # Specified method of assembly, mainly for testing
                if kind == 'implemented':
                    d = self.assemble(assemble)
                    _assembly_method += '_implemented'
                elif kind == 'stencil':
                    assert sp.sympify(measure).is_polynomial(), 'Cannot use `stencil` with non-polynomial coefficients'
                    if test[0].short_name() in ('P1', 'P2', 'P3', 'P4') and test[0].N != trial[0].N:
                        d = assemble_phi(test, trial, measure)
                        _assembly_method += '_phi'
                    else:
                        d = assemble_stencil(test, trial, measure)
                        _assembly_method += '_stencil'
                elif kind == 'vandermonde':
                    d = _get_matrix(test, trial, measure, assemble='quadrature', fixed_resolution=fixed_resolution)
                    _assembly_method += '_vandermonde'
        if test[0].domain_factor() != 1:
            scale *= float(test[0].domain_factor())**(test[1]+trial[1]-1)
        SparseMatrix.__init__(self, d, shape, scale)
        self._assembly_method = _assembly_method
        self.incorporate_scale()

    def assemble(self, method):
        r"""Return diagonals of :class:`.SpectralMatrix`

        Parameters
        ----------
        method : str
            Type of integration

            - 'exact'
            - 'quadrature'

        Note
        ----
        Subclass :class:`.SpectralMatrix` and overload this method in order
        to provide a fast and accurate implementation of the matrix representing
        an inner product. See the `matrix` modules in either one of

        - :mod:`.legendre.matrix`
        - :mod:`.chebyshev.matrix`
        - :mod:`.chebyshevu.matrix`
        - :mod:`.ultraspherical.matrix`
        - :mod:`.fourier.matrix`
        - :mod:`.laguerre.matrix`
        - :mod:`.hermite.matrix`
        - :mod:`.jacobi.matrix`

        Example
        -------
        The mass matrix for Chebyshev polynomials is

        .. math::

            (T_j, T_i)_{\omega} = \frac{c_i \pi}{2}\delta_{ij},

        where :math:`c_0=2` and :math:`c_i=1` for integer :math:`i>0`. We can
        implement this as

        >>> from shenfun import SpectralMatrix
        >>> class Bmat(SpectralMatrix):
        ...     def assemble(self, method):
        ...         test, trial = self.testfunction, self.trialfunction
        ...         ci = np.ones(test[0].N)
        ...         ci[0] = 2
        ...         if test[0].quad == 'GL' and method != 'exact':
        ...             # Gauss-Lobatto quadrature inexact at highest polynomial order
        ...             ci[-1] = 2
        ...         return {0: ci*np.pi/2}

        Here `{0: ci*np.pi/2}` is the 0'th diagonal of the matrix.
        Note that `test` and `trial` are two-tuples of `(instance of :class:`.SpectralBase`, number)`,
        where the number represents the number of derivatives. For the mass matrix
        the number will be 0. Also note that the length of the diagonal must be
        correct.

        """
        return None

    def matvec(self, v, c, format=None, axis=0):
        u = self.trialfunction[0]
        ss = [slice(None)]*len(v.shape)
        ss[axis] = u.slice()
        c = super(SpectralMatrix, self).matvec(v[tuple(ss)], c, format=format, axis=axis)
        return c

    @property
    def tensorproductspace(self):
        """Return the :class:`.TensorProductSpace` this matrix has been
        computed for"""
        return self.testfunction[0].tensorproductspace

    @property
    def axis(self):
        """Return the axis of the :class:`.TensorProductSpace` this matrix is
        created for"""
        return self.testfunction[0].axis

    def __hash__(self):
        return hash(((self.testfunction[0].__class__, self.testfunction[1]),
                     (self.trialfunction[0].__class__, self.trialfunction[1])))

    def get_key(self):
        if self.__class__.__name__.endswith('mat'):
            return  self.__class__.__name__
        return self.__hash__()

    def __eq__(self, a):
        if isinstance(a, Number):
            return False
        if not isinstance(a, SparseMatrix):
            return False
        if self.shape != a.shape:
            return False
        if self.get_key() != a.get_key():
            return False
        d0 = self.diags('csr', False).data
        a0 = a.diags('csr', False).data
        if d0.shape[0] != a0.shape[0]:
            return False
        if not np.linalg.norm(d0-a0) < 1e-8:
            return False
        return True

    def is_bc_matrix(self):
        return self.trialfunction[0].boundary_condition() == 'Apply'

class Identity(SparseMatrix):
    """The identity matrix in :class:`.SparseMatrix` form

    Parameters
    ----------
    shape : 2-tuple of ints
        The shape of the matrix
    scale : number, optional
        Scalar multiple of the matrix, defaults to unity

    """
    def __init__(self, shape, scale=1):
        SparseMatrix.__init__(self, {0: 1}, shape, scale)
        self.measure = 1

    def solve(self, b, u=None, axis=0, constraints=()):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b
        u *= (1/self.scale)
        return u

class ScipyMatrix(csr_matrix):

    def __init__(self, mats):
        assert isinstance(mats, (SparseMatrix, list))
        self.bc_mats = []
        if isinstance(mats, list):
            bc_mats = extract_bc_matrices([mats])
            mats = sum(mats[1:], mats[0])
            self.bc_mats = bc_mats
        csr_matrix.__init__(self, mats.diags('csr'))

    def matvec(self, v, c, axis=0):
        """Matrix vector product

        Returns c = dot(self, v)

        Parameters
        ----------
        v : array
            Numpy input array of ndim>=1
        c : array
            Numpy output array of same shape as v
        axis : int, optional
            The axis over which to take the matrix vector product

        """
        M = self.shape[1]
        c.fill(0)

        # Roll relevant axis to first
        if axis > 0:
            v = np.moveaxis(v, axis, 0)
            c = np.moveaxis(c, axis, 0)

        P = int(np.prod(v.shape[1:]))
        y = self.dot(v[:M].reshape(M, P)).squeeze()
        d = tuple([slice(0, m) for m in y.shape])
        c[d] = y.reshape(c[d].shape)

        if self.bc_mats:
            w0 = np.zeros_like(c)
            for bc_mat in self.bc_mats:
                c += bc_mat.matvec(v, w0, axis=0)

        if axis > 0:
            c = np.moveaxis(c, 0, axis)
            v = np.moveaxis(v, 0, axis)

        return c


def BlockMatrices(tpmats):
    """Return two instances of the :class:`.BlockMatrix` class.

    Parameters
    ----------
    tpmats : sequence of :class:`.TPMatrix`'es or single :class:`.BlockMatrix`
        There can be both boundary matrices from inhomogeneous Dirichlet
        or Neumann conditions, as well as regular matrices.

    Note
    ----
    Use :class:`.BlockMatrix` directly if you do not have any inhomogeneous
    boundary conditions.
    """
    if isinstance(tpmats, BlockMatrix):
        tpmats = tpmats.get_mats()
    bc_mats = extract_bc_matrices([tpmats])
    assert len(bc_mats) > 0, 'No boundary matrices - use BlockMatrix'
    return BlockMatrix(tpmats), BlockMatrix(bc_mats)

class BlockMatrix:
    r"""A class for block matrices

    Parameters
    ----------
        tpmats : sequence of :class:`.TPMatrix` or :class:`.SparseMatrix`
            The individual blocks for the matrix

    Note
    ----
    The tensor product matrices may be either boundary
    matrices, regular matrices, or a mixture of both.

    Example
    -------
    Stokes equations, periodic in x and y-directions

    .. math::

        -\nabla^2 u - \nabla p &= 0 \\
        \nabla \cdot u &= 0 \\
        u(x, y, z=\pm 1) &= 0

    We use for the z-direction a Dirichlet basis (SD) and a regular basis with
    no boundary conditions (ST). This is combined with Fourier in the x- and
    y-directions (K0, K1), such that we get two TensorProductSpaces (TD, TT)
    that are tensor products of these bases

    .. math::

        TD &= K0 \otimes K1 \otimes SD \\
        TT &= K0 \otimes K1 \otimes ST

    We choose trialfunctions :math:`u \in [TD]^3` and :math:`p \in TT`, and then
    solve the weak problem

    .. math::

        \left( \nabla v, \nabla u\right) + \left(\nabla \cdot v, p \right) = 0\\
        \left( q, \nabla \cdot u\right) = 0

    for all :math:`v \in [TD]^3` and :math:`q \in TT`.

    To solve the problem we need to assemble a block matrix

    .. math::

        \begin{bmatrix}
            \left( \nabla v, \nabla u\right) & \left(\nabla \cdot v, p \right) \\
            \left( q, \nabla \cdot u\right) & 0
        \end{bmatrix}

    This matrix is assembled below

    >>> from shenfun import *
    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD
    >>> N = (24, 24, 24)
    >>> K0 = FunctionSpace(N[0], 'Fourier', dtype='d')
    >>> K1 = FunctionSpace(N[1], 'Fourier', dtype='D')
    >>> SD = FunctionSpace(N[2], 'Legendre', bc=(0, 0))
    >>> ST = FunctionSpace(N[2], 'Legendre')
    >>> TD = TensorProductSpace(comm, (K0, K1, SD), axes=(2, 1, 0))
    >>> TT = TensorProductSpace(comm, (K0, K1, ST), axes=(2, 1, 0))
    >>> VT = VectorSpace(TD)
    >>> Q = CompositeSpace([VT, TD])
    >>> up = TrialFunction(Q)
    >>> vq = TestFunction(Q)
    >>> u, p = up
    >>> v, q = vq
    >>> A00 = inner(grad(v), grad(u))
    >>> A01 = inner(div(v), p)
    >>> A10 = inner(q, div(u))
    >>> M = BlockMatrix(A00+A01+A10)

    """
    def __init__(self, tpmats):
        assert isinstance(tpmats, (list, tuple))
        if isinstance(tpmats[0], TPMatrix):
            if len(tpmats[0].naxes) > 0:
                tpmats = get_simplified_tpmatrices(tpmats)
        tpmats = [tpmats] if not isinstance(tpmats[0], (list, tuple)) else tpmats
        self.testbase = testbase = tpmats[0][0].testbase
        self.trialbase = trialbase = tpmats[0][0].trialbase
        self.dims = dims = (testbase.num_components(), trialbase.num_components())
        self.mats = np.zeros(dims, dtype=int).tolist()
        self._Ai = None
        self.solver = None
        self += tpmats

    def __add__(self, a):
        """Return copy of self.__add__(a) <==> self+a"""
        return BlockMatrix(self.get_mats()+a.get_mats())

    def __iadd__(self, a):
        """self.__iadd__(a) <==> self += a

        Parameters
        ----------
        a : :class:`.BlockMatrix` or list of :class:`.TPMatrix` instances

        """
        if isinstance(a, BlockMatrix):
            tpmats = a.get_mats()
        elif isinstance(a, (list, tuple)):
            tpmats = a
        for mat in tpmats:
            if not isinstance(mat, list):
                mat = [mat]
            for m in mat:
                assert isinstance(m, (TPMatrix, SparseMatrix))
                i, j = m.global_index
                m0 = self.mats[i][j]
                if isinstance(m0, int):
                    self.mats[i][j] = [m]
                else:
                    found = False
                    for n in m0:
                        if m == n:
                            n += m
                            found = True
                            continue
                    if not found:
                        self.mats[i][j].append(m)

    def get_mats(self, return_first=False):
        """Return flattened list of matrices in self

        Parameters
        ----------
        return_first : bool, optional
            Return just the first matrix in the loop if True
        """
        tpmats = []
        for mi in self.mats:
            for mij in mi:
                if isinstance(mij, (list, tuple)):
                    for m in mij:
                        if isinstance(m, (TPMatrix, SparseMatrix)):
                            if return_first:
                                return m
                            else:
                                tpmats.append(m)
        return tpmats

    def matvec(self, v, c, format=None, use_scipy=None):
        """Compute matrix vector product

        .. math::

            c = A v

        where :math:`A` is the self block matrix and :math:`v,c` are flattened
        instances of the class :class:`.Function`.

        Parameters
        ----------
        v : :class:`.Function`
        c : :class:`.Function`
        format : str, optional
            The format of the matrices used for the matvec.
            See `Scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_

        use_scipy : boolean, optional
            Whether to assemble and use scipy's bmat for the matvec, or to use
            the matvec methods of this BlockMatrix's TPMatrices.
            Using ``config['matrix']['block']['use_scipy']`` if use_scipy is None
        Returns
        -------
        c : :class:`.Function`

        """
        assert v.function_space() == self.trialbase
        assert c.function_space() == self.testbase
        nvars = c.function_space().num_components()
        c = np.expand_dims(c, 0) if nvars == 1 else c
        v = np.expand_dims(v, 0) if nvars == 1 else v
        c.v.fill(0)
        use_scipy = config['matrix']['block']['use_scipy'] if use_scipy is None else use_scipy
        if self.contains_bc_matrix():
            use_scipy = False

        if use_scipy:
            self.assemble(format)
            daxes = self.testbase.get_diagonal_axes()
            if len(daxes) == self.testbase.dimensions:
                # Only Fourier
                assert isinstance(self._Ai, spmatrix)
                c.flatten()[:] = self._Ai * v.flatten()
            else:
                if len(daxes) > 0:
                    daxes += 1
                sl1, dims1 = self.trialbase._get_ndiag_slices_and_dims()
                sl2, dims2 = self.testbase._get_ndiag_slices_and_dims()
                gi = np.zeros(dims1[-1], dtype=v.dtype)
                go = np.zeros(dims2[-1], dtype=v.dtype)
                for key, val in self._Ai.items():
                    key = np.atleast_1d(key)
                    if len(daxes) > 0:
                        sl1.T[daxes] = np.array(key)[:, None]
                        sl2.T[daxes] = np.array(key)[:, None]
                    gi = v.copy_to_flattened(gi, key, dims1, sl1)
                    go[:] = val * gi
                    c = c.copy_from_flattened(go, key, dims2, sl2)

        else:
            z = np.zeros_like(c.v[0])
            for i, mi in enumerate(self.mats):
                for j, mij in enumerate(mi):
                    if isinstance(mij, Number):
                        if abs(mij) > 1e-8:
                            c.v[i] += mij*v.v[j]
                    else:
                        for m in mij:
                            z.fill(0)
                            z = m.matvec(v.v[j], z, format=format)
                            c.v[i] += z
        c = c.reshape(c.shape[1:]) if nvars == 1 else c
        v = v.reshape(v.shape[1:]) if nvars == 1 else v
        return c

    def __getitem__(self, ij):
        return self.mats[ij[0]][ij[1]]

    def contains_bc_matrix(self):
        for mi in self.mats:
            for mij in mi:
                if isinstance(mij, (list, tuple)):
                    for m in mij:
                        if m.is_bc_matrix() is True:
                            return True
        return False

    def contains_regular_matrix(self):
        for mi in self.mats:
            for mij in mi:
                if isinstance(mij, (list, tuple)):
                    for m in mij:
                        if m.is_bc_matrix() is False:
                            return True
        return False

    def assemble(self, format=None):
        """Assemble matrices in scipy sparse format

        Parameters
        ----------
        format : str or None, optional
            The format of the sparse scipy matrix. Using
            ``config['matrix']['block']['assemble']`` if None.
        """
        if self._Ai is not None:
            return
        self._Ai = {}
        N = self.testbase.forward.output_array.shape
        dimensions = self.testbase.dimensions
        daxes = self.get_diagonal_axes()
        ndindices = [(0,)] if (len(daxes) == 0 or len(daxes) == dimensions) else np.ndindex(tuple(np.array(N)[daxes]))
        format = config['matrix']['block']['assemble'] if format is None else format
        for i in ndindices:
            i = i[0] if len(i) == 1 else i
            if format == 'csc':
                Ai = self.diags(i, format='csr').tocsc() # because of bug in scipy
            else:
                Ai = self.diags(i, format=format)
            self._Ai[i] = Ai

    def get_diagonal_axes(self):
        if self.testbase.dimensions == 1:
            return np.array([])
        tpmat = self.get_mats(True)
        return np.setxor1d(tpmat.naxes, range(tpmat.dimensions)).astype(int)

    def diags(self, it=None, format=None):
        """Return global block matrix in scipy sparse format

        For multidimensional forms the returned matrix is constructed for
        given indices in the periodic directions.

        Parameters
        ----------
        it : n-tuple of ints or None, optional
            where n is dimensions-1. These are the indices into the diagonal
            axes, or the axes with Fourier bases.
        format : str or None, optional
            The format of the returned matrix. See `Scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
            If None, then use default for :class:`.TPMatrix`.

        """
        from .spectralbase import MixedFunctionSpace
        if self.contains_bc_matrix() and self.contains_regular_matrix():
            raise RuntimeError('diags only works for pure boundary or pure regular matrices. Consider splitting this BlockMatrix using :func:`.BlockMatrices`')
        bm = []
        for mi in self.mats:
            bm.append([])
            for mij in mi:
                if isinstance(mij, Number):
                    bm[-1].append(None)
                else:
                    m = mij[0]
                    if isinstance(self.testbase, MixedFunctionSpace) or len(m.naxes) == len(m.mats) or len(m.naxes) == 0:
                        d = m.diags(format)
                        for mj in mij[1:]:
                            d = d + mj.diags(format)

                    elif len(m.naxes) == 2: # 2 non-periodic directions
                        iit = np.where(np.array(m.scale.shape) == 1, 0, it) # if shape is 1 use index 0, else use given index (shape=1 means the scale is constant in that direction)
                        d = m.scale[tuple(iit)]*kron(m.mats[m.naxes[0]].diags(format=format), m.mats[m.naxes[1]].diags(format=format))
                        for mj in mij[1:]:
                            iit = np.where(np.array(mj.scale.shape) == 1, 0, it)
                            sc = mj.scale[tuple(iit)]
                            d = d + sc*kron(mj.mats[mj.naxes[0]].diags(format=format), mj.mats[mj.naxes[1]].diags(format=format))

                    else:
                        assert len(m.naxes) == 1
                        iit = np.zeros(m.dimensions, dtype=int)
                        diagonal_axes = self.get_diagonal_axes()
                        assert len(diagonal_axes) + len(m.naxes) == m.dimensions
                        iit[diagonal_axes] = it
                        ij = np.where(np.array(m.scale.shape) == 1, 0, iit) # if shape is 1 use index 0, else use given index (shape=1 means the scale is constant in that direction)
                        sc = m.scale[tuple(ij)]
                        d = sc*m.mats[m.naxes[0]].diags(format)
                        for mj in mij[1:]:
                            ij = np.where(np.array(mj.scale.shape) == 1, 0, iit)
                            sc = mj.scale[tuple(ij)]
                            d = d + sc*mj.mats[mj.naxes[0]].diags(format)
                    bm[-1].append(d)
        return bmat(bm, format=format)

    def solve(self, b, u=None, constraints=()):
        r"""
        Solve matrix system Au = b

        where A is the current :class:`.BlockMatrix` (self)

        Parameters
        ----------
        b : array
            Array of right hand side
        u : array, optional
            Output array
        constraints : sequence of 3-tuples of (int, int, number)
            Any 3-tuple describe a dof to be constrained. The first int
            represents the block number of the function to be constrained. The
            second int gives which degree of freedom to constrain and the number
            gives the value it should obtain. For example, for the global
            restriction that

            .. math::

                \frac{1}{V}\int p dx = number

            where we have

            .. math::

                p = \sum_{k=0}^{N-1} \hat{p}_k \phi_k

            it is sufficient to fix the first dof of p, \hat{p}_0, since
            the bases are created such that all basis functions except the
            first integrates to zero. So in this case the 3-tuple can be
            (2, 0, 0) if p is found in block 2 of the mixed basis.

            The constraint can only be applied to bases with no given
            explicit boundary condition, like the pure Chebyshev or Legendre
            bases.

        """
        from .la import BlockMatrixSolver
        sol = self.solver
        if self.solver is None:
            sol = BlockMatrixSolver(self)
            self.solver = sol
        u = sol(b, u, constraints)
        return u


class TPMatrix:
    """Tensor product matrix

    A :class:`.TensorProductSpace` is the tensor product of ``D`` univariate
    function spaces. A normal matrix (a second order tensor) is assembled from
    bilinear forms (i.e., forms containing both test and trial functions) on
    one univariate function space. A bilinear form on a tensor product space
    will assemble to ``D`` outer products of such univariate matrices. That is,
    for a two-dimensional tensor product you get fourth order tensors (outer
    product of two matrices), and three-dimensional tensor product spaces leads
    to a sixth order tensor (outer product of three matrices). This class
    contains ``D`` second order matrices. The complete matrix is as such the
    outer product of these ``D`` matrices.

    Note that the outer product of two matrices often is called the Kronecker
    product.

    Parameters
    ----------
    mats : sequence, or sequence of sequence of matrices
        Instances of :class:`.SpectralMatrix` or :class:`.SparseMatrix`
        The length of ``mats`` is the number of dimensions of the
        :class:`.TensorProductSpace`
    testspace : Function space
        The test :class:`.TensorProductSpace`
    trialspace : Function space
        The trial :class:`.TensorProductSpace`
    scale : array, optional
        Scalar multiple of matrices. Must have ndim equal to the number of
        dimensions in the :class:`.TensorProductSpace`, and the shape must be 1
        along any directions with a nondiagonal matrix.
    global_index : 2-tuple, optional
        Indices (test, trial) into mixed space :class:`.CompositeSpace`.
    testbase : :class:`.CompositeSpace`, optional
         Instance of the base test space
    trialbase : :class:`.CompositeSpace`, optional
         Instance of the base trial space
    """
    def __init__(self, mats, testspace, trialspace, scale=1.0, global_index=None,
                 testbase=None, trialbase=None):
        assert isinstance(mats, (list, tuple))
        assert len(mats) == len(testspace)
        self.mats = mats
        self.space = testspace
        self.trialspace = trialspace
        self.scale = scale
        self.pmat = 1
        self.naxes = testspace.get_nondiagonal_axes()
        self.global_index = global_index
        self.testbase = testbase
        self.trialbase = trialbase
        self._issimplified = False

    def get_simplified(self):
        diagonal_axes = np.setxor1d(self.naxes, range(self.space.dimensions)).astype(int)
        if len(diagonal_axes) == 0 or self._issimplified:
            return self

        mats = []
        scale = copy(self.scale)
        for axis in range(self.dimensions):
            mat = self.mats[axis]
            if axis in diagonal_axes:
                d = mat[0]
                if np.ndim(d):
                    d = self.space[axis].broadcast_to_ndims(d*mat.scale)
                scale = scale*d
                mat = Identity(mat.shape)
            mats.append(mat)
        tpmat = TPMatrix(mats, self.space, self.trialspace, scale=scale,
                         global_index=self.global_index,
                         testbase=self.testbase, trialbase=self.trialbase)

        # Decomposition
        if len(self.space) > 1:
            s = tpmat.scale.shape
            ss = [slice(None)]*self.space.dimensions
            ls = self.space.local_slice()
            for axis, shape in enumerate(s):
                if shape > 1:
                    ss[axis] = ls[axis]
            tpmat.scale = (tpmat.scale[tuple(ss)]).copy()

        # If only one non-diagonal matrix, then make a simple link to
        # this matrix.
        if len(tpmat.naxes) == 1:
            tpmat.pmat = tpmat.mats[tpmat.naxes[0]]
        elif len(tpmat.naxes) == 2: # 2 nondiagonal
            tpmat.pmat = tpmat.mats
        tpmat._issimplified = True
        return tpmat

    def simplify_diagonal_matrices(self):
        if self._issimplified:
            return

        diagonal_axes = np.setxor1d(self.naxes, range(self.space.dimensions)).astype(int)
        if len(diagonal_axes) == 0:
            return

        for axis in diagonal_axes:
            mat = self.mats[axis]
            if self.dimensions == 1: # Don't bother with the 1D case
                continue
            else:
                d = mat[0]    # get diagonal
                if np.ndim(d):
                    d = self.space[axis].broadcast_to_ndims(d*mat.scale)
                self.scale = self.scale*d
                self.mats[axis] = Identity(mat.shape)

        # Decomposition
        if len(self.space) > 1:
            s = self.scale.shape
            ss = [slice(None)]*self.space.dimensions
            ls = self.space.local_slice()
            for axis, shape in enumerate(s):
                if shape > 1:
                    ss[axis] = ls[axis]
            self.scale = (self.scale[tuple(ss)]).copy()

        # If only one non-diagonal matrix, then make a simple link to
        # this matrix.
        if len(self.naxes) == 1:
            self.pmat = self.mats[self.naxes[0]]
        elif len(self.naxes) == 2: # 2 nondiagonal
            self.pmat = self.mats
        self._issimplified = True

    def solve(self, b, u=None, constraints=()):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b

        tpmat = self.get_simplified()
        if len(tpmat.naxes) == 0:
            if np.all([isinstance(m, Identity) for m in tpmat.mats]) and isinstance(tpmat.scale, Number):
                if abs(tpmat.scale-1) < 1e-8:
                    return u
            sl = tuple([s.slice() for s in tpmat.trialspace.bases])
            d = tpmat.scale
            with np.errstate(divide='ignore'):
                d = 1./tpmat.scale
            if constraints:
                assert constraints[0] == (0, 0)
            # Constraint is enforced automatically
            d = np.where(np.isfinite(d), d, 0)
            u[sl] = b[sl] * d[sl]

        elif len(tpmat.naxes) == 1:
            from shenfun.la import SolverGeneric1ND
            H = SolverGeneric1ND([tpmat])
            u = H(b, u, constraints=constraints)

        elif len(tpmat.naxes) == 2:
            from shenfun.la import SolverGeneric2ND
            H = SolverGeneric2ND([tpmat])
            u = H(b, u, constraints=constraints)
        return u

    def matvec(self, v, c, format=None):
        tpmat = self.get_simplified()
        c.fill(0)
        if len(tpmat.naxes) == 0:
            c[:] = tpmat.scale*v
        elif len(tpmat.naxes) == 1:
            axis = tpmat.naxes[0]
            rank = v.rank if hasattr(v, 'rank') else 0
            if rank == 0:
                c = tpmat.pmat.matvec(v, c, format=format, axis=axis)
            else:
                c = tpmat.pmat.matvec(v[tpmat.global_index[1]], c, format=format, axis=axis)
            c[:] = c*tpmat.scale
        elif len(tpmat.naxes) == 2:
            # 2 non-periodic directions (may be non-aligned in second axis, hence transfers)
            npaxes = deepcopy(list(tpmat.naxes))
            space = tpmat.space
            newspace = False
            if space.forward.input_array.shape != space.forward.output_array.shape:
                space = space.get_unplanned(True) # in case self.space is padded
                newspace = True

            pencilA = space.forward.output_pencil
            subcomms = [s.Get_size() for s in pencilA.subcomm]
            axis = pencilA.axis
            assert subcomms[axis] == 1
            npaxes.remove(axis)
            second_axis = npaxes[0]
            pencilB = pencilA.pencil(second_axis)
            transAB = pencilA.transfer(pencilB, c.dtype.char)
            cB = np.zeros(transAB.subshapeB, dtype=c.dtype)
            cC = np.zeros(transAB.subshapeB, dtype=c.dtype)
            bb = tpmat.mats[axis]
            c = bb.matvec(v, c, format=format, axis=axis)
            # align in second non-periodic axis
            transAB.forward(c, cB)
            bb = tpmat.mats[second_axis]
            cC = bb.matvec(cB, cC, format=format, axis=second_axis)
            transAB.backward(cC, c)
            c *= tpmat.scale
            if newspace:
                space.destroy()

        return c

    def get_key(self):
        naxis = self.space.get_nondiagonal_axes()
        assert len(naxis) == 1
        return self.mats[naxis[0]].get_key()

    def isidentity(self):
        return np.all([m.isidentity() for m in self.mats])

    def isdiagonal(self):
        return np.all([m.isdiagonal() for m in self.mats])

    def is_bc_matrix(self):
        for m in self.mats:
            if m.is_bc_matrix():
                return True
        return False

    @property
    def dimensions(self):
        """Return dimension of TPMatrix"""
        return len(self.mats)

    def __mul__(self, a):
        """Returns copy of self.__mul__(a) <==> self*a"""
        if isinstance(a, Number):
            return TPMatrix(self.mats, self.space, self.trialspace, self.scale*a,
                            self.global_index, self.testbase, self.trialbase)

        assert isinstance(a, np.ndarray)
        c = np.empty_like(a)
        c = self.matvec(a, c)
        return c

    def __rmul__(self, a):
        """Returns copy of self.__rmul__(a) <==> a*self"""
        if isinstance(a, Number):
            return self.__mul__(a)
        else:
            raise NotImplementedError

    def __imul__(self, a):
        """Returns self.__imul__(a) <==> self*=a"""
        if isinstance(a, Number):
            self.scale *= a
        elif isinstance(a, np.ndarray):
            self.scale = self.scale*a
        return self

    def __div__(self, a):
        """Returns copy self.__div__(a) <==> self/a"""
        if isinstance(a, Number):
            return TPMatrix(self.mats, self.space, self.trialspace, self.scale/a,
                            self.global_index, self.testbase, self.trialbase)
        elif isinstance(a, np.ndarray):
            b = np.zeros_like(a)
            b = self.solve(a, b)
            return b
        else:
            raise NotImplementedError

    def __neg__(self):
        """self.__neg__() <==> -self"""
        A = self.copy()
        A.scale = self.scale*-1
        return A

    def __eq__(self, a):
        """Check if matrices and global_index are the same.

        Note
        ----
        The attribute scale may still be different
        """
        assert isinstance(a, TPMatrix)
        if not self.global_index == a.global_index:
            return False
        for m0, m1 in zip(self.mats, a.mats):
            if not m0.get_key() == m1.get_key():
                return False
            if not m0 == m1:
                return False
        return True

    def __ne__(self, a):
        return not self.__eq__(a)

    def __add__(self, a):
        """Return copy of self.__add__(a) <==> self+a"""
        assert isinstance(a, TPMatrix)
        assert self == a
        return TPMatrix(self.mats, self.space, self.trialspace, self.scale+a.scale,
                        self.global_index, self.testbase, self.trialbase)

    def __iadd__(self, a):
        """self.__iadd__(a) <==> self += a"""
        assert isinstance(a, TPMatrix)
        assert self == a
        self.scale = self.scale + a.scale
        return self

    def __sub__(self, a):
        """Return copy of self.__sub__(a) <==> self-a"""
        assert isinstance(a, TPMatrix)
        assert self == a
        return TPMatrix(self.mats, self.space, self.trialspace, self.scale-a.scale,
                        self.global_index, self.testbase, self.trialbase)

    def __isub__(self, a):
        """self.__isub__(a) <==> self -= a"""
        assert isinstance(a, TPMatrix)
        assert self == a
        self.scale = self.scale - a.scale
        return self

    def copy(self):
        """Return TPMatrix deep copy of self"""
        return self.__deepcopy__()

    def __copy__(self):
        mats = []
        for mat in self.mats:
            mats.append(mat.__copy__())
        return TPMatrix(mats, self.space, self.trialspace, self.scale,
                        self.global_index, self.testbase, self.trialbase)

    def __deepcopy__(self, memo=None, _nil=[]):
        mats = []
        for mat in self.mats:
            mats.append(mat.__deepcopy__())
        return TPMatrix(mats, self.space, self.trialspace, self.scale,
                        self.global_index, self.testbase, self.trialbase)

    def diags(self, format=None):
        assert self._issimplified is False
        if self.dimensions == 2:
            mat = kron(self.mats[0].diags(format=format),
                       self.mats[1].diags(format=format),
                       format=format)
        elif self.dimensions == 3:
            mat = kron(self.mats[0].diags(format=format),
                       kron(self.mats[1].diags(format=format),
                            self.mats[2].diags(format=format),
                            format=format),
                       format=format)
        elif self.dimensions == 4:
            mat = kron(self.mats[0].diags(format=format),
                       kron(self.mats[1].diags(format=format),
                            kron(self.mats[2].diags(format=format),
                                 self.mats[3].diags(format=format),
                                 format=format),
                            format=format),
                       format=format)
        elif self.dimensions == 5:
            mat = kron(self.mats[0].diags(format=format),
                       kron(self.mats[1].diags(format=format),
                            kron(self.mats[2].diags(format=format),
                                 kron(self.mats[3].diags(format=format),
                                      self.mats[4].diags(format=format),
                                      format=format),
                                 format=format),
                            format=format),
                       format=format)

        return mat*np.atleast_1d(self.scale).item()

def get_simplified_tpmatrices(tpmats):
    """Return copy of tpmats list, where diagonal matrices have been
    simplified and placed in scale arrays.

    Parameters
    ----------
    tpmats
        Instances of :class:`.TPMatrix`

    Returns
    -------
    List[TPMatrix]
        List of :class:`.TPMatrix`'es, that have been simplified

    """
    A = []
    for tpmat in tpmats:
        A.append(tpmat.get_simplified())

    # Add equal matrices
    B = [A[0]]
    for a in A[1:]:
        found = False
        for b in B:
            if a == b:
                b += a
                found = True
        if not found:
            B.append(a)
    return B


def check_sanity(A, test, trial, measure=1, assemble='quadrature', kind='vandermonde', fixed_resolution=None):
    """Sanity check for matrix.

    Test that created matrix agrees with quadrature computed using a
    memory-consuming Vandermonde implementation.

    Parameters
    ----------
    A : matrix
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
        - :mod:`.chebyshevu.bases`
        - :mod:`.ultraspherical.bases`
        - :mod:`.fourier.bases`
        - :mod:`.laguerre.bases`
        - :mod:`.hermite.bases`
        - :mod:`.jacobi.bases`

        The int represents the number of times the test function
        should be differentiated. Representing matrix row.
    trial : 2-tuple of (basis, int)
        As test, but representing matrix column.
    measure : sympy function of coordinate, optional
        Function in the physical coordinate. Gets mapped to
        reference domain.
    assemble : str, optional
        Determines how to perform the integration we compare with

        - 'exact'
        - 'adaptive'
        - 'quadrature'
    kind : str, optional
        The type of quadrature to compare with.

        - 'vandermonde'
        - 'stencil'
    fixed_resolution : None or str, optional
        A fixed number of quadrature points used to compute the matrix.
        If 'fixed_resolution' is set, then assemble is set to 'quadrature' and
        kind is set to 'vandermonde'.

    """
    if fixed_resolution is not None:
        kind = 'vandermonde'
    if len(sp.sympify(measure).free_symbols) == 1:
        if test[0].domain != test[0].reference_domain():
            x0 = measure.free_symbols.pop()
            xm = test[0].map_true_domain(x0)
            measure = measure.replace(x0, xm)
    Dsp = SpectralMatrix(test, trial, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
    for key, val in A.items():
        assert np.allclose(val*A.scale, Dsp[key])

def extract_diagonal_matrix(M, lowerband=None, upperband=None, abstol=1e-10, reltol=1e-10):
    """Return SparseMatrix version of dense matrix ``M``

    Parameters
    ----------
    M : Numpy array of ndim=2 or sparse scipy matrix
    lowerband : int or None
        Assumed lower bandwidth of M
    upperband : int or None
        Assumed upper bandwidth of M
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
    if isinstance(M, spmatrix):
        M = M.tocsr()
    relmax = abs(M).max()
    dtype = float if M.dtype == 'O' else M.dtype # For mpf object
    upperband = M.shape[1] if upperband is None else min(upperband+1, M.shape[1])
    lowerband = M.shape[0]-1 if lowerband is None else min(lowerband, M.shape[0]-1)
    for i in range(-lowerband, upperband):
        u = M.diagonal(i).copy()
        if abs(u).max() > abstol and abs(u).max()/relmax > reltol:
            d[i] = np.array(u, dtype=dtype)
    return SparseMatrix(d, M.shape)

def extract_bc_matrices(mats):
    """Extract boundary matrices from list of ``mats``

    Parameters
    ----------
    mats : list of list of instances of :class:`.TPMatrix` or
        :class:`.SparseMatrix`

    Returns
    -------
    list
        list of boundary matrices.

    Note
    ----
    The ``mats`` list is modified in place since boundary matrices are
    extracted.
    """
    bc_mats = []
    for a in mats:
        for b in a.copy():
            if b.is_bc_matrix():
                bc_mats.append(b)
                a.remove(b)
    return bc_mats

def _get_matrix(test, trial, measure=1, assemble=None, fixed_resolution=None):
    """Return assembled matrix

    This internal function is used by :class:`.SpectralMatrix`

    Parameters
    ----------
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
        - :mod:`.chebyshevu.bases`
        - :mod:`.ultraspherical.bases`
        - :mod:`.fourier.bases`
        - :mod:`.laguerre.bases`
        - :mod:`.hermite.bases`
        - :mod:`.jacobi.bases`

        The int represents the number of times the test function
        should be differentiated. Representing matrix row.
    trial : 2-tuple of (basis, int)
        As test, but representing matrix column.
    measure : Sympy expression of coordinate, or number, optional
        Additional weight to integral. For example, in cylindrical
        coordinates an additional measure is the radius `r`.
    assemble : None or str, optional
        Determines how to perform the integration

        - 'quadrature' (default)
        - 'exact'
        - 'adaptive'
    fixed_resolution : None or str, optional
        A fixed number of quadrature points used to compute the inner product.
        If 'fixed_resolution' is set, then assemble is set to 'quadrature'.

    Note
    ----
    The computed matrix is not compensated for a non-standard domain size.
    This is because all pre-computed matrices use the reference domain, and
    compensate for the domain size later. This function is a drop-in for all
    pre-computed matrices, and thus needs to assume a standard reference domain.
    To create a true matrix with this function, do not use it directly, but
    wrapped in the SpectralMatrix class, like

    >>> from shenfun import inner, TestFunction, TrialFunction, FunctionSpace
    >>> L = FunctionSpace(4, 'L', domain=(-2, 2))
    >>> u = TrialFunction(L)
    >>> v = TestFunction(L)
    >>> D = inner(u, v, assemble='exact')
    >>> dict(D)
    {0: array([4.        , 1.33333333, 0.8       , 0.57142857])}

    """
    K0 = test[0].slice().stop - test[0].slice().start
    K1 = trial[0].slice().stop - trial[0].slice().start

    if assemble == 'quadrature':

        if fixed_resolution is not None:
            test2 = test[0].get_refined(fixed_resolution)
            N = test2.N
            x = test2.points_and_weights(N, map_true_domain=False)[0]
            ws = test2.get_measured_weights(N, measure, map_true_domain=False)
        else:
            N = test[0].N
            x = test[0].points_and_weights(N, map_true_domain=False)[0]
            ws = test[0].get_measured_weights(N, measure, map_true_domain=False)
        u = trial[0].evaluate_basis_derivative_all(x=x, k=trial[1])[:, :K1]
        if trial[0].boundary_condition() == 'Apply':
            if np.linalg.norm(u) < 1e-14:
                return {}
        v = test[0].evaluate_basis_derivative_all(x=x, k=test[1])[:, :K0]
        V = np.dot(np.conj(v.T)*ws[np.newaxis, :], u)

    else: # exact or adaptive

        x = sp.Symbol('x', real=True)

        # Exact integration is much more expensive than quadrature and
        # as such we use quadrature first simply to get the sparsity pattern.
        try:
            R = _get_matrix(test, trial, measure=measure, assemble='quadrature')
        except:
            R = {k: None for k in np.arange(-test[0].dim(), test[0].dim()+1)}

        V = np.zeros((K0, K1), dtype=test[0].forward.output_array.dtype)
        if test[0].family() == 'chebyshev' and assemble == 'exact':
            # Transform integral using x=cos(theta)
            if not measure == 1:
                if isinstance(measure, sp.Expr):
                    s = measure.free_symbols
                    assert len(s) == 1
                    x = s.pop()
                    xm = test[0].map_true_domain(x)
                    measure = measure.subs(x, sp.cos(xm))
                else:
                    assert isinstance(measure, Number)
            assert test[1] == 0
            S0 = test[0].stencil_matrix().diags('csr')
            S1 = trial[0].stencil_matrix().diags('csr')
            for i in range(test[0].slice().start, test[0].slice().stop):
                M0 = S0.getrow(i)
                pi = sp.S(0)
                for ind, d in zip(M0.indices, M0.data):
                    pi += d*sp.cos(ind*x)
                for jq in R.keys():
                    j = i+jq
                    if j < 0 or j >= K1:
                        continue
                    M1 = S1.getrow(j)
                    pj = sp.S(0)
                    for ind, d in zip(M1.indices, M1.data):
                        pj += d*sp.cos(ind*x)

                    # df(theta)/dx = df/dtheta*dtheta/dx - apply recursively
                    for _ in range(trial[1]):
                        pj = -pj.diff(x, 1)/sp.sin(x)

                    V[i, j] = sp.integrate(measure*pi*pj, (x, 0, sp.pi))

        else:
            if not measure == 1:
                if isinstance(measure, sp.Expr):
                    s = measure.free_symbols
                    assert len(s) == 1
                    x = s.pop()
                    xm = test[0].map_true_domain(x)
                    measure = measure.subs(x, xm)
                else:
                    assert isinstance(measure, Number)

            cheb = test[0].family() == 'chebyshev'
            if cheb: # use adaptive quadrature with weight incorporated
                w = {'weight': 'alg',
                     'wvar': (-0.5, -0.5)}
            else:
                w = {}
                measure *= test[0].weight() # Weight of weighted space (in reference domain)
            domain = test[0].reference_domain()
            for i in range(test[0].slice().start, test[0].slice().stop):
                pi = np.conj(test[0].basis_function(i, x=x))
                for jq in R.keys():
                    j = i+jq
                    if j < 0 or j >= K1:
                        continue
                    pj = trial[0].basis_function(j, x=x)
                    integrand = measure*pi.diff(x, test[1])*pj.diff(x, trial[1])
                    if assemble == 'exact':
                        V[i, j] = integrate_sympy(integrand, (x, domain[0], domain[1]))
                    elif assemble == 'adaptive':
                        if isinstance(integrand, Number):
                            if cheb:
                                V[i, j] = integrand*np.pi
                            else:
                                V[i, j] = integrand*float(domain[1]-domain[0])
                        else:
                            V[i, j] = quad(sp.lambdify(x, integrand), float(domain[0]), float(domain[1]), **w)[0]

    if V.dtype.char in 'FDG':
        ni = np.linalg.norm(V.imag)
        if ni == 0:
            V = V.real.copy()
        elif np.linalg.norm(V.real) / ni > 1e14:
            V = V.real.copy()
    return extract_diagonal_matrix(V)

def assemble_stencil(test, trial, measure=1):
    if trial[0].is_boundary_basis:
        return _assemble_stencil_bc(test, trial, measure)
    return _assemble_stencil(test, trial, measure)

def _assemble_stencil_bc(test, trial, measure=1):
    from shenfun.spectralbase import inner_product
    Tv = test[0].get_orthogonal(domain=(-1, 1))
    Tu = trial[0].get_orthogonal(domain=(-1, 1))
    B = inner_product((Tv, test[1]), (Tu, trial[1]))
    if len(B) == 0:
        return {}
    K = test[0].stencil_matrix()
    q = sp.degree(measure)

    K.shape = (test[0].dim(), test[0].N)
    S = extract_diagonal_matrix(trial[0].stencil_matrix().T).diags('csr')
    if measure != 1:
        from shenfun.jacobi.recursions import pmat, a
        from shenfun.utilities import split
        assert sp.sympify(measure).is_polynomial()
        A = sp.S(0)
        for dv in split(measure, expand=True):
            alpha = test[0].alpha
            beta = test[0].beta
            gn = test[0].gn
            sc = dv['coeff']
            msi = dv['x']
            qi = sp.degree(msi)
            Ax = pmat(a, qi, alpha, beta, test[0].N, test[0].N, gn)
            A = A + sc*Ax.diags('csr')
        A = K.diags('csr') * A.T * B.diags('csr') * S
    else:
        A = K.diags('csr') * B.diags('csr') * S
    M = B.shape[1]
    K.shape = (test[0].N, test[0].N)
    d = extract_diagonal_matrix(A, lowerband=M+q, upperband=M)
    d = d._storage
    return d

def _assemble_stencil(test, trial, measure=1):
    from shenfun.spectralbase import inner_product
    Tv = test[0].get_orthogonal(domain=(-1, 1))
    Tu = trial[0].get_orthogonal(domain=(-1, 1))
    alpha = test[0].alpha
    beta = test[0].beta
    gn = test[0].gn
    # This needs to be either implemented or quadrature:
    B = inner_product((Tv, test[1]), (Tu, trial[1]))
    K = test[0].stencil_matrix()
    q = sp.degree(measure)

    K.shape = (test[0].dim(), test[0].N)
    S = trial[0].stencil_matrix()
    S.shape = (trial[0].dim(), trial[0].N)
    if measure != 1:
        from shenfun.jacobi.recursions import pmat, a
        from shenfun.utilities import split
        assert sp.sympify(measure).is_polynomial()
        A = sp.S(0)
        for dv in split(measure, expand=True):
            sc = dv['coeff']
            msi = dv['x']
            qi = sp.degree(msi)
            Ax = pmat(a, qi, alpha, beta, test[0].N, test[0].N, gn)
            A = A + sc*Ax.diags('csr')
        A = K.diags('csr') * A.T * B.diags('csr') * S.diags('csr').T
    else:
        A = K.diags('csr') * B.diags('csr') * S.diags('csr').T
    K.shape = (test[0].N, test[0].N)
    S.shape = (trial[0].N, trial[0].N)
    if test[1]+trial[1] == 0 and test[0].family() == trial[0].family():
        keysK = np.sort(np.array(list(K.keys())))
        keysS = np.sort(np.array(list(S.keys())))
        lb = -keysK[0]+keysS[-1]+q
        ub = keysK[-1]-keysS[0]+q
        d = extract_diagonal_matrix(A, lowerband=lb, upperband=ub)
    else:
        # compute the sparsity pattern
        Ac = A.tocsc()
        ub = Ac.getrow(0).indices
        ub2 = Ac.getrow(1).indices
        if len(ub) == 0 and len(ub2) == 0:
            ub = 0
        else:
            ub = trial[0].dim() if len(ub) == 0 else ub.max()
            ub2 = trial[0].dim() if len(ub2) == 0 else ub2.max()-1
            ub = max(ub, ub2)

        lb = Ac.getcol(0).indices
        lb2 = Ac.getcol(1).indices
        if len(lb) == 0 and len(lb2) == 0:
            lb = 0
        else:
            lb = test[0].dim() if len(lb) == 0 else lb.max()
            lb2 = test[0].dim() if len(lb2) == 0 else lb2.max()-1
            lb = max(lb, lb2)
        d = extract_diagonal_matrix(A, lowerband=lb, upperband=ub)
    d = d._storage
    return d

def assemble_phi(test, trial, measure=1):
    assert test[0].short_name() in ('P1', 'P2', 'P3', 'P4')
    if trial[0].is_boundary_basis:
        return _assemble_phi_bc(test, trial, measure)
    return _assemble_phi(test, trial, measure)

def _assemble_phi(test, trial, measure=1):
    from shenfun.jacobi.recursions import Lmat
    from shenfun.utilities import split
    assert test[0].quad != 'GL'
    alpha = test[0].alpha
    beta = test[0].beta
    gn = test[0].gn
    q = sp.degree(measure)
    k = (test[0].N-test[0].dim())//2
    l = k-trial[1]
    assert l >= 0
    D = sp.S(0)
    for dv in split(measure, expand=True):
        sc = dv['coeff']
        msi = dv['x']
        assert sp.sympify(msi).is_polynomial()
        qi = sp.degree(msi)
        Ax = Lmat(k, qi, l, test[0].dim(), trial[0].N, alpha, beta, gn)
        D = D + sc*Ax
    if trial[0].is_orthogonal:
        d = extract_diagonal_matrix(D, lowerband=q-k+l, upperband=q+k+l)
    else:
        K = trial[0].stencil_matrix()
        K.shape = (trial[0].dim(), trial[0].N)
        keys = np.sort(np.array(list(K.keys())))
        lb, ub = -keys[0], keys[-1]
        d = extract_diagonal_matrix(D*K.diags('csr').T, lowerband=q-k+l+ub, upperband=q+k+l+lb)
        K.shape = (trial[0].N, trial[0].N)
    d = d._storage
    return d

def _assemble_phi_bc(test, trial, measure=1):
    assert trial[0].short_name() == 'BG'
    from shenfun.jacobi.recursions import Lmat
    from shenfun.utilities import split
    alpha = test[0].alpha
    beta = test[0].beta
    gn = test[0].gn
    M = test[0].dim()
    N = trial[0].dim_ortho
    q = sp.degree(measure)
    k = (test[0].N-test[0].dim())//2
    l = k-trial[1]
    assert l >= 0
    D = sp.S(0)
    if k <= N:
        for dv in split(measure, expand=True):
            sc = dv['coeff']
            msi = dv['x']
            assert sp.sympify(msi).is_polynomial()
            qi = sp.degree(msi)
            Ax = Lmat(k, qi, l, M, N, alpha, beta, gn)
            D = D + sc*Ax
    if D is sp.S.Zero:
        d = {0: 1}
    else:
        K = trial[0].stencil_matrix()
        d = extract_diagonal_matrix(D*extract_diagonal_matrix(K).diags('csr').T, lowerband=N+q, upperband=N)
        d = d._storage
    return d

def assemble_sympy(test, trial, measure=1, implicit=True, assemble='exact'):
    """Return sympy representation of mass matrix

    Parameters
    ----------
    test : :class:`.TestFunction` or 2-tuple of :class:`.SpectralBase`, int
        If 2-tuple, then the integer represents the number of derivatives, which
        should be zero for this function
    trial : Like test but representing trial function
    measure : Number or Sympy function
        Function of coordinate
    implicit : bool, optional
        Whether to use unevaluated Sympy functions instead of the actual values
        of the diagonals.
    assemble : str, optional
        - 'exact'
        - 'quadrature'

    Example
    -------
    >>> from shenfun import assemble_sympy, TrialFunction, TestFunction
    >>> N = 8
    >>> D = FunctionSpace(N, 'C', bc=(0, 0))
    >>> v = TestFunction(D)
    >>> u = TrialFunction(D)
    >>> assemble_sympy(v, u)
    (KroneckerDelta(i, j) - KroneckerDelta(i, j + 2))*h(i) - (KroneckerDelta(j, i + 2) - KroneckerDelta(i + 2, j + 2))*h(i + 2)

    Note that when implicit is True, then h(i) represents the l2-norm,
    or the L2-norm (if exact) of the orthogonal basis.

    >>> D = FunctionSpace(N, 'C', bc={'left': {'N': 0}, 'right': {'N': 0}})
    >>> u = TrialFunction(D)
    >>> v = TestFunction(D)
    >>> assemble_sympy(v, u, implicit=True)
    (KroneckerDelta(i, j) + KroneckerDelta(i, j + 2)*s2(j))*h(i) + (KroneckerDelta(j, i + 2) + KroneckerDelta(i + 2, j + 2)*s2(j))*h(i + 2)*k2(i)
    >>> assemble_sympy(v, u, implicit=False)
    -i**2*(-j**2*KroneckerDelta(i + 2, j + 2)/(j + 2)**2 + KroneckerDelta(j, i + 2))*h(i + 2)/(i + 2)**2 + (-j**2*KroneckerDelta(i, j + 2)/(j + 2)**2 + KroneckerDelta(i, j))*h(i)

    Here the implicit version uses 'k' for the diagonals of the test function,
    and 's' for the trial function. The number represents the location of the
    diagonal, so 's2' is the second upper diagonal of the stencil matrix of the
    trial function.

    You can get the diagonals like this:
    >>> import sympy as sp
    >>> i, j = sp.symbols('i,j', integer=True)
    >>> M = assemble_sympy(v, u, implicit=False)
    >>> M.subs(j, i) # main diagonal
    pi*i**4/(2*(i + 2)**4) + pi/2
    >>> M.subs(j, i+2) # second upper diagonal
    -pi*i**2/(2*(i + 2)**2)
    >>> M.subs(j, i-2) # second lower diagonal
    -pi*(i - 2)**2/(2*i**2)

    i is the row number, so the last one starts for i=2.

    """
    if isinstance(test, tuple) and isinstance(trial, tuple):
        assert len(test) == 2 and len(trial) == 2
        assert test[1]+trial[1] == 0, 'Only implemented for mass matrix, because need B to be diagonal.'
        test = test[0]
        trial = trial[0]
    else:
        from shenfun.forms import TestFunction, TrialFunction
        assert isinstance(test, TestFunction) and isinstance(trial, TrialFunction)
        test = test.function_space()
        trial = trial.function_space()

    alpha = test.alpha
    beta = test.beta
    gn = test.gn
    Tv = test.get_orthogonal(domain=(-1, 1))
    i, j, k, l, m = sp.symbols('i,j,k,l,m', integer=True)
    K = test.sympy_stencil(i, k, implicit='k' if implicit is True else False)
    q = sp.degree(measure)
    if measure != 1:
        from shenfun.jacobi.recursions import a, matpow
        from shenfun.utilities import split
        assert sp.sympify(measure).is_polynomial()
        A = sp.S(0)
        for dv in split(measure, expand=True):
            sc = dv['coeff']
            msi = dv['x']
            qi = sp.degree(msi)
            A = A + sc*matpow(a, qi, alpha, beta, l, k, gn)

        if assemble == 'exact':
            B = Tv.sympy_L2_norm_sq(l)*sp.KroneckerDelta(l, m)
        else:
            B = Tv.sympy_l2_norm_sq(l)*sp.KroneckerDelta(l, m)
        S = trial[0].sympy_stencil(j, m, implicit='s' if implicit is True else False)
        A = K * A * B * S
        M = sp.S(0)
        nd = test.N-test.dim()
        for kk in range(-nd, nd+1):
            M1 = A.subs(k, i + kk)
            for ll in range(-q, q+1):
                M2 = M1.subs(l, i + kk + ll)
                for mm in range(-nd, nd+1):
                    M3 = M2.subs(m, i + kk + ll + mm)
                    M += M3
    else:
        if assemble == 'exact':
            B = Tv.sympy_L2_norm_sq(k)*sp.KroneckerDelta(k, l)
        else:
            B = Tv.sympy_l2_norm_sq(k)*sp.KroneckerDelta(k, l)
        S = trial.sympy_stencil(j, l, implicit='s' if implicit is True else False)
        A = K * B * S
        M = sp.S(0)
        nd = test[0].N-test[0].dim()
        for kk in range(-nd, nd+1):
            M1 = A.subs(k, i + kk)
            for ll in range(-nd, nd+1):
                M2 = M1.subs(l, i + kk + ll)
                M += M2
    return M

def sympy2SparseMatrix(sympymat, shape):
    i, j = sp.symbols('i,j', integer=True)
    d = {}
    M, N = shape
    numzerorow = 0
    for k in range(N):
        val = sp.simplify(sympymat.subs(j, i+k))
        if val == 0:
            numzerorow += 1
            if numzerorow >=2:
                break
            continue
        d[k] = np.array([val.subs(i, l) for l in np.arange(min(M, N-k))]).astype(float)
        if k > 0:
            d[-k] = d[k].copy()
        numzerorow = 0
    return SparseMatrix(d, shape)


class SpectralMatDict(dict):
    """Dictionary for inner product matrices

    Matrices are looked up with keys that are one of::

        ((test, k), (trial, l))
        ((test, k), (trial, l), measure)

    where test and trial are classes subclassed from SpectralBase and k and l
    are integers >= 0 that determines how many times the test or trial functions
    should be differentiated. The measure is optional.

    """
    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        c = functools.partial(SpectralMatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        if len(key) == 3:
            matrix = functools.partial(dict.__getitem__(self, key),
                                       measure=key[2])
        else:
            matrix = dict.__getitem__(self, key)
        return matrix
