r"""
This module contains classes for working with sparse matrices.

"""
from __future__ import division
from copy import deepcopy
from collections.abc import Mapping, MutableMapping
from numbers import Number, Integral
import numpy as np
import sympy as sp
from scipy.sparse import bmat, dia_matrix, kron, diags as sp_diags
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from .utilities import integrate_sympy

__all__ = ['SparseMatrix', 'SpectralMatrix', 'extract_diagonal_matrix',
           'extract_bc_matrices', 'check_sanity', 'get_dense_matrix',
           'TPMatrix', 'BlockMatrix', 'BlockMatrices', 'Identity',
           'get_dense_matrix_sympy', 'get_dense_matrix_quadpy']

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
    The dictionary can use a function to generate its values. See, e.g.,
    the ASDSDmat matrix, where diagonals for keys > 2 are functions that return
    the proper diagonal when looked up.

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

    def __init__(self, d, shape, scale=1.0):
        self._storage = dict(d)
        self.shape = shape
        self._diags = dia_matrix((1, 1))
        self.scale = scale
        self._matvec_methods = []

    def matvec(self, v, c, format='dia', axis=0):
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
        axis : int, optional
            The axis over which to take the matrix vector product

        """
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
            if format not in ('csr', 'dia'): # Fallback on 'csr'. Should probably throw warning
                format = 'csr'
            diags = self.diags(format=format)
            P = int(np.prod(v.shape[1:]))
            y = diags.dot(v[:M].reshape(M, P)).squeeze()
            d = tuple([slice(0, m) for m in y.shape])
            c[d] = y.reshape(c[d].shape)

        if axis > 0:
            c = np.moveaxis(c, 0, axis)
            v = np.moveaxis(v, 0, axis)

        return c

    def diags(self, format='dia'):
        """Return a regular sparse matrix of specified format

        Parameters
        ----------
        format : str, optional
            Choice of matrix type (see scipy.sparse.diags)

            - dia - Sparse matrix with DIAgonal storage
            - csr - Compressed sparse row
            - csc - Compressed sparse column

        Note
        ----
        This method returns the matrix scaled by self.scale.

        """
        #if self._diags.shape != self.shape or self._diags.format != format:
        self._diags = sp_diags(list(self.values()), list(self.keys()),
                               shape=self.shape, format=format)
        scale = self.scale
        if isinstance(scale, np.ndarray):
            scale = np.atleast_1d(scale).item()
        self._diags = self._diags*scale

        return self._diags

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

    def __quasi__(self, Q):
        return Q.diags('csc')*self.diags('csc')

    def __eq__(self, a):
        if self.shape != a.shape:
            return False
        if not self.same_keys(a):
            return False
        if (np.abs(self.diags('csr') - a.diags('csr')) >= 2e-8).nnz > 0:
            return False
        if self.scale != a.scale:
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
            return SparseMatrix(deepcopy(dict(self)), self.shape,
                                scale=self.scale*y)
        elif isinstance(y, np.ndarray):
            c = np.empty_like(y)
            c = self.matvec(y, c)
            return c
        elif isinstance(y, SparseMatrix):
            return self.diags('csc')*y.diags('csc')

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
        if self == d:
            if abs(self.scale+d.scale) < 1e-15:
                f = SparseMatrix({0: 0}, self.shape)
            else:
                f = SparseMatrix(deepcopy(dict(self)), self.shape,
                                 self.scale+d.scale)
        else:

            if abs(self.scale) < 1e-15 and abs(d.scale) < 1e-15:
                f = SparseMatrix({0: 0}, self.shape)

            elif abs(self.scale) < 1e-15:
                f = SparseMatrix(deepcopy(dict(d)), d.shape, d.scale)

            else:
                f = SparseMatrix(deepcopy(dict(self)), self.shape, self.scale)
                assert isinstance(d, Mapping)
                for key, val in d.items():
                    if key in f:
                        # Check if symmetric and make copy if necessary
                        if -key in f:
                            if id(f[key]) == id(f[-key]):
                                f[-key] = deepcopy(f[key])
                        f[key] = f[key] + d.scale/self.scale*val
                    else:
                        f[key] = d.scale/f.scale*val

        return f

    def __iadd__(self, d):
        """self.__iadd__(d) <==> self += d"""
        assert isinstance(d, Mapping)
        assert d.shape == self.shape
        #if self == d:
        #    self.scale += d.scale
        if abs(d.scale) < 1e-16:
            pass

        elif abs(self.scale) < 1e-16:
            self.clear()
            for key, val in d.items():
                self[key] = val
            self.scale = d.scale

        else:
            for key, val in d.items():
                if key in self:
                    # Check if symmetric and make copy if necessary
                    #self[key] = self[key]*self.scale
                    if -key in self:
                        if id(self[key]) == id(self[-key]):
                            self[-key] = deepcopy(self[key])
                    self[key] = self[key] + d.scale/self.scale*val
                else:
                    self[key] = d.scale/self.scale*val
        return self

    def __sub__(self, d):
        """Return copy of self.__sub__(d) <==> self-d"""
        assert isinstance(d, Mapping)
        if self == d:
            f = SparseMatrix(deepcopy(dict(self)), self.shape,
                             self.scale-d.scale)

        elif abs(self.scale) < 1e-16:
            f = SparseMatrix(deepcopy(dict(d)), d.shape, -d.scale)

        else:
            f = SparseMatrix(deepcopy(dict(self)), self.shape, self.scale)
            for key, val in d.items():
                if key in f:
                    # Check if symmetric and make copy if necessary
                    if -key in f:
                        if id(f[key]) == id(f[-key]):
                            f[-key] = deepcopy(f[key])
                    f[key] = f[key] - d.scale/self.scale*val
                else:
                    f[key] = -d.scale/self.scale*val

        return f

    def __isub__(self, d):
        """self.__isub__(d) <==> self -= d"""
        assert isinstance(d, Mapping)
        assert d.shape == self.shape
        if self == d:
            self.scale -= d.scale

        elif abs(self.scale) < 1e-16:
            self.clear()
            for key, val in d.items():
                self[key] = val
            self.scale = -d.scale

        else:
            for key, val in d.items():
                if key in self:
                    #self[key] = self[key]*self.scale
                    # Check if symmetric and make copy if necessary
                    if -key in self:
                        if id(self[key]) == id(self[-key]):
                            self[-key] = deepcopy(self[key])
                    self[key] =self[key] - d.scale/self.scale*val
                else:
                    self[key] = -d.scale/self.scale*val
            #self.scale = 1
        return self

    def __neg__(self):
        """self.__neg__() <==> self *= -1"""
        self.scale *= -1
        return self

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
        if abs(self.scale-1) < 1e-8:
            return
        if hasattr(self, '_keyscale'):
            self._keyscale *= self.scale
        else:
            for key, val in self.items():
                self[key] = val*self.scale
        self.scale = 1

    def solve(self, b, u=None, axis=0, use_lu=False):
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
        use_lu : bool, optional
            Look for already computed LU-matrix

        Note
        ----
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
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        if b.ndim == 1:
            if use_lu:
                if b.dtype.char in 'FDG' and self._lu.U.dtype.char in 'fdg':
                    u.real[:] = self._lu.solve(b.real)
                    u.imag[:] = self._lu.solve(b.imag)
                else:
                    u[:] = self._lu.solve(b)
            else:
                u[:] = spsolve(self.diags('csc'), b)
        else:
            N = b.shape[0]
            P = np.prod(b.shape[1:])
            if use_lu:
                if b.dtype.char in 'FDG' and self._lu.U.dtype.char in 'fdg':
                    u.real[:] = self._lu.solve(b.real.reshape((N, P))).reshape(u.shape)
                    u.imag[:] = self._lu.solve(b.imag.reshape((N, P))).reshape(u.shape)
                else:
                    u[:] = self._lu.solve(b.reshape((N, P))).reshape(u.shape)
            else:
                u[:] = spsolve(self.diags('csc'), b.reshape((N, P))).reshape(u.shape)

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)
        return u

    def isdiagonal(self):
        if len(self) == 1:
            if (0 in self):
                return True
        return False

    def isidentity(self):
        if not len(self) == 1:
            return False
        if (0 not in self):
            return False
        d = self[0]
        if np.all(d == 1):
            return True
        return False

    @property
    def issymmetric(self):
        M = self.diags()
        return (abs(M-M.T) > 1e-8).nnz == 0

    def clean_diagonals(self, reltol=1e-8):
        """Eliminate essentially zerovalued diagonals

        Parameters
        ----------
        reltol : number
            Relative tolerance
        """
        a = self * np.ones(self.shape[1])
        relmax = abs(a).max() / self.shape[1]
        list_keys = []
        for key, val in self.items():
            if abs(np.linalg.norm(val))/relmax < reltol:
                list_keys.append(key)
        for key in list_keys:
            del self[key]
        return self


class SpectralMatrix(SparseMatrix):
    r"""Base class for inner product matrices.

    Parameters
    ----------
    d : dict
        Dictionary, where keys are the diagonal offsets and values the
        diagonals
    trial : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
        - :mod:`.fourier.bases`
        - :mod:`.laguerre.bases`
        - :mod:`.hermite.bases`
        - :mod:`.jacobi.bases`

        The int represents the number of times the trial function
        should be differentiated. Representing matrix column.
    test : 2-tuple of (basis, int)
        As trial, but representing matrix row.
    scale : number, optional
        Scale matrix with this number

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

        SD = ShenDirichlet
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
    def __init__(self, d, test, trial, scale=1.0, measure=1):
        assert isinstance(test[1], (int, np.integer))
        assert isinstance(trial[1], (int, np.integer))
        self.testfunction = test
        self.trialfunction = trial
        self.measure = measure
        shape = (test[0].dim(), trial[0].dim())
        if d == {}:
            D = get_dense_matrix(test, trial, measure)[:shape[0], :shape[1]]
            #D = get_denser_matrix(test, trial, measure)[:shape[0], :shape[1]]
            #D = get_dense_matrix_sympy(test, trial, measure)[:shape[0], :shape[1]]
            d = extract_diagonal_matrix(D)
        SparseMatrix.__init__(self, d, shape, scale)
        if shape[0] == shape[1]:
            from shenfun.la import Solve
            self.solver = Solve(self, test[0])

    def matvec(self, v, c, format='csr', axis=0):
        u = self.trialfunction[0]
        ss = [slice(None)]*len(v.shape)
        ss[axis] = u.slice()
        c = super(SpectralMatrix, self).matvec(v[tuple(ss)], c, format=format, axis=axis)
        if self.testfunction[0].use_fixed_gauge:
            ss[axis] = 0
            c[tuple(ss)] = 0
        return c

    def solve(self, b, u=None, axis=0, use_lu=False):
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
        use_lu : bool, optional
            Look for already computed LU-matrix

        Note
        ----
        Vectors may be one- or multidimensional.

        """
        u = self.solver(b, u=u, axis=axis, use_lu=use_lu)
        return u

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

    def simplify_diagonal_matrices(self):
        if self.isdiagonal():
            self.scale = self.scale*self[0]
            self[0] = 1

    def __eq__(self, a):
        if isinstance(a, Number):
            return False
        if not isinstance(a, SparseMatrix):
            return False
        if self.shape != a.shape:
            return False
        if self.get_key() != a.get_key():
            return False
        sl = np.array(list(self.keys()))
        sa = np.array(list(a.keys()))
        if not sl.shape == sa.shape:
            return False
        sl.sort()
        sa.sort()
        if not np.linalg.norm((sl-sa)**2) == 0:
            return False
        if np.linalg.norm(sl[0] - sa[0]) > 1e-8:
            return False
        return True

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        if isinstance(y, Number):
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale*y)
        elif isinstance(y, np.ndarray):
            f = SparseMatrix.__mul__(self, y)
        elif isinstance(y, SparseMatrix):
            f = self.diags('csc')*y.diags('csc')
        return f

    def __div__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        if isinstance(y, Number):
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale/y)
        elif isinstance(y, np.ndarray):
            f = SparseMatrix.__div__(self, y)

        return f

    def __add__(self, y):
        """Return copy of self.__add__(y) <==> self+y"""
        assert isinstance(y, Mapping)
        if self == y:
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale+y.scale)
        else:
            f = SparseMatrix.__add__(self, y)
        return f

    def __iadd__(self, d):
        """self.__iadd__(d) <==> self += d"""
        SparseMatrix.__iadd__(self, d)
        if self == d:
            return self
        else: # downcast
            return SparseMatrix(dict(self), self.shape, self.scale)

    def __sub__(self, y):
        """Return copy of self.__sub__(y) <==> self-y"""
        assert isinstance(y, Mapping)
        if self == y:
            f = SpectralMatrix(deepcopy(dict(self)), self.testfunction,
                               self.trialfunction, self.scale-y.scale)
        else:
            f = SparseMatrix.__sub__(self, y)
        return f

    def __isub__(self, y):
        """self.__isub__(d) <==> self -= y"""
        SparseMatrix.__isub__(self, y)
        if self == y:
            return self
        else: # downcast
            return SparseMatrix(dict(self), self.shape, self.scale)


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
        SparseMatrix.__init__(self, {0:1}, shape, scale)
        self.measure = 1

    def solve(self, b, u=None, axis=0):
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b
        u *= (1/self.scale)
        return u

def BlockMatrices(tpmats):
    """Return two instances of the :class:`.BlockMatrix` class.

    Parameters
    ----------
    tpmats : sequence of :class:`.TPMatrix`'es
        There should be both boundary matrices from inhomogeneous Dirichlet
        or Neumann conditions, as well as regular matrices.

    Note
    ----
    Use :class:`.BlockMatrix` directly if you do not have any inhomogeneous
    boundary conditions.
    """
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
    The tensor product matrices must be either boundary
    matrices or regular matrices, not both. If your problem contains
    inhomogeneous boundary conditions, then create two BlockMatrices,
    one for the implicit terms and one for the boundary terms. To this
    end you can use :class:`.BlockMatrices`.

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
    that are the Cartesian product of these bases

    .. math::

        TD &= K0 \times K1 \times SD \\
        TT &= K0 \times K1 \times ST

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

    This matrix is assemble below

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
    def __init__(self, tpmats, extract_bc_mats=True):
        assert isinstance(tpmats, (list, tuple))

        tpmats = [tpmats] if not isinstance(tpmats[0], (list, tuple)) else tpmats
        self.mixedbase = mixedbase = tpmats[0][0].mixedbase
        self.dims = dims = mixedbase.num_components()
        self.mats = np.zeros((dims, dims), dtype=int).tolist()
        tps = mixedbase.flatten() if hasattr(mixedbase, 'flatten') else [mixedbase]
        offset = [np.zeros(tps[0].dimensions, dtype=int)]
        for i, tp in enumerate(tps):
            dims = tp.dim() if not hasattr(tp, 'dims') else tp.dims() # 1D basis does not have dims()
            offset.append(np.array(dims + offset[i]))
        self.offset = offset
        self.global_shape = self.offset[-1]
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
        """Return flattened list of matrices in self"""
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

    def matvec(self, v, c):
        """Compute matrix vector product

            c = self * v

        Parameters
        ----------
        v : :class:`.Function`
        c : :class:`.Function`

        Returns
        -------
        c : :class:`.Function`

        """
        assert v.function_space() == self.mixedbase
        assert c.function_space() == self.mixedbase
        c.v.fill(0)
        z = np.zeros_like(c.v[0])
        for i, mi in enumerate(self.mats):
            for j, mij in enumerate(mi):
                if isinstance(mij, Number):
                    if abs(mij) > 1e-8:
                        c.v[i] += mij*v.v[j]
                else:
                    for m in mij:
                        z.fill(0)
                        z = m.matvec(v.v[j], z)
                        c.v[i] += z
        return c

    def __getitem__(self, ij):
        return self.mats[ij[0]][ij[1]]

    def get_offset(self, i, axis=0):
        return self.offset[i][axis]

    def diags(self, it=(0,), format='csr'):
        """Return global block matrix in scipy sparse format

        For multidimensional forms the returned matrix is constructed for
        given indices in the periodic directions.

        Parameters
        ----------
        it : n-tuple of ints
            where n is dimensions. These are the indices into the scale arrays
            of the TPMatrices in various blocks. Should be zero along the non-
            periodic direction.
        format : str
            The format of the returned matrix. See `Scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_

        """
        from .spectralbase import MixedFunctionSpace
        bm = []
        for mi in self.mats:
            bm.append([])
            for mij in mi:
                if isinstance(mij, Number):
                    bm[-1].append(None)
                else:
                    m = mij[0]
                    if isinstance(self.mixedbase, MixedFunctionSpace):
                        d = m.diags(format)
                        for mj in mij[1:]:
                            d = d + mj.diags(format)
                    elif len(m.naxes) == 2:
                        # 2 non-periodic directions
                        assert len(m.mats) == 2, "Only implemented without periodic directions"
                        d = m.scale.item()*kron(m.mats[0].diags(format), m.mats[1].diags(format))
                        for mj in mij[1:]:
                            d = d + mj.scale.item()*kron(mj.mats[0].diags(format), mj.mats[1].diags(format))
                    else:
                        iit = np.where(np.array(m.scale.shape) == 1, 0, it) # if shape is 1 use index 0, else use given index (shape=1 means the scale is constant in that direction)
                        sc = m.scale[tuple(iit)]
                        d = sc*m.pmat.diags(format)
                        for mj in mij[1:]:
                            iit = np.where(np.array(mj.scale.shape) == 1, 0, it)
                            sc = mj.scale[tuple(iit)]
                            d = d + sc*mj.pmat.diags(format)
                    bm[-1].append(d)
        return bmat(bm, format=format)

    def solve(self, b, u=None, constraints=(), return_system=False, Alu=None, BM=None):
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

        Other Parameters
        ----------------
        return_system : bool, optional
            If True then return the assembled block matrix as well as the
            solution in a 2-tuple (solution, matrix). This is helpful for
            repeated solves, because the returned matrix may then be
            factorized once and reused.
            Only for non-periodic problems

        Alu : pre-factorized matrix, optional
            Computed with Alu = splu(self), where self is the assembled block
            matrix. Only for non-periodic problems.

        """
        from .forms.arguments import Function
        import scipy.sparse as sp
        space = b.function_space()
        if u is None:
            u = Function(space)
        else:
            assert u.shape == b.shape

        if BM: # Add contribution to right hand side due to inhomogeneous boundary conditions
            u.set_boundary_dofs()
            w0 = np.zeros_like(u)
            b -= BM.matvec(u, w0)

        tpmat = self.get_mats(True)
        axis = tpmat.naxes[0] if isinstance(tpmat, TPMatrix) else 0
        tp = space.flatten() if hasattr(space, 'flatten') else [space]
        nvars = b.shape[0] if len(b.shape)>space.dimensions else 1
        u = u.reshape(1, *u.shape) if nvars == 1 else u
        b = b.reshape(1, *b.shape) if nvars == 1 else b
        for con in constraints:
            assert len(con) == 3
            assert isinstance(con[0], Integral)
            assert isinstance(con[1], Integral)
            assert isinstance(con[2], Number)
        N = self.global_shape[axis]
        gi = np.zeros(N, dtype=b.dtype)
        go = np.zeros(N, dtype=b.dtype)
        if space.dimensions == 1:
            s = [0, 0]
            Ai = self.diags((0,))
            for k in range(nvars):
                s[0] = k
                s[1] = tp[k].slice()
                gi[self.offset[k][axis]:self.offset[k+1][axis]] = b[tuple(s)]
            if Alu is not None:
                go[:] = Alu.solve(gi)
            else:
                go[:] = sp.linalg.spsolve(Ai, gi)
            for k in range(nvars):
                s[0] = k
                s[1] = tp[k].slice()
                u[tuple(s)] = go[self.offset[k][axis]:self.offset[k+1][axis]]
            if return_system:
                return u, Ai

        elif space.dimensions == 2:
            if len(tpmat.naxes) == 2: # 2 non-periodic axes
                s = [0, 0, 0]
                if Alu is None:
                    Ai = self.diags(format='csr')
                gi = np.zeros(space.dim(), dtype=b.dtype)
                go = np.zeros(space.dim(), dtype=b.dtype)
                start = 0
                for k in range(nvars):
                    s[0] = k
                    s[1] = tp[k].bases[0].slice()
                    s[2] = tp[k].bases[1].slice()
                    gi[start:(start+tp[k].dim())] = b[tuple(s)].ravel()
                    start += tp[k].dim()
                for con in constraints:
                    dim = 0
                    for i in range(con[0]):
                        dim += tp[i].dim()
                    if Alu is None:
                        Ai, gi = self.apply_constraint(Ai, gi, dim, 0, con)
                    else:
                        gi[dim] = con[2]
                if Alu is not None:
                    go[:] = Alu.solve(gi)
                else:
                    go[:] = sp.linalg.spsolve(Ai, gi)
                start = 0
                for k in range(nvars):
                    s[0] = k
                    s[1] = tp[k].bases[0].slice()
                    s[2] = tp[k].bases[1].slice()
                    u[tuple(s)] = go[start:(start+tp[k].dim())].reshape((1, tp[k].bases[0].dim(), tp[k].bases[1].dim()))
                    start += tp[k].dim()
                if return_system:
                    return u, Ai
            else:
                s = [0]*3
                ii, jj = {0:(2, 1), 1:(1, 2)}[axis]
                d0 = [0, 0]
                for i in range(b.shape[ii]):
                    d0[(axis+1)%2] = i
                    Ai = self.diags(d0)
                    s[ii] = i
                    for k in range(nvars):
                        s[0] = k
                        s[jj] = tp[k].bases[axis].slice()
                        gi[self.offset[k][axis]:self.offset[k+1][axis]] = b[tuple(s)]
                    for con in constraints:
                        Ai, gi = self.apply_constraint(Ai, gi, self.offset[con[0]][axis], i, con)
                    go[:] = sp.linalg.spsolve(Ai, gi)
                    for k in range(nvars):
                        s[0] = k
                        s[jj] = tp[k].bases[axis].slice()
                        u[tuple(s)] = go[self.offset[k][axis]:self.offset[k+1][axis]]

        elif space.dimensions == 3:
            s = [0]*4
            ii, jj = {0:(2, 3), 1:(1, 3), 2:(1, 2)}[axis]
            d0 = [0, 0, 0]
            for i in range(b.shape[ii]):
                for j in range(b.shape[jj]):
                    d0[ii-1], d0[jj-1] = i, j
                    Ai = self.diags(d0)
                    s[ii], s[jj] = i, j
                    for k in range(nvars):
                        s[0] = k
                        s[axis+1] = tp[k].bases[axis].slice()
                        gi[self.offset[k][axis]:self.offset[k+1][axis]] = b[tuple(s)]
                    for con in constraints:
                        Ai, gi = self.apply_constraint(Ai, gi, self.offset[con[0]][axis], (i, j), con)
                    go[:] = sp.linalg.spsolve(Ai, gi)
                    for k in range(nvars):
                        s[0] = k
                        s[axis+1] = tp[k].bases[axis].slice()
                        u[tuple(s)] = go[self.offset[k][axis]:self.offset[k+1][axis]]

        u = u.reshape(u.shape[1:]) if nvars == 1 else u
        b = b.reshape(b.shape[1:]) if nvars == 1 else b
        return u

    @staticmethod
    def apply_constraint(A, b, offset, i, constraint):
        if constraint is None or comm.Get_rank() > 0:
            return A, b

        if isinstance(i, int):
            if i > 0:
                return A, b

        if isinstance(i, tuple):
            if np.sum(np.array(i)) > 0:
                return A, b

        row = offset + constraint[1]
        #print('Applying constraint row %d con (%d, %d, %2.5f)' %(row, *constraint))

        assert isinstance(constraint, tuple)
        assert len(constraint) == 3
        val = constraint[2]
        b[row] = val
        r = A.getrow(row).nonzero()
        #rp = A.getrow(row-1).nonzero()
        A[(row, r[1])] = 0
        #A[(row, rp[1])] = 0
        #A[row, offset] = 1
        A[row, row] = 1
        #A[offset, row] = 1
        #A[row-1, row-1] = 1

        return A, b

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
    mixedbase : :class:`.CompositeSpace`, optional
         Instance of the base space
    """
    def __init__(self, mats, testspace, trialspace, scale=1.0, global_index=None, mixedbase=None):
        assert isinstance(mats, (list, tuple))
        assert len(mats) == len(testspace)
        self.mats = mats
        self.space = testspace
        self.trialspace = trialspace
        self.scale = scale
        self.pmat = 1
        self.naxes = []
        self.global_index = global_index
        self.mixedbase = mixedbase
        self._issimplified = False

    def simplify_diagonal_matrices(self):
        self.naxes = []
        for axis, mat in enumerate(self.mats):
            if not mat:
                continue

            #if mat.isdiagonal():
            if axis not in self.space.get_nondiagonal_axes():
                if self.dimensions == 1: # Don't bother with the 1D case
                    continue
                else:
                    d = mat[0]    # get diagoal
                    if np.ndim(d):
                        d = self.space[axis].broadcast_to_ndims(d)
                        d = d*mat.scale
                    self.scale = self.scale*d
                    self.mats[axis] = Identity(mat.shape)

            else:
                self.naxes.append(axis)

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

    def solve(self, b, u=None):

        if len(self.naxes) == 0:
            sl = tuple([s.slice() for s in self.trialspace.bases])
            d = self.scale
            with np.errstate(divide='ignore'):
                d = 1./self.scale
            d = np.where(np.isfinite(d), d, 0)
            if u is None:
                from .forms.arguments import Function
                u = Function(self.space)

            u[sl] = b[sl] * d[sl]
            return u

        elif len(self.naxes) == 1:
            axis = self.naxes[0]
            u = self.pmat.solve(b, u=u, axis=axis)
            with np.errstate(divide='ignore'):
                u /= self.scale
            u[:] = np.where(np.isfinite(u), u, 0)
            return u

        elif len(self.naxes) == 2:
            from shenfun.la import SolverGeneric2ND
            H = SolverGeneric2ND([self])
            u = H(b, u)
            return u

    def matvec(self, v, c):
        c.fill(0)
        if len(self.naxes) == 0:
            c[:] = self.scale*v
        elif len(self.naxes) == 1:
            axis = self.naxes[0]
            rank = v.rank if hasattr(v, 'rank') else 0
            if rank == 0:
                c = self.pmat.matvec(v, c, axis=axis)
            else:
                c = self.pmat.matvec(v[self.global_index[1]], c, axis=axis)
            c = c*self.scale
        elif len(self.naxes) == 2:
            # 2 non-periodic directions (may be non-aligned in second axis, hence transfers)
            npaxes = deepcopy(self.naxes)
            pencilA = self.space.forward.output_pencil
            subcomms = [s.Get_size() for s in pencilA.subcomm]
            axis = pencilA.axis
            assert subcomms[axis] == 1
            npaxes.remove(axis)
            second_axis = npaxes[0]
            pencilB = pencilA.pencil(second_axis)
            transAB = pencilA.transfer(pencilB, c.dtype.char)
            cB = np.zeros(transAB.subshapeB, dtype=c.dtype)
            cC = np.zeros(transAB.subshapeB, dtype=c.dtype)
            bb = self.mats[axis]
            c = bb.matvec(v, c, axis=axis)
            # align in second non-periodic axis
            transAB.forward(c, cB)
            bb = self.mats[second_axis]
            cC = bb.matvec(cB, cC, axis=second_axis)
            transAB.backward(cC, c)
            c *= self.scale
        return c

    def get_key(self):
        """Return key of the one nondiagonal matrix in the TPMatrix

        Note
        ----
        Raises an error of there are more than one single nondiagonal matrix
        in TPMatrix.
        """
        naxis = self.space.get_nondiagonal_axes()
        assert len(naxis) == 1
        return self.mats[naxis[0]].get_key()

    def isidentity(self):
        return np.all([m.isidentity() for m in self.mats])

    def isdiagonal(self):
        return np.all([m.isdiagonal() for m in self.mats])

    def is_bc_matrix(self):
        for m in self.mats:
            if hasattr(m, 'trialfunction'):
                if m.trialfunction[0].boundary_condition() == 'Apply':
                    return True
        return False

    @property
    def dimensions(self):
        """Return dimension of TPMatrix"""
        return len(self.mats)

    def __mul__(self, a):
        """Returns copy of self.__mul__(a) <==> self*a"""
        if isinstance(a, Number):
            TPMatrix(self.mats, self.space, self.trialspace, self.scale*a,
                     self.global_index, self.mixedbase)

        elif isinstance(a, np.ndarray):
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
                            self.global_index, self.mixedbase)
        elif isinstance(a, np.ndarray):
            b = np.zeros_like(a)
            b = self.solve(a, b)
            return b
        else:
            raise NotImplementedError

    def __neg__(self):
        """self.__neg__() <==> self *= -1"""
        self.scale *= -1
        return self

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
                        self.global_index, self.mixedbase)

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
                        self.global_index, self.mixedbase)

    def __isub__(self, a):
        """self.__isub__(a) <==> self -= a"""
        assert isinstance(a, TPMatrix)
        assert self == a
        self.scale = self.scale - a.scale
        return self

    def diags(self, format='csr'):
        if self.dimensions == 2:
            return kron(self.mats[0].diags(format=format), self.mats[1].diags(format=format))
        elif self.dimensions == 3:
            return kron(self.mats[0].diags(format=format), kron(self.mats[1].diags(format=format), self.mats[2].diags(format=format)))


def check_sanity(A, test, trial, measure=1):
    """Sanity check for matrix.

    Test that automatically created matrix agrees with overloaded one

    Parameters
    ----------
    A : matrix
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
        - :mod:`.fourier.bases`
        - :mod:`.laguerre.bases`
        - :mod:`.hermite.bases`
        - :mod:`.jacobi.bases`

        The int represents the number of times the test function
        should be differentiated. Representing matrix row.
    trial : 2-tuple of (basis, int)
        As test, but representing matrix column.
    measure : sympy function of coordinate, optional
    """
    N, M = A.shape
    if measure == 1:
        D = get_dense_matrix(test, trial, measure)[:N, :M]
    else:
        D = get_denser_matrix(test, trial, measure)
    Dsp = extract_diagonal_matrix(D)
    for key, val in A.items():
        assert np.allclose(val*A.scale, Dsp[key])

def get_dense_matrix(test, trial, measure=1):
    """Return dense matrix automatically computed from basis

    Parameters
    ----------
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
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
    """
    K0 = test[0].slice().stop - test[0].slice().start
    K1 = trial[0].slice().stop - trial[0].slice().start
    N = test[0].N
    x = test[0].mpmath_points_and_weights(N, map_true_domain=False)[0]
    ws = test[0].get_measured_weights(N, measure)
    v = test[0].evaluate_basis_derivative_all(x=x, k=test[1])[:, :K0]
    u = trial[0].evaluate_basis_derivative_all(x=x, k=trial[1])[:, :K1]
    A = np.dot(np.conj(v.T)*ws[np.newaxis, :], u)
    if A.dtype.char in 'FDG':
        if np.linalg.norm(A.real) / np.linalg.norm(A.imag) > 1e14:
            A = A.real.copy()
    return A

def get_denser_matrix(test, trial, measure=1):
    """Return dense matrix automatically computed from basis

    Use slightly more quadrature points than usual N

    Parameters
    ----------
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
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
    """
    test2 = test[0].get_refined((test[0].N*3)//2)
    trial2 = trial[0].get_refined((trial[0].N*3)//2)

    K0 = test[0].slice().stop - test[0].slice().start
    K1 = trial[0].slice().stop - trial[0].slice().start
    N = test2.N
    x = test2.mpmath_points_and_weights(N, map_true_domain=False)[0]
    ws = test2.get_measured_weights(N, measure)
    v = test[0].evaluate_basis_derivative_all(x=x, k=test[1])[:, :K0]
    u = trial[0].evaluate_basis_derivative_all(x=x, k=trial[1])[:, :K1]
    return np.dot(np.conj(v.T)*ws[np.newaxis, :], u)

def extract_diagonal_matrix(M, abstol=1e-10, reltol=1e-10):
    """Return SparseMatrix version of dense matrix ``M``

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
    dtype = float if M.dtype == 'O' else M.dtype # For mpf object
    for i in range(M.shape[1]):
        u = M.diagonal(i).copy()
        if abs(u).max() > abstol and abs(u).max()/relmax > reltol:
            d[i] = np.array(u, dtype=dtype)

    for i in range(1, M.shape[0]):
        l = M.diagonal(-i).copy()
        if abs(l).max() > abstol and abs(l).max()/relmax > reltol:
            d[-i] = np.array(l, dtype=dtype)

    return SparseMatrix(d, M.shape)

def extract_bc_matrices(mats):
    """Extract boundary matrices from list of ``mats``

    Parameters
    ----------
    mats : list of list of :class:`.TPMatrix`es

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
            if isinstance(b, SparseMatrix):
                if b.trialfunction[0].boundary_condition() == 'Apply':
                    bc_mats.append(b)
                    a.remove(b)
            elif isinstance(b, TPMatrix):
                if b.is_bc_matrix():
                    bc_mats.append(b)
                    a.remove(b)
    return bc_mats

def get_dense_matrix_sympy(test, trial, measure=1):
    """Return dense matrix automatically computed from basis

    Parameters
    ----------
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
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
    """
    N = test[0].slice().stop - test[0].slice().start
    M = trial[0].slice().stop - trial[0].slice().start
    V = np.zeros((N, M), dtype=test[0].forward.output_array.dtype)
    x = sp.Symbol('x', real=True)

    if not measure == 1:
        if isinstance(measure, sp.Expr):
            s = measure.free_symbols
            assert len(s) == 1
            x = s.pop()
            xm = test[0].map_true_domain(x)
            measure = measure.subs(x, xm)
        else:
            assert isinstance(measure, Number)

    # Weight of weighted space
    measure *= test[0].weight()

    for i in range(test[0].slice().start, test[0].slice().stop):
        pi = np.conj(test[0].sympy_basis(i, x=x))
        for j in range(trial[0].slice().start, trial[0].slice().stop):
            pj = trial[0].sympy_basis(j, x=x)
            integrand = sp.simplify(measure*pi.diff(x, test[1])*pj.diff(x, trial[1]))
            V[i, j] = integrate_sympy(integrand,
                                      (x, test[0].sympy_reference_domain()[0], test[0].sympy_reference_domain()[1]))

    return V

def get_dense_matrix_quadpy(test, trial, measure=1):
    """Return dense matrix automatically computed from basis

    Using quadpy to compute the integral adaptively with high accuracy.
    This should be equivalent to integrating analytically with sympy,
    as long as the integrand is smooth enough and the integral can be
    found with quadrature.

    Parameters
    ----------
    test : 2-tuple of (basis, int)
        The basis is an instance of a class for one of the bases in

        - :mod:`.legendre.bases`
        - :mod:`.chebyshev.bases`
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
    """
    import quadpy
    N = test[0].slice().stop - test[0].slice().start
    M = trial[0].slice().stop - trial[0].slice().start
    V = np.zeros((N, M), dtype=test[0].forward.output_array.dtype)
    x = sp.Symbol('x', real=True)

    if not measure == 1:
        if isinstance(measure, sp.Expr):
            s = measure.free_symbols
            assert len(s) == 1
            x = s.pop()
            xm = test[0].map_true_domain(x)
            measure = measure.subs(x, xm)
        else:
            assert isinstance(measure, Number)

    # Weight of weighted space
    measure *= test[0].weight()

    for i in range(test[0].slice().start, test[0].slice().stop):
        pi = np.conj(test[0].sympy_basis(i, x=x))
        for j in range(trial[0].slice().start, trial[0].slice().stop):
            pj = trial[0].sympy_basis(j, x=x)
            integrand = sp.simplify(measure*pi.diff(x, test[1])*pj.diff(x, trial[1]))
            if not integrand == 0:
                V[i, j] = quadpy.c1.integrate_adaptive(sp.lambdify(x, integrand),
                                                       test[0].reference_domain())[0]

    return V
