from numbers import Number
import numpy as np

__all__ = ('Expr', 'BasisFunction', 'TestFunction', 'TrialFunction', 'Function',
           'Array', 'Basis')

def Basis(N, family='Fourier', bc=None, dtype='d', quad=None, domain=None,
          scaled=None, padding_factor=1.0, dealias_direct=False):
    """Return basis for one dimension

    Parameters
    ----------

        N : int
            Number of quadrature points
        family : str, optional
                 Choose one of (``Chebyshev``, ``C``, ``Legendre``, ``L``,
                 ``Fourier``, ``F``), where ``C``, ``L`` and ``F`` are short-
                 forms
        bc : str or two-tuple, optional
             Choose one of

             - two-tuple (a, b) - Dirichlet boundary condition with
               :math:`v(-1)=a` and :math:`v(1)=b`. For solving Poisson equation.
             - Dirichlet - Homogeneous Dirichlet
             - Neumann - Homogeneous Neumann
             - Biharmonic - Homogeneous Dirichlet and Neumann at both ends
        dtype : str or np.dtype, optional
                The datatype of physical space (input to forward transforms)
        quad : str, optional
               Type of quadrature

               * For family=Chebyshev:

                 - GL - Chebyshev-Gauss-Lobatto
                 - GC - Chebyshev-Gauss

               * For family=Legendre:

                 - LG - Legendre-Gauss
                 - GL - Legendre-Gauss-Lobatto
        domain : two-tuple of floats, optional
                 The computational domain
        scaled : bool
                 Whether to use scaled basis (only Legendre)
        padding_factor : float, optional
                         For padding backward transform (for dealiasing, and
                         only for Fourier)
        dealias_direct : bool, optional
                         Use 2/3-rule dealiasing (only Fourier)

    Examples
    --------
    >>> from shenfun import Basis
    >>> F0 = Basis(16, 'F')
    >>> C1 = Basis(32, 'C', quad='GC')

    """
    par = {}
    if domain is not None:
        par['domain'] = domain
    if family.lower() in ('fourier', 'f'):
        from shenfun import fourier
        par.update({'padding_factor': padding_factor,
                    'dealias_direct': dealias_direct})
        if np.dtype(dtype).char in 'FDG':
            B = fourier.bases.C2CBasis
        else:
            B = fourier.bases.R2CBasis
        return B(N, **par)
    elif family.lower() in ('chebyshev', 'c'):
        from shenfun import chebyshev
        if quad is not None:
            assert quad in ('GC', 'GL')
            par['quad'] = quad

        if bc is None:
            B = chebyshev.bases.Basis

        elif isinstance(bc, tuple):
            assert len(bc) == 2
            par['bc'] = bc
            B = chebyshev.bases.ShenDirichletBasis

        elif isinstance(bc, str):
            if bc.lower() == 'dirichlet':
                B = chebyshev.bases.ShenDirichletBasis
            elif bc.lower() == 'neumann':
                B = chebyshev.bases.ShenNeumannBasis
            elif bc.lower() == 'biharmonic':
                B = chebyshev.bases.ShenBiharmonicBasis

        else:
            raise NotImplementedError

        return B(N, **par)

    elif family.lower() in ('legendre', 'l'):
        from shenfun import legendre
        if quad is not None:
            assert quad in ('LG', 'GL')
            par['quad'] = quad

        if scaled is not None:
            assert isinstance(scaled, bool)
            par['scaled'] = scaled

        if bc is None:
            B = legendre.bases.Basis

        elif isinstance(bc, tuple):
            assert len(bc) == 2
            par['bc'] = bc
            B = legendre.bases.ShenDirichletBasis

        elif isinstance(bc, str):
            if bc.lower() == 'dirichlet':
                B = legendre.bases.ShenDirichletBasis
            elif bc.lower() == 'neumann':
                B = legendre.bases.ShenNeumannBasis
            elif bc.lower() == 'biharmonic':
                B = legendre.bases.ShenBiharmonicBasis

        else:
            raise NotImplementedError

        return B(N, **par)

    else:
        raise NotImplementedError


class Expr(object):
    r"""
    Class for spectral Galerkin forms

    An Expression that is linear in :class:`.TestFunction`,
    :class:`.TrialFunction` or :class:`.Function`. Exprs are used as input
    to :func:`.inner` or :func:`.project`.

    Parameters
    ----------
    basis : :class:`.BasisFunction`
        :class:`.TestFunction`, :class:`.TrialFunction` or :class:`.Function`
    terms : Numpy array of ndim = 3
        Describes operations performed in Expr

        - Index 0: Vector component. If Expr is rank = 0, then terms[0] = 1.
          For vectors it equals dim

        - Index 1: One for each term in the form. For example `div(grad(u))`
          has three terms in 3D:

        .. math::

           \partial^2u/\partial x^2 + \partial^2u/\partial y^2 + \partial^2u/\partial z^2

        - Index 2: The operations stored as an array of length = dim

        The Expr `div(grad(u))`, where u is a scalar, is as such represented
        as an array of shape (1, 3, 3), 1 meaning it's a scalar, the first 3
        because the Expr consists of the sum of three terms, and the last 3
        because it is 3D. The entire representation is::

           array([[[2, 0, 0],
                   [0, 2, 0],
                   [0, 0, 2]]])

        where the first [2, 0, 0] term has two derivatives in first direction
        and none in the others, the second [0, 2, 0] has two derivatives in
        second direction, etc.

    scales :  Numpy array of shape == terms.shape[:2]
        Representing a scalar multiply of each inner product

    indices : Numpy array of shape == terms.shape[:2]
        Index into VectorTensorProductSpace. Only for vector coefficients

    Examples
    --------
    >>> from shenfun import *
    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD
    >>> C0 = Basis(16, 'F', dtype='D')
    >>> C1 = Basis(16, 'F', dtype='D')
    >>> R0 = Basis(16, 'F', dtype='d')
    >>> T = TensorProductSpace(comm, (C0, C1, R0))
    >>> v = TestFunction(T)
    >>> e = div(grad(v))
    >>> e.terms()
    array([[[2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]]])
    >>> e2 = grad(v)
    >>> e2.terms()
    array([[[1, 0, 0]],
    <BLANKLINE>
           [[0, 1, 0]],
    <BLANKLINE>
           [[0, 0, 1]]])

    Note that `e2` in the example has shape (3, 1, 3). The first 3 because it
    is a vector, the 1 because each vector item contains one term, and the
    final 3 since it is a 3-dimensional tensor product space.
    """

    def __init__(self, basis, terms=None, scales=None, indices=None):
        #assert isinstance(basis, BasisFunction)
        self._basis = basis
        self._terms = terms
        self._scales = scales
        self._indices = indices
        ndim = self.function_space().ndim()
        if terms is None:
            self._terms = np.zeros((self.function_space().num_components(), 1, ndim),
                                   dtype=np.int)
        if scales is None:
            self._scales = np.ones((self.function_space().num_components(), 1))

        if indices is None:
            self._indices = np.arange(self.function_space().num_components())[:, np.newaxis]
            if basis.index() is not None:
                if isinstance(basis, BasisFunction) and self._indices.shape == (1, 1):
                    self._indices[0, 0] = basis.index()

        assert np.prod(self._scales.shape) == self.num_terms()*self.num_components()

    def basis(self):
        """Return basis of Expr"""
        return self._basis

    @property
    def base(self):
        """Return base array used in Expr"""
        # base is always the non-sliced parent array
        if isinstance(self._basis, np.ndarray):
            return self._basis if self._basis.base is None else self._basis.base
        return self._basis

    def function_space(self):
        """Return function space of basis in Expr"""
        return self._basis.function_space()

    def terms(self):
        """Return terms of Expr"""
        return self._terms

    def scales(self):
        """Return scales of Expr"""
        return self._scales

    @property
    def argument(self):
        """Return argument of Expr's basis"""
        return self._basis.argument

    def expr_rank(self):
        """Return rank of Expr"""
        return 1 if self._terms.shape[0] == 1 else 2

    def rank(self):
        """Return rank of Expr's basis"""
        return self._basis.rank()

    def indices(self):
        """Return indices of Expr"""
        return self._indices

    def num_components(self):
        """Return number of components in Expr"""
        return self._terms.shape[0]

    def num_terms(self):
        """Return number of terms in Expr"""
        return self._terms.shape[1]

    def dim(self):
        """Return ndim of Expr"""
        return self._terms.shape[2]

    def __getitem__(self, i):
        #assert self.num_components() == self.dim()
        basis = self._basis
        if self.rank() == 2:
            basis = self._basis[i]
        return Expr(basis,
                    self._terms[i][np.newaxis, :, :],
                    self._scales[i][np.newaxis, :],
                    self._indices[i][np.newaxis, :])

    def __mul__(self, a):
        if self.expr_rank() == 1:
            assert isinstance(a, Number)
            sc = self.scales().copy()*a
        elif self.expr_rank() == 2:
            sc = self.scales().copy()
            if isinstance(a, tuple):
                assert len(a) == self.dim()
                for i in range(self.dim()):
                    assert isinstance(a[i], Number)
                    sc[i] = sc[i]*a[i]

            elif isinstance(a, Number):
                sc *= a

            else:
                raise NotImplementedError
            #elif isinstance(a, np.ndarray):
                #assert len(a) == self.dim() or len(a) == 1
                #sc *= a

        return Expr(self._basis, self._terms.copy(), sc, self._indices.copy())

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        sc = self.scales()
        if self.expr_rank() == 1:
            assert isinstance(a, Number)
            sc *= a
        elif self.expr_rank() == 2:
            if isinstance(a, tuple):
                assert len(a) == self.dim()
                for i in range(self.dim()):
                    assert isinstance(a[i], Number)
                    sc[i] = sc[i]*a[i]

            elif isinstance(a, Number):
                sc *= a

            else:
                raise NotImplementedError
            #elif isinstance(a, np.ndarray):
                #assert len(a) == self.dim() or len(a) == 1
                #sc *= a

        return self

    def __add__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        return Expr(self._basis,
                    np.concatenate((self.terms(), a.terms()), axis=1),
                    np.concatenate((self.scales(), a.scales()), axis=1),
                    np.concatenate((self.indices(), a.indices()), axis=1))

    def __iadd__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        self._terms = np.concatenate((self.terms(), a.terms()), axis=1)
        self._scales = np.concatenate((self.scales(), a.scales()), axis=1)
        self._indices = np.concatenate((self.indices(), a.indices()), axis=1)
        return self

    def __sub__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        return Expr(self._basis,
                    np.concatenate((self.terms(), a.terms()), axis=1),
                    np.concatenate((self.scales(), -a.scales()), axis=1),
                    np.concatenate((self.indices(), a.indices()), axis=1))

    def __isub__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        self._terms = np.concatenate((self.terms(), a.terms()), axis=1)
        self._scales = np.concatenate((self.scales(), -a.scales()), axis=1)
        self._indices = np.concatenate((self.indices(), a.indices()), axis=1)
        return self

    def __neg__(self):
        return Expr(self.basis(), self.terms().copy(), -self.scales().copy(),
                    self.indices().copy())


class BasisFunction(object):
    """Base class for arguments to shenfun's Exprs

    Parameters
    ----------
    space: TensorProductSpace
    index: int
        Component of basis with rank > 1
    """

    def __init__(self, space, index=0):
        self._space = space
        self._index = index

    def rank(self):
        """Return rank of basis"""
        return self.function_space().rank()

    def expr_rank(self):
        """Return rank of expression involving basis"""
        return self.function_space().rank()

    def function_space(self):
        """Return function space of BasisFunction"""
        return self._space

    @property
    def argument(self):
        """Return argument of basis"""
        raise NotImplementedError

    def num_components(self):
        """Return number of components in basis"""
        return self.function_space().num_components()

    def ndim(self):
        """Return dimensions of function space"""
        return self.function_space().ndim()

    def index(self):
        """Return index into vector of rank 2"""
        return None

    def __getitem__(self, i):
        assert self.rank() == 2
        t0 = BasisFunction(self._space[i], i)
        return t0

    def __mul__(self, a):
        b = Expr(self)
        return b*a

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        raise RuntimeError

    def __add__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        b = Expr(self)
        return b+a

    def __iadd__(self, a):
        raise RuntimeError

    def __sub__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        b = Expr(self)
        return b-a

    def __isub__(self, a):
        raise RuntimeError


class TestFunction(BasisFunction):
    """Test function - BasisFunction with argument = 0

    Parameters
    ----------
    space: TensorProductSpace
    index: int, optional
        Component of basis with rank > 1
    """

    def __init__(self, space, index=0):
        BasisFunction.__init__(self, space, index)

    def __getitem__(self, i):
        assert self.rank() == 2
        t0 = TestFunction(self._space[i], index=i)
        return t0

    @property
    def argument(self):
        return 0

class TrialFunction(BasisFunction):
    """Trial function - BasisFunction with argument = 1

    Parameters
    ----------
    space: TensorProductSpace
    index: int, optional
        Component of basis with rank > 1
    """
    def __init__(self, space, index=0):
        BasisFunction.__init__(self, space, index)

    def __getitem__(self, i):
        assert self.rank() == 2
        t0 = TrialFunction(self._space[i], index=i)
        return t0

    @property
    def argument(self):
        return 1

class Function(np.ndarray, BasisFunction):
    r"""
    Spectral Galerkin function for given :class:`.TensorProductSpace` or :func:`.Basis`

    The Function is the product of all 1D basis expansions, that for each
    dimension is defined like

    .. math::

        u(x) = \sum_{k \in \mathcal{K}} \hat{u}_k \psi_k(x),

    where :math:`\psi_k(x)` are the trial functions and
    :math:`\{\hat{u}_k\}_{k\in\mathcal{K}}` are the expansion coefficients.
    Here an index set :math:`\mathcal{K}=0, 1, \ldots, N` is used
    to simplify notation.

    For an M+1-dimensional TensorProductSpace with coordinates
    :math:`x_0, x_1, \ldots, x_M` we get

    .. math::

        u(x_{0}, x_{1}, \ldots, x_{M}) = \sum_{k_0 \in \mathcal{K}_0}\sum_{k_1 \in \mathcal{K}_1} \ldots \sum_{k_M \in \mathcal{K}_M} \hat{u}_{k_0, k_1, \ldots k_M} \psi_{k_0}(x_0) \psi_{k_1}(x_1) \ldots \psi_{k_M}(x_M),

    where :math:`\mathcal{K}_j` is the index set for the wavenumber mesh
    along axis :math:`j`.

    Note that for a Cartesian mesh in 3D it would be natural to use coordinates
    :math:`(x, y, z) = (x_0, x_1, x_2)` and the expansion would be the
    simpler and somewhat more intuitive

    .. math::

        u(x, y, z) = \sum_{l \in \mathcal{K}_0}\sum_{m \in \mathcal{K}_1} \sum_{n \in \mathcal{K}_2} \hat{u}_{l, m, n} \psi_{l}(x) \psi_{m}(y) \psi_{n}(z).

    The Function's values (the Numpy array) represent the :math:`\hat{u}` array.
    The trial functions :math:`\psi` may differ in the different directions.
    They are chosen when creating the TensorProductSpace.

    Parameters
    ----------
    space : :class:`.TensorProductSpace`
    val : int or float
        Value used to initialize array
    buffer : Numpy array or :class:`.Function`
        Must be of correct shape


    .. note:: For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from shenfun import Basis, TensorProductSpace, Function
    >>> K0 = Basis(8, 'F', dtype='D')
    >>> K1 = Basis(8, 'F', dtype='d')
    >>> T = TensorProductSpace(MPI.COMM_WORLD, [K0, K1])
    >>> u = Function(T)
    >>> K2 = Basis(8, 'C', bc=(0, 0))
    >>> T2 = TensorProductSpace(MPI.COMM_WORLD, [K0, K1, K2])
    >>> v = Function(T2)

    """
    # pylint: disable=too-few-public-methods,too-many-arguments
    def __new__(cls, space, forward_output=True, val=0, buffer=None):

        if isinstance(buffer, np.ndarray):
            shape = buffer.shape
            dtype = buffer.dtype

        else:

            shape = space.forward.output_array.shape
            dtype = space.forward.output_array.dtype

            if not space.num_components() == 1:
                shape = (space.num_components(),) + shape

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype,
                                 buffer=buffer)

        if buffer is None:
            obj.fill(val)
        return obj

    def __init__(self, space, val=0, buffer=None):
        #super(Function, self).__init__(space)
        BasisFunction.__init__(self, space)

    def index(self):
        if self.base is None:
            return None

        if self.base.shape == self.shape:
            return None

        data_self = self.ctypes.data
        data_base = self.base.ctypes.data
        itemsize = self.itemsize
        return (data_self - data_base) // (itemsize*np.prod(self.shape))

    @property
    def argument(self):
        return 2

    def function_space(self):
        if self.base is None:
            return self._space
        if self.base.shape == self.shape:
            return self._space
        return self._space[self.index()]

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if hasattr(obj, '_space'):
            self._space = obj._space

    def eval(self, x, output_array=None):
        """Evaluate Function at points

        Parameters
        ----------
        points : float or array of floats
        coefficients : array
            Expansion coefficients
        output_array : array, optional
            Return array, function values at points

        Examples
        --------
        >>> import sympy as sp
        >>> K0 = Basis(9, 'F', dtype='D')
        >>> K1 = Basis(8, 'F', dtype='d')
        >>> T = TensorProductSpace(MPI.COMM_WORLD, [K0, K1], axes=(0, 1))
        >>> X = T.local_mesh()
        >>> x, y = sp.symbols("x,y")
        >>> ue = sp.sin(2*x) + sp.cos(3*y)
        >>> ul = sp.lambdify((x, y), ue, 'numpy')
        >>> ua = Array(T, buffer=ul(*X))
        >>> points = np.random.random((2, 4))
        >>> u = ua.forward()
        >>> u0 = u.eval(points).real
        >>> assert np.allclose(u0, ul(*points))
        """
        return self.function_space().eval(x, self, output_array)

    def backward(self, output_array=None):
        """Return Function evaluated on quadrature mesh"""
        space = self.function_space()
        if output_array is None:
            output_array = Array(space)
        output_array = space.backward(self, output_array)
        return output_array

class Array(np.ndarray):
    r"""
    Numpy array for :class:`.TensorProductSpace`

    The Array is the result of a :class:`.Function` evaluated on its quadrature
    mesh.

    The Function is the product of all 1D basis expansions, that for each
    dimension is defined like

    .. math::

        u(x) = \sum_{k \in \mathcal{K}} \hat{u}_k \psi_k(x),

    where :math:`\psi_k(x)` are the trial functions and
    :math:`\{\hat{u}_k\}_{k\in\mathcal{K}}` are the expansion coefficients.
    Here an index set :math:`\mathcal{K}=0, 1, \ldots, N` is used to
    simplify notation.

    For an M+1-dimensional TensorProductSpace with coordinates
    :math:`x_0, x_1, \ldots, x_M` we get

    .. math::

        u(x_{0}, x_{1}, \ldots, x_{M}) = \sum_{k_0 \in \mathcal{K}_0}\sum_{k_1 \in \mathcal{K}_1} \ldots \sum_{k_M \in \mathcal{K}_M} \hat{u}_{k_0, k_1, \ldots k_M} \psi_{k_0}(x_0) \psi_{k_1}(x_1) \ldots \psi_{k_M}(x_M),

    where :math:`\mathcal{K}_j` is the index set for the wavenumber mesh
    along axis :math:`j`.

    Note that for a Cartesian mesh in 3D it would be natural to use coordinates
    :math:`(x, y, z) = (x_0, x_1, x_2)` and the expansion would be the
    simpler and somewhat more intuitive

    .. math::

        u(x, y, z) = \sum_{l \in \mathcal{K}_0}\sum_{m \in \mathcal{K}_1} \sum_{n \in \mathcal{K}_2} \hat{u}_{l, m, n} \psi_{l}(x) \psi_{m}(y) \psi_{n}(z).

    The Array's values (the Numpy array) represent the left hand side,
    evaluated on the Cartesian quadrature mesh. With this we mean the
    :math:`u(x_i, y_j, z_k)` array, where :math:`\{x_i\}_{i=0}^{N_0}`,
    :math:`\{y_j\}_{j=0}^{N_1}` and :math:`\{z_k\}_{k=0}^{N_2}` represent
    the mesh along the three directions. The quadrature mesh is then

    .. math::

        (x_i, y_j, z_k) \quad \forall \, (i, j, k) \in [0, 1, \ldots, N_0] \times [0, 1, \ldots, N_1] \times [0, 1, \ldots, N_2]

    The entire spectral Galerkin function can be obtained using the
    :class:`.Function` class.

    Parameters
    ----------

    space : :class:`.TensorProductSpace`
    val : int or float
        Value used to initialize array
    buffer : Numpy array or Array
        Must be of correct shape


    .. note:: For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from shenfun import Basis, TensorProductSpace, Function
    >>> K0 = Basis(8, 'F', dtype='D')
    >>> K1 = Basis(8, 'F', dtype='d')
    >>> FFT = TensorProductSpace(MPI.COMM_WORLD, [K0, K1])
    >>> u = Array(FFT)
    """
    # pylint: disable=too-few-public-methods,too-many-arguments
    def __new__(cls, space, val=0, buffer=None):

        shape = space.forward.input_array.shape
        dtype = space.forward.input_array.dtype

        if not space.num_components() == 1:
            shape = (space.num_components(),) + shape

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype,
                                 buffer=buffer)

        obj._space = space
        if buffer is None:
            obj.fill(val)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if hasattr(obj, '_space'):
            self._space = obj._space

    def function_space(self):
        """Return function space of Array"""
        if self.base is None:
            return self._space
        if self.base.shape == self.shape:
            return self._space
        return self._space[self.index()]

    def rank(self):
        """Return rank of basis"""
        return self.function_space().rank()

    def index(self):
        if self.base is None:
            return None

        if self.base.shape == self.shape:
            return None

        data_self = self.ctypes.data
        data_base = self.base.ctypes.data
        itemsize = self.itemsize
        return (data_self - data_base) // (itemsize*np.prod(self.shape))

    @property
    def argument(self):
        """Return argument of basis"""
        return 2

    def forward(self, output_array=None):
        """Return Function used to evaluate Array"""
        space = self.function_space()
        if output_array is None:
            output_array = Function(space)
        output_array = space.forward(self, output_array)
        return output_array
