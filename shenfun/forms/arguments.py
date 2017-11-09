import numpy as np
from numbers import Number

__all__ = ('Expr', 'BasisFunction', 'TestFunction', 'TrialFunction', 'Function',
           'Array')


class Expr(object):
    """Class for spectral Galerkin forms

    An Expr instance is a form that is linear in TestFunction (v), TrialFunction
    (u) or Function (c). The Function is a Numpy array evaluated at quadrature
    points. Inner products are constructed from forms consisting of two Exprs,
    one with a TestFunction and one with either TrialFunction or Function.

    args:
        basis:     BasisFunction instance
                      TestFunction
                      TrialFunction
                      Function

    kwargs:
        terms:     Numpy array of ndim = 3. Describes operations in Expr

                   Index 0: Vector component. If Expr is rank = 0, then
                            terms[0] = 1. For vectors it equals dim

                   Index 1: One for each term in the form. For example
                            div(grad(u)) has three terms in 3D:
                              d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2

                   Index 2: The operations stored as an array of len = dim
                            For example, div(grad(u)) in 3D is represented
                            with the three arrays
                              [[2, 0, 0],
                               [0, 2, 0],
                               [0, 0, 2]]
                            meaning the first term has two derivatives in first
                            direction and none in the others (d^2u/dx^2),
                            the second has two derivatives in second direction,
                            etc.

                   The Expr div(grad(u)), where u is a scalar, is as such
                   represented as an array of shape (1, 3, 3), 1 meaning
                   it's a scalar, the first 3 because the Expr consists of
                   the sum of three terms, and the last 3 because it is 3D:
                            array([[[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 2]]])

        scales:   Representing a scalar multiply of each inner product
                  Numpy array of shape == terms.shape[:2]

        indices:  Index into VectorTensorProductSpace. Only for vector
                  coefficients
                  Numpy array of shape == terms.shape[:2]


    """

    def __init__(self, basis, terms=None, scales=None, indices=None):
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

        assert np.prod(self._scales.shape) == self.num_terms()*self.num_components()

    def basis(self):
        return self._basis

    def function_space(self):
        return self._basis.function_space()

    def terms(self):
        return self._terms

    def scales(self):
        return self._scales

    def argument(self):
        return self._basis.argument()

    def expr_rank(self):
        return 1 if self._terms.shape[0] == 1 else 2

    def rank(self):
        return self._basis.rank()

    def indices(self):
        return self._indices

    def num_components(self):
        return self._terms.shape[0]

    def num_terms(self):
        return self._terms.shape[1]

    def dim(self):
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
        assert self.argument() == a.argument()
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
        assert self.argument() == a.argument()
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
        assert self.argument() == a.argument()
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
        assert self.argument() == a.argument()
        self._terms = np.concatenate((self.terms(), a.terms()), axis=1)
        self._scales = np.concatenate((self.scales(), -a.scales()), axis=1)
        self._indices = np.concatenate((self.indices(), a.indices()), axis=1)
        return self

    def __neg__(self):
        return Expr(self.basis(), self.terms().copy(), -self.scales().copy(),
                    self.indices().copy())


class BasisFunction(object):

    def __init__(self, space, argument=0, index=0):
        self._space = space
        self._argument = argument
        self._index = index

    def rank(self):
        return self._space.rank()

    def expr_rank(self):
        return self._space.rank()

    def function_space(self):
        return self._space

    def argument(self):
        return self._argument

    def num_components(self):
        return self.function_space().num_components()

    def index(self):
        """Return index into vector of rank 2"""
        return self._index

    def __getitem__(self, i):
        assert self.rank() == 2
        t0 = BasisFunction(self._space[i], self._argument, i)
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

    def __init__(self, space, index=0):
        return BasisFunction.__init__(self, space, 0, index)

    def __getitem__(self, i):
        assert self.rank() == 2
        t0 = TestFunction(self._space[i], index=i)
        return t0

class TrialFunction(BasisFunction):

    def __init__(self, space, index=0):
        return BasisFunction.__init__(self, space, 1, index)

    def __getitem__(self, i):
        assert self.rank() == 2
        t0 = TrialFunction(self._space[i], index=i)
        return t0


class Function(np.ndarray, BasisFunction):
    """Numpy array for TensorProductSpace

    Parameters
    ----------

    space : Instance of TensorProductSpace (T)
    forward_output : boolean.
        If False then create Function of shape/type for input to T.forward,
        otherwise create Function of shape/type for output from T.forward
    val : int or float
        Value used to initialize array
    buffer : Numpy array or Function with data. Must be of correct shape

    For more information, see numpy.ndarray

    Examples
    --------
    from mpi4py_fft import MPI
    from shenfun.tensorproductspace import TensorProductSpace, Function
    from shenfun.fourier.bases import R2CBasis, C2CBasis

    K0 = C2CBasis(8)
    K1 = R2CBasis(8)
    FFT = TensorProductSpace(MPI.COMM_WORLD, [K0, K1])
    u = Function(FFT, False)
    uhat = Function(FFT, True)

    """

    # pylint: disable=too-few-public-methods,too-many-arguments
    def __new__(cls, space, forward_output=True, val=0, buffer=None):

        if isinstance(buffer, np.ndarray):
            shape = buffer.shape
            dtype = buffer.dtype

        else:

            shape = space.forward.input_array.shape
            dtype = space.forward.input_array.dtype
            if forward_output is True:
                shape = space.forward.output_array.shape
                dtype = space.forward.output_array.dtype

            ndim = space.ndim()
            if not space.num_components() == 1:
                shape = (space.num_components(),) + shape

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype,
                                 buffer=buffer)

        if buffer is None:
            obj.fill(val)
        return obj

    def __init__(self, space, forward_output=True, val=0, buffer=None):
        #super(Function, self).__init__(space, 2)
        BasisFunction.__init__(self, space, 2)

    def __getitem__(self, i):
        # If it's a vector space, then return component, otherwise just return sliced numpy array
        if hasattr(self, '_space'):
            if self.rank() == 2 and i in range(self.num_components()):
                v0 = BasisFunction.__getitem__(self, i)
                v1 = np.ndarray.__getitem__(self, i)
                fun = v0.function_space()
                forward = fun.is_forward_output(v1)
                f0 = Function(fun, forward, buffer=v1)
                f0._index = i
                f0._argument = 2
                return f0
            else:
                v = np.ndarray.__getitem__(self, i)
                return v

        else:
            v = np.ndarray.__getitem__(self, i)
            return v


    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, '_space'):
            self._argument = 2
            self._space = obj._space
            self._index = obj._index

    def as_array(self):
        fun = self.function_space()
        return Array(fun, forward_output=fun.is_forward_output(self), buffer=self)


class Array(np.ndarray):
    """Numpy array for TensorProductSpace

    Parameters
    ----------

    space : Instance of TensorProductSpace
    forward_output : boolean.
        If False then create Array of shape/type for input to
        TensorProductSpace.forward, otherwise create Array of shape/type
        for output from TensorProductSpace.forward
    val : int or float
        Value used to initialize array
    buffer : Numpy array or Array with data. Must be of correct shape

    For more information, see numpy.ndarray

    Examples
    --------
    from mpi4py_fft import MPI
    from shenfun.tensorproductspace import TensorProductSpace, Array
    from shenfun.fourier.bases import R2CBasis, C2CBasis

    K0 = C2CBasis(8)
    K1 = R2CBasis(8)
    FFT = TensorProductSpace(MPI.COMM_WORLD, [K0, K1])
    u = Array(FFT, False)
    uhat = Array(FFT, True)

    """

    # pylint: disable=too-few-public-methods,too-many-arguments
    def __new__(cls, space, forward_output=True, val=0, buffer=None):

        shape = space.forward.input_array.shape
        dtype = space.forward.input_array.dtype
        if forward_output is True:
            shape = space.forward.output_array.shape
            dtype = space.forward.output_array.dtype

        ndim = space.ndim()
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
        if obj is None: return
        if hasattr(obj, '_space'):
            self._space = obj._space

    def function_space(self):
        return self._space

    def rank(self):
        return self._space.rank()

    def as_function(self):
        space = self.function_space()
        forward_output = space.is_forward_output(self)
        return Function(space, forward_output=forward_output, buffer=self)

