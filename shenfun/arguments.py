import numpy as np

__all__ = ('Expr', 'BasisFunction', 'TestFunction', 'TrialFunction', 'Function')


class Expr(object):
    """Class for spectral Galerkin forms

    An Expr instance is a form that is linear in TestFunction (v), TrialFunction
    (u) or Function (c). The Function is a Numpy array evaluated at quadrature
    points. Inner products are constructed from forms consisting of two Exprs,
    one with a TestFunction and one with either TrialFunction or Function.

    args:
        space:     TensorProductSpace or VectorTensorProductSpace
        argument:  int  0: Test function
                        1: Trial function
                        2: Function
    kwargs:
        terms:     Numpy array of ndim = 3. Describes operations in Expr

                   Index 0: Vector component. If Expr is rank = 0, then
                            terms[0] = 1. For vectors it equals dim

                   Index 1: One for each term in the form. For example
                            div(grad(u)) has three terms in 3D:
                              d^2u/dx^2 + d^2u/dy^2) + d^2u/dz^2

                   Index 2: The operations stored as an array of len = dim
                            For example, div(grad(u)) in 3D is represented
                            with the three arrays
                              [[2, 0, 0],
                               [0, 2, 0],
                               [0, 0, 2]]
                            meaning the first has two derivatives in first
                            direction and none in the others (d^2u/dx^2),
                            the second has two derivatives in second direction,
                            etc.

                   The Expr div(grad(u)), where u is a scalar, is as such
                   represented in as an array of shape (1, 3, 3), 1 meaning
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

    def __init__(self, space, argument, terms=None, scales=None, indices=None):
        self._space = space
        self._terms = terms
        self._argument = argument
        self._scales = scales
        self._indices = indices
        ndim = space.ndim()
        if terms is None:
            self._terms = np.zeros((ndim**(space.rank()-1), 1, ndim),
                                       dtype=np.int)
        if scales is None:
            self._scales = np.ones((ndim**(space.rank()-1), 1))

        if indices is None:
            self._indices = np.arange(ndim**(space.rank()-1))[:, np.newaxis]

        assert np.prod(self._scales.shape) == self.num_terms()*self.num_components()

    def function_space(self):
        return self._space

    def terms(self):
        return self._terms

    def scales(self):
        return self._scales

    def argument(self):
        return self._argument

    def rank(self):
        return self._space.rank()

    def indices(self):
        return self._indices

    def num_components(self):
        return self._terms.shape[0]

    def num_terms(self):
        return self._terms.shape[1]

    def dim(self):
        return self._terms.shape[2]

    def __getitem__(self, i):
        assert self.num_components() == self.dim()
        return Expr(self._space[i], self._argument,
                    self._terms[i][np.newaxis, :, :],
                    self._scales[i][np.newaxis, :],
                    self._indices[i][np.newaxis, :])

    def __mul__(self, a):
        assert isinstance(a, (int, float, np.floating, np.integer))
        sc = self.scales().copy()*a
        return Expr(self._space, self._argument, self._terms, sc, self._indices)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        assert isinstance(a, (int, float, np.floating, np.integer))
        sc = self.scales()
        sc *= a
        return self

    def __add__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        return Expr(self.function_space(), self.argument(),
                    np.concatenate((self.terms(), a.terms()), axis=1),
                    np.concatenate((self.scales(), a.scales()), axis=1),
                    np.concatenate((self.indices(), a.indices()), axis=1))

    def __iadd__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        self._terms = np.concatenate((self.terms(), a.terms()), axis=1)
        self._scales = np.concatenate((self.scales(), a.scales()), axis=1)
        self._indices = np.concatenate((self.indices(), a.indices()), axis=1)
        return self

    def __sub__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        return Expr(self.function_space(), self.argument(),
                    np.concatenate((self.terms(), a.terms()), axis=1),
                    np.concatenate((self.scales(), -a.scales()), axis=1),
                    np.concatenate((self.indices(), a.indices()), axis=1))

    def __isub__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        self._terms = np.concatenate((self.terms(), a.terms()), axis=1)
        self._scales = np.concatenate((self.scales(), -a.scales()), axis=1)
        self._indices = np.concatenate((self.indices(), a.indices()), axis=1)
        return self


class BasisFunction(Expr):

    def __init__(self, space, argument=0):
        return Expr.__init__(self, space, argument)


class TestFunction(BasisFunction):

    def __init__(self, space):
        return BasisFunction.__init__(self, space, 0)


class TrialFunction(BasisFunction):

    def __init__(self, space):
        return BasisFunction.__init__(self, space, 1)


class Function(np.ndarray, Expr):
    """Numpy array for TensorProductSpace

    Parameters
    ----------

    space : Instance of TensorProductSpace
    forward_output : boolean.
        If False then create Function of shape/type for input to PFFT.forward,
        otherwise create Function of shape/type for output from PFFT.forward
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

        shape = space.forward.input_array.shape
        dtype = space.forward.input_array.dtype
        if forward_output is True:
            shape = space.forward.output_array.shape
            dtype = space.forward.output_array.dtype

        ndim = space.ndim()
        if space.rank() > 1:
            shape = (ndim**(space.rank()-1),) + shape

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype,
                                 buffer=buffer)

        if buffer is None:
            obj.fill(val)
        return obj

    def __init__(self, space, forward_output=True, val=0, buffer=None):
        Expr.__init__(self, space, 2)

    def __getitem__(self, i):
        self.i = i
        v = np.ndarray.__getitem__(self, i)
        return v

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, 'i'):
            if obj.i in (0, 1, 2):
                if obj.rank() == 2:
                    self._space = getattr(obj, '_space')[obj.i]
                    self._argument = getattr(obj, '_argument')
                    self._terms = getattr(obj, '_terms')[obj.i][np.newaxis, :, :]
                    self._scales = getattr(obj, '_scales')[obj.i][np.newaxis, :]
                    self._indices = getattr(obj, '_indices')[obj.i][np.newaxis, :]

                else:
                    self._space = getattr(obj, '_space')
                    self._argument = getattr(obj, '_argument')
                    self._terms = getattr(obj, '_terms')
                    self._scales = getattr(obj, '_scales')
                    self._indices = getattr(obj, '_indices')

            else:
                self._space = getattr(obj, '_space')
                self._argument = getattr(obj, '_argument')
                self._terms = getattr(obj, '_terms')
                self._scales = getattr(obj, '_scales')
                self._indices = getattr(obj, '_indices')

        else:
            self._space = getattr(obj, '_space')
            self._argument = getattr(obj, '_argument')
            self._terms = getattr(obj, '_terms')
            self._scales = getattr(obj, '_scales')
            self._indices = getattr(obj, '_indices')

