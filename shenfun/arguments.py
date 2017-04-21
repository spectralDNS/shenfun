import numpy as np

__all__ = ('Expr', 'BasisFunction', 'TestFunction', 'TrialFunction', 'Function')


class Expr(object):
    """Class for spectral Galerkin forms

    """

    def __init__(self, space, argument, integrals=None, scales=None):
        self._space = space
        self._integrals = integrals
        self._argument = argument
        self._scales = scales
        ndim = len(space)
        if integrals is None:
            self._integrals = np.zeros((ndim**(space.rank()-1), 1, ndim),
                                       dtype=np.int)
        if scales is None:
            self._scales = np.ones((ndim**(space.rank()-1), 1))

        assert np.prod(self._scales.shape) == self.num_integrals()*self.num_components()

    def function_space(self):
        return self._space

    def integrals(self):
        return self._integrals

    def scales(self):
        return self._scales

    def argument(self):
        return self._argument

    def num_components(self):
        return self._integrals.shape[0]

    def num_integrals(self):
        return self._integrals.shape[1]

    def dim(self):
        return self._integrals.shape[2]

    def __getitem__(self, i):
        assert self.num_components() == self.dim()
        return Expr(self._space, self._argument,
                    self._integrals[i][np.newaxis, :, :],
                    self._scales[i][np.newaxis, :])

    def __mul__(self, a):
        assert isinstance(a, (int, float, np.floating, np.integer))
        sc = self.scales().copy()*a
        return Expr(self._space, self._argument, self._integrals, sc)

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
                    np.concatenate((self.integrals(), a.integrals()), axis=1),
                    np.concatenate((self.scales(), a.scales()), axis=1))

    def __iadd__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        self._integrals = np.concatenate((self.integrals(), a.integrals()), axis=1)
        self._scales = np.concatenate((self.scales(), a.scales()), axis=1)
        return self

    def __sub__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        return Expr(self.function_space(), self.argument(),
                    np.concatenate((self.integrals(), a.integrals()), axis=1),
                    np.concatenate((self.scales(), -a.scales()), axis=1))

    def __isub__(self, a):
        assert isinstance(a, Expr)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument() == a.argument()
        self._integrals = np.concatenate((self.integrals(), a.integrals()), axis=1)
        self._scales = np.concatenate((self.scales(), -a.scales()), axis=1)
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

        ndim = len(space)
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

