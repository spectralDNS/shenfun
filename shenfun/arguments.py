import numpy as np

__all__ = ('Expr', 'BasisFunction', 'TestFunction', 'TrialFunction', 'Function')


class Expr(object):

    def __init__(self, space, integrals, argument):
        self._space = space
        self._integrals = integrals
        self._argument = argument

    def function_space(self):
        return self._space

    def integrals(self):
        return self._integrals

    def argument(self):
        return self._argument

    def rank(self):
        return self._integrals.shape[0]

    def num_integrals(self):
        return self._integrals.shape[1]

    def dim(self):
        return self._integrals.shape[2]


class BasisFunction(Expr):

    def __init__(self, space, argument=0):
        return Expr.__init__(self, space,
                             np.zeros((space.rank(), 1, len(space)), dtype=np.int),
                             argument)


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
    forward_output: boolean.
        If False then create Function of shape/type for input to PFFT.forward,
        otherwise create Function of shape/type for output from PFFT.forward
    val : int or float
        Value used to initialize array

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

        if space.rank() > 1:
            shape = (space.rank(),) + shape

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype,
                                 buffer=buffer)

        if buffer is None:
            obj.fill(val)
        return obj

    def __init__(self, space, forward_output=True, val=0, buffer=None):
        Expr.__init__(self, space, np.zeros((space.rank(), 1, len(space)),
                                            dtype=np.int), 2)

