import numpy as np
from shenfun.tensorproductspace import MixedTensorProductSpace
from .arguments import Expr, TestFunction, TrialFunction, BasisFunction, Function
from .inner import inner

__all__ = ('project',)

def project(uh, T, output_array=None):
    r"""Project uh to tensor product space T

    Parameters
    ----------
        uh : Expr, Function or Array
        T : TensorProductSpace
        output_array : Function(T)
                       Return array

    .. note:: Returns spectral expansion coefficients of projection,
              not the projection in physical space.

    Example
    -------

    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> from shenfun import chebyshev, fourier, project, TensorProductSpace, \
    ...     Array, Function, Dx
    >>> N = 16
    >>> comm = MPI.COMM_WORLD
    >>> T0 = chebyshev.bases.Basis(N)
    >>> K0 = fourier.bases.R2CBasis(N)
    >>> T = TensorProductSpace(comm, (T0, K0))
    >>> uj = Array(T)
    >>> uj[:] = np.random.random(uj.shape)
    >>> u = Function(T)
    >>> u = project(uj, T, output_array=u) # Same as u = T.forward(uj, u)
    >>> du = project(Dx(u, 0, 1), T)

    """

    if output_array is None:
        output_array = Function(T)

    if isinstance(uh, np.ndarray):
        # Just regular forward transform
        output_array = T.forward(uh, output_array)
        return output_array

    assert isinstance(uh, (Expr, BasisFunction))

    v = TestFunction(T)
    u = TrialFunction(T)
    output_array = inner(v, uh, output_array=output_array)
    B = inner(v, u)
    if isinstance(B, list) and not isinstance(T, MixedTensorProductSpace):
        # Means we have two non-periodic directions
        npaxes = [b for b in B[0].keys() if isinstance(b, int)]
        assert len(npaxes) == 2

        pencilA = T.forward.output_pencil
        axis = pencilA.axis
        npaxes.remove(axis)
        second_axis = npaxes[0]
        pencilB = pencilA.pencil(second_axis)
        transAB = pencilA.transfer(pencilB, 'd')
        output_arrayB = np.zeros(transAB.subshapeB)
        output_arrayB2 = np.zeros(transAB.subshapeB)
        b = B[0][axis]
        output_array = b.solve(output_array, output_array, axis=axis)
        transAB.forward(output_array, output_arrayB)
        b = B[0][second_axis]
        output_arrayB2 = b.solve(output_arrayB, output_arrayB2, axis=second_axis)
        transAB.backward(output_arrayB2, output_array)

    else:
        # Just zero or one non-periodic direction
        if v.rank() == 1:
            axis = B.axis if hasattr(B, 'axis') else 0 # if periodic the solve is just an elementwise division not using axis
            output_array = B.solve(output_array, output_array, axis=axis)
        else:
            for i in range(v.function_space().ndim()):
                axis = B[i].axis if hasattr(B[i], 'axis') else 0
                output_array[i] = B[i].solve(output_array[i], output_array[i], axis=axis)
    return output_array

