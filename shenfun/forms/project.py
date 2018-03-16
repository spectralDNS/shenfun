#pylint: disable=line-too-long, missing-docstring

import numpy as np
from shenfun.tensorproductspace import MixedTensorProductSpace
from .arguments import Expr, TestFunction, TrialFunction, BasisFunction, Array
from .inner import inner

__all__ = ('project',)

def project(uh, T, output_array=None, uh_hat=None):
    """Project uh to tensor product space T

    args:
        uh        Expr or Function
        T         TensorProductSpace instance

    kwargs:
        output_array  Function(T, True)  Return array
        uh_hat        Function(T, True)  The transform of uh in uh's space,
                                         i.e.,
                                           TK = uh.function_space()
                                           uh_hat = TK.forward(uh)
                                         This is ok even though uh is part of
                                         a form, like div(grad(uh))

    Note: Returns spectral expansion coefficients of projection, not the
          projection in physical space.

    Example:

    import numpy as np
    from mpi4py import MPI
    from shenfun import chebyshev, fourier, project, TensorProductSpace, \
        Function, Dx

    N = 16
    comm = MPI.COMM_WORLD
    T0 = chebyshev.bases.Basis(N)
    K0 = fourier.bases.R2CBasis(N)
    T = TensorProductSpace(comm, (T0, K0))
    uj = Function(T, False)
    uj[:] = np.random.random(uj.shape)
    u_hat = project(uj, T)    # Same as u_hat=Function(T);u_hat = T.forward(uj, u_hat)
    du_hat = project(Dx(uj, 0, 1), T, uh_hat=u_hat)

    """

    if output_array is None:
        output_array = Array(T)

    if isinstance(uh, np.ndarray):
        # Just regular forward transform
        output_array = T.forward(uh, output_array)
        return output_array

    assert isinstance(uh, (Expr, BasisFunction))

    v = TestFunction(T)
    u = TrialFunction(T)
    output_array = inner(v, uh, output_array=output_array, uh_hat=uh_hat)
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
