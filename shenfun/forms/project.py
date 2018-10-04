import numpy as np
import sympy as sp
from shenfun.tensorproductspace import TensorProductSpace, MixedTensorProductSpace
from .arguments import Expr, TestFunction, TrialFunction, BasisFunction, \
    Function, Array
from .inner import inner

__all__ = ('project',)

def project(uh, T, output_array=None):
    r"""
    Project uh to tensor poduct space T

    Find :math:`u \in T`, such that

    .. math::

        (u - u_h, v)_w^N = 0 \quad \forall v \in T

    Parameters
    ----------
    uh : Instance of either one of
        - :class:`.Expr`
        - :class:`.BasisFunction`
        - :class:`.Array`
        - A sympy function
    T : :class:`.TensorProductSpace` or :class:`.MixedTensorProductSpace`
    output_array : :class:`.Function`
        Return array

    Returns
    -------
    Function
        The projection of uh in T

    See Also
    --------
    :func:`.inner`

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

    if hasattr(uh, 'evalf'):
        # lambdify sympy function for fast execution
        x, y, z = sp.symbols("x,y,z")
        uh = ({1: lambda a: sp.lambdify((x,), a, 'numpy'),
               2: lambda a: sp.lambdify((x, y), a, 'numpy'),
               3: lambda a: sp.lambdify((x, y, z), a, 'numpy')}[len(T)])(uh)

    if hasattr(uh, '__call__'):
        # Evaluate sympy function on entire mesh
        if isinstance(T, TensorProductSpace):
            uh = Array(T, buffer=uh(*T.local_mesh(True)).astype(T.forward.input_array.dtype))
        else:
            uh = Array(T, buffer=uh(T.mesh()).astype(T.forward.input_array.dtype))

    if isinstance(uh, np.ndarray):
        # Project is just regular forward transform
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
