from copy import copy
import numpy as np
import sympy as sp
from shenfun.tensorproductspace import TensorProductSpace
from shenfun.matrixbase import TPMatrix
from .arguments import Expr, TestFunction, TrialFunction, BasisFunction, \
    Function, Array
from .inner import inner

__all__ = ('project',)

def project(uh, T, output_array=None, fill=True, use_to_ortho=True):
    r"""
    Project ``uh`` to tensor product space ``T``

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
    output_array : :class:`.Function`, optional
        Return array
    fill : bool, optional
        Whether to fill the `output_array` with zeros before projection
    use_to_ortho : bool, optional
        Whether to use fast `to_ortho` method for projection of Functions
        to orthogonal space.

    Returns
    -------
    Function
        The projection of ``uh`` in T

    See Also
    --------
    :func:`.inner`

    Example
    -------

    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> from shenfun import Basis, project, TensorProductSpace, Array, \
    ...     Function, Dx
    >>> N = 16
    >>> comm = MPI.COMM_WORLD
    >>> T0 = Basis(N, 'C')
    >>> K0 = Basis(N, 'F', dtype='d')
    >>> T = TensorProductSpace(comm, (T0, K0))
    >>> uj = Array(T)
    >>> uj[:] = np.random.random(uj.shape)
    >>> u = Function(T)
    >>> u = project(uj, T, output_array=u) # Same as u = T.forward(uj, u)
    >>> du = project(Dx(u, 0, 1), T)

    """

    if output_array is None:
        output_array = Function(T)
    elif fill:
        output_array.fill(0)

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

    if isinstance(uh, np.ndarray) and not isinstance(uh, (Array, Function)):
        uh = Array(T, buffer=uh)

    if isinstance(uh, Array) and uh.function_space() == T:
        # Project is just regular forward transform
        output_array = T.forward(uh, output_array)
        return output_array

    if isinstance(uh, Function) and T.is_orthogonal and use_to_ortho:
        # Try to use fast to_ortho for projection to orthogonal space
        try:
            output_array = uh.to_ortho(output_array)
            return output_array
        except:
            pass

    assert isinstance(uh, (Expr, Array, BasisFunction))

    v = TestFunction(T)
    u = TrialFunction(T)
    output_array = inner(v, uh, output_array=output_array)
    B = inner(v, u)

    if isinstance(T, TensorProductSpace):
        if len(T.get_nonperiodic_axes()) == 2:
            # Means we have two non-periodic directions
            B = [B] if isinstance(B, TPMatrix) else B
            npaxes = copy(B[0].naxes)
            assert len(npaxes) == 2

            pencilA = T.forward.output_pencil
            axis = pencilA.axis
            npaxes.remove(axis)
            second_axis = npaxes[0]
            pencilB = pencilA.pencil(second_axis)
            transAB = pencilA.transfer(pencilB, 'd')
            output_arrayB = np.zeros(transAB.subshapeB)
            output_arrayB2 = np.zeros(transAB.subshapeB)
            b = B[0].pmat[axis]
            output_array = b.solve(output_array, output_array, axis=axis)
            transAB.forward(output_array, output_arrayB)
            b = B[0].pmat[second_axis]
            output_arrayB2 = b.solve(output_arrayB, output_arrayB2, axis=second_axis)
            transAB.backward(output_arrayB2, output_array)
            return output_array

    # Just zero or one non-periodic direction
    if v.rank == 0:
        output_array = B.solve(output_array, output_array)
    else:
        for oa, b in zip(output_array, B):
            oa = b.solve(oa, oa)
    return output_array
