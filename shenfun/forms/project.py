from copy import copy
import types
import numpy as np
from shenfun.tensorproductspace import TensorProductSpace
from shenfun.matrixbase import TPMatrix, BlockMatrix, SpectralMatrix
from .arguments import Expr, TestFunction, TrialFunction, BasisFunction, \
    Function, Array
from .inner import inner

__all__ = ('project',)


def project(uh, T, output_array=None, fill=True, use_to_ortho=True, use_assign=True):
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
    use_assign : bool, optional
        Whether to use fast `assign` method for projection of Function to
        a denser space of the same kind.

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
    >>> from shenfun import FunctionSpace, project, TensorProductSpace, Array, \
    ...     Function, Dx
    >>> N = 16
    >>> comm = MPI.COMM_WORLD
    >>> T0 = FunctionSpace(N, 'C')
    >>> K0 = FunctionSpace(N, 'F', dtype='d')
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

    if hasattr(uh, 'free_symbols'):
        # Evaluate sympy function on entire mesh
        uh = Array(T, buffer=uh)

    if isinstance(uh, types.LambdaType):
        raise NotImplementedError('Do not use lambda functions in project')

    if isinstance(uh, Function):
        W = uh.function_space()
        if W == T:
            output_array[:] = uh
            return output_array

        assert W.rank == T.rank
        compatible_bases = W.compatible_base(T)
        if (not compatible_bases) and use_assign:
            # If the underlysing bases are the same, but of different size,
            # then use assign to simply copy to the new space
            try:
                uh.assign(output_array)
                return output_array
            except:
                pass

        elif T.is_orthogonal and use_to_ortho:
            # Try to use fast to_ortho for projection to orthogonal space
            try:
                output_array = uh.to_ortho(output_array)
                return output_array
            except:
                pass

    if isinstance(uh, np.ndarray) and not isinstance(uh, (Array, Function)):
        #assert np.all(uh.shape == T.shape(False))
        uh = Array(T, buffer=uh)

    if isinstance(uh, Array):
        if uh.function_space().compatible_base(T):
            # Project is just regular forward transform
            output_array = T.forward(uh, output_array)
            return output_array
        else:
            raise RuntimeError('Provided Array not the same shape as space projected into')

    assert isinstance(uh, (Expr, BasisFunction))

    v = TestFunction(T)
    u = TrialFunction(T)
    output_array = inner(v, uh, output_array=output_array)
    B = inner(v, u)

    if isinstance(T, TensorProductSpace):
        if len(T.get_nonperiodic_axes()) > 2:
            raise NotImplementedError

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
            transAB = pencilA.transfer(pencilB, output_array.dtype.char)
            output_arrayB = np.zeros(transAB.subshapeB, dtype=output_array.dtype)
            output_arrayB2 = np.zeros(transAB.subshapeB, dtype=output_array.dtype)
            b = B[0].pmat[axis]
            output_array = b.solve(output_array, output_array, axis=axis)
            transAB.forward(output_array, output_arrayB)
            b = B[0].pmat[second_axis]
            output_arrayB2 = b.solve(output_arrayB, output_arrayB2, axis=second_axis)
            transAB.backward(output_arrayB2, output_array)
            return output_array

    if isinstance(B, (TPMatrix, SpectralMatrix)):
        output_array = B.solve(output_array, output_array)

    elif T.coors.is_orthogonal and (len(output_array) == len(B)):
        for oa, b in zip(output_array, B):
            oa = b.solve(oa, oa)

    else:
        M = BlockMatrix(B)
        output_array = M.solve(output_array, output_array)

    return output_array
