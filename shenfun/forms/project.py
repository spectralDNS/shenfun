import types
import numpy as np
from shenfun import la
from shenfun.utilities import CachedArrayDict
from shenfun.tensorproductspace import TensorProductSpace
from shenfun.matrixbase import TPMatrix, BlockMatrix, SpectralMatrix, \
    Identity
from .arguments import Expr, TestFunction, TrialFunction, BasisFunction, \
    Function, Array
from .inner import inner

__all__ = ('project', 'Project')

work = CachedArrayDict()

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
    T : :class:`.SpectralBase`, :class:`.TensorProductSpace` or :class:`.CompositeSpace`
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
            uh = uh.forward()
            #raise RuntimeError('Provided Array not the same shape as space projected into')

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
            npaxes = list(B[0].naxes)
            assert len(npaxes) == 2

            pencilA = T.forward.output_pencil
            axis = pencilA.axis
            npaxes.remove(axis)
            second_axis = npaxes[0]
            pencilB = pencilA.pencil(second_axis)
            transAB = pencilA.transfer(pencilB, output_array.dtype.char)
            output_arrayB = np.zeros(transAB.subshapeB, dtype=output_array.dtype)
            output_arrayB2 = np.zeros(transAB.subshapeB, dtype=output_array.dtype)
            b = B[0].mats[axis]
            output_array = b.solve(output_array, output_array, axis=axis)
            transAB.forward(output_array, output_arrayB)
            b = B[0].mats[second_axis]
            output_arrayB2 = b.solve(output_arrayB, output_arrayB2, axis=second_axis)
            transAB.backward(output_arrayB2, output_array)
            return output_array

    if isinstance(B, (TPMatrix, SpectralMatrix)):
        output_array = B.solve(output_array)

    elif T.coors.is_orthogonal and (len(output_array) == len(B)):
        for oa, b in zip(output_array.v, B):
            oa = b.solve(oa, oa)

    else:
        M = BlockMatrix(B)
        output_array = M.solve(output_array, output_array)

    return output_array


class Project:
    """Return an instance of a class that can perform projections efficiently

    Parameters
    ----------
    uh : Instance of either one of
        - :class:`.Expr`
        - :class:`.BasisFunction`
    T : :class:`.SpectralBase`, :class:`.TensorProductSpace` or :class:`.CompositeSpace`

    Example
    -------
    >>> from shenfun import TestFunction, Function, Dx, Project, FunctionSpace
    >>> T = FunctionSpace(8, 'C')
    >>> u = Function(T)
    >>> u[1] = 1
    >>> dudx = Project(Dx(u, 0, 1), T)
    >>> dudx()
    Function([1., 0., 0., 0., 0., 0., 0., 0.])

    Note that u[1] = 1 sets the coefficient of the second Chebyshev polynomial (i.e., x)
    to 1. Hence the derivative of u is dx/dx=1, which is the result of dudx().
    """
    def __init__(self, uh, T, output_array=None):
        assert isinstance(uh, (Expr, BasisFunction))
        v = TestFunction(T)
        u = TrialFunction(T)
        self.B = inner(v, u)
        # replace uh with trial function and assemble matrices used to compute
        # right hand side through matrix vector product
        self.uh = uh
        self.A = inner(v, uh, return_matrices=True)
        self.output_array = output_array
        if output_array is None:
            self.output_array = Function(T)

        if T.dimensions == 1:
            self.sol = la.Solver(self.B)
        else:
            if isinstance(self.B, TPMatrix):
                if len(self.B.naxes) == 0:
                    self.sol = la.SolverDiagonal([self.B])
                elif len(self.B.naxes) == 1:
                    if len(self.A) == 1:
                        axis = self.B.naxes.item()
                        if self.B.mats[axis] == self.A[0].mats[axis]:
                            # If the non-diagonal matrices are the same, we can skip the solve step. This is simply an optimization.
                            self.B.mats[axis] = Identity(shape=self.B.mats[axis].shape)
                            self.A[0].mats[axis] = Identity(shape=self.B.mats[axis].shape)
                            self.sol = la.SolverDiagonal([self.B])
                            self.sol.mat.naxes = np.array([])
                            self.A[0].naxes = np.array([])
                            self.A[0].simplify_diagonal_matrices()
                        else:
                            self.sol = la.SolverGeneric1ND([self.B])
                    else:
                        self.sol = la.SolverGeneric1ND([self.B])
                elif len(self.B.naxes) == 2:
                    self.sol = la.SolverGeneric2ND([self.B])
                else:
                    self.sol = la.SolverND([self.B])
            elif T.coors.is_orthogonal and (len(self.output_array) == len(self.B)):
                # vector with uncoupled components
                class sol:
                    def __init__(self, B):
                        self.solvers = []
                        for b in B:
                            if len(b.naxes) == 0:
                                self.solvers.append(la.SolverDiagonal([b]))
                            elif len(b.naxes) == 1:
                                self.solvers.append(la.SolverGeneric1ND([b]))
                            elif len(b.naxes) == 2:
                                self.solvers.append(la.SolverGeneric2ND([b]))
                            else:
                                self.solvers.append(la.SolverND([b]))
                    def __call__(self, u, c):
                        for oa, solver in zip(u.v, self.solvers):
                            oa = solver(oa, oa)
                        return u
                self.sol = sol(self.B)
            else:
                self.sol = la.BlockMatrixSolver(BlockMatrix(self.B))

    def __call__(self):
        wh = work[(self.output_array, 0, True)]
        uh = self.uh.base
        self.output_array.fill(0)
        for b in self.A:
            if uh.function_space().is_composite_space and wh.ndim == b.dimensions:
                wh = b.matvec(uh.v[b.global_index[1]], wh)
                self.output_array += wh
            elif uh.function_space().is_composite_space and wh.ndim > b.dimensions:
                wh[b.global_index[0]] = b.matvec(uh.v[b.global_index[1]], wh[b.global_index[0]])
                self.output_array.v[b.global_index[0]] += wh[b.global_index[0]]
            else:
                wh = b.matvec(uh, wh)
                self.output_array += wh
            wh.fill(0)
        out = self.sol(self.output_array, self.output_array)
        return out
