"""
This module contains the inner function that computes the
weighted inner product.
"""
from numbers import Number
import numpy as np
from shenfun.spectralbase import inner_product, SpectralBase, MixedBasis
from shenfun.matrixbase import TPMatrix
from shenfun.tensorproductspace import TensorProductSpace, MixedTensorProductSpace
from shenfun.utilities import dx, split
from .arguments import Expr, Function, BasisFunction, Array

__all__ = ('inner',)

#pylint: disable=line-too-long,inconsistent-return-statements,too-many-return-statements


def inner(expr0, expr1, output_array=None, level=0):
    r"""
    Return (weighted) discrete inner product of linear or bilinear form

    .. math::

        (f, g)_w^N = \sum_{i\in\mathcal{I}}f(x_i) \overline{g}(x_i) w_i \approx \int_{\Omega} g\, \overline{f}\, w\, dx

    where :math:`\mathcal{I}=0, 1, \ldots, N, N \in \mathbb{Z}^+`, :math:`f`
    is an expression linear in a :class:`.TestFunction`, and :math:`g` is an
    expression that is linear in :class:`.TrialFunction` or :class:`.Function`,
    or it is simply an :class:`.Array` (a solution interpolated on the
    quadrature mesh in physical space). :math:`w` is a weight associated with
    chosen basis, and :math:`w_i` are quadrature weights.

    If the expressions are created in a multidimensional :class:`.TensorProductSpace`,
    then the sum above is over all dimensions. In 2D it becomes:

    .. math::

        (f, g)_w^N = \sum_{i\in\mathcal{I}}\sum_{j\in\mathcal{J}} f(x_i, y_j) \overline{g}(x_i, y_j) w_j w_i

    where :math:`\mathcal{J}=0, 1, \ldots, M, M \in \mathbb{Z}^+`.

    Parameters
    ----------
    expr0, expr1 : :class:`.Expr`, :class:`.BasisFunction`, :class:`.Array`
        or number.
        Either one can be an expression involving a
        BasisFunction (:class:`.TestFunction`, :class:`.TrialFunction` or
        :class:`.Function`) an Array or a number. With expressions (Expr) on a
        BasisFunction we typically mean terms like div(u) or grad(u), where
        u is any one of the different types of BasisFunction.
        If one of ``expr0``/``expr1`` involves a TestFunction and the other one
        involves a TrialFunction, then a tensor product matrix (or a list of
        tensor product matrices) is returned.
        If one of ``expr0``/``expr1`` involves a TestFunction and the other one
        involves a Function, or a plain Array, then a linear form is assembled
        and a Function is returned.
        If one of ``expr0``/``expr1`` is a number (typically 1), or a tuple of
        numbers for a vector, then the inner product represents a non-weighted
        integral over the domain. If a single number, then the other expression
        must be a scalar Array or a Function. If a tuple of numbers, then the
        Array/Function must be a vector.

    output_array:  Function
        Optional return array for linear form.

    level: int
        The level of postprocessing for assembled matrices. Applies only
        to bilinear forms

        - 0 Full postprocessing - diagonal matrices to scale arrays
          and add equal matrices
        - 1 Diagonal matrices to scale arrays, but don't add equal
          matrices
        - 2 No postprocessing, return all assembled matrices

    Returns
    -------
    Depending on dimensionality and the arguments to the forms

        :class:`.Function`
        for linear forms.

        :class:`.SparseMatrix`
        for bilinear 1D forms.

        :class:`.TPMatrix` or list of :class:`.TPMatrix`
        for bilinear multidimensional forms.

        Number, for non-weighted integral where either one of the arguments
        is a number.

    See Also
    --------
    :func:`.project`

    Example
    -------
    Compute mass matrix of Shen's Chebyshev Dirichlet basis:

    >>> from shenfun import Basis
    >>> from shenfun import TestFunction, TrialFunction
    >>> SD = Basis(6, 'Chebyshev', bc=(0, 0))
    >>> u = TrialFunction(SD)
    >>> v = TestFunction(SD)
    >>> B = inner(v, u)
    >>> d = {-2: np.array([-np.pi/2]),
    ...       0: np.array([ 1.5*np.pi, np.pi, np.pi, np.pi]),
    ...       2: np.array([-np.pi/2])}
    >>> [np.all(abs(B[k]-v) < 1e-7) for k, v in d.items()]
    [True, True, True]

    """
    # Wrap a pure numpy array in Array
    if isinstance(expr0, np.ndarray) and not isinstance(expr0, (Array, Function)):
        assert isinstance(expr1, (Expr, BasisFunction))
        if not expr0.flags['C_CONTIGUOUS']:
            expr0 = expr0.copy()
        expr0 = Array(expr1.function_space(), buffer=expr0)
    if isinstance(expr1, np.ndarray) and not isinstance(expr1, (Array, Function)):
        assert isinstance(expr0, (Expr, BasisFunction))
        if not expr1.flags['C_CONTIGUOUS']:
            expr1 = expr1.copy()
        expr1 = Array(expr0.function_space(), buffer=expr1)

    if isinstance(expr0, Number):
        assert isinstance(expr1, (Array, Function))
        space = expr1.function_space()
        if isinstance(space, (TensorProductSpace, MixedTensorProductSpace)):
            df = np.prod(np.array([base.domain_factor() for base in space.bases]))
        elif isinstance(space, SpectralBase):
            df = space.domain_factor()
        if isinstance(expr1, Function):
            return (expr0/df)*dx(expr1.backward())
        if hasattr(space, 'hi'):
            if space.hi.prod() != 1:
                expr1 = space.get_measured_array(expr1.copy())

        return (expr0/df)*dx(expr1)

    if isinstance(expr1, Number):
        assert isinstance(expr0, (Array, Function))
        space = expr0.function_space()
        if isinstance(space, (TensorProductSpace, MixedTensorProductSpace)):
            df = np.prod(np.array([base.domain_factor() for base in space.bases]))
        elif isinstance(space, SpectralBase):
            df = space.domain_factor()
        if isinstance(expr0, Function):
            return (expr1/df)*dx(expr0.backward())
        if hasattr(space, 'hi'):
            if space.hi.prod() != 1:
                expr0 = space.get_measured_array(expr0.copy())

        return (expr1/df)*dx(expr0)

    if isinstance(expr0, tuple):
        assert isinstance(expr1, (Array, Function))
        space = expr1.function_space()
        assert isinstance(space, MixedTensorProductSpace)
        assert len(expr0) == len(space)
        result = 0.0
        for e0i, e1i in zip(expr0, expr1):
            result += inner(e0i, e1i)
        return result

    if isinstance(expr1, tuple):
        assert isinstance(expr0, (Array, Function))
        space = expr0.function_space()
        assert isinstance(space, MixedTensorProductSpace)
        assert len(expr1) == len(space)
        result = 0.0
        for e0i, e1i in zip(expr0, expr1):
            result += inner(e0i, e1i)
        return result

    assert np.all([hasattr(e, 'argument') for e in (expr0, expr1)])
    t0 = expr0.argument
    t1 = expr1.argument
    if t0 == 0:
        assert t1 in (1, 2)
        test = expr0
        trial = expr1
    elif t0 in (1, 2):
        assert t1 == 0
        test = expr1
        trial = expr0
    else:
        raise RuntimeError

    if isinstance(test, BasisFunction):
        recursive = test.rank > 0
    elif isinstance(test, Expr):
        recursive = test.function_space().is_composite_space
    if isinstance(trial, Array):
        assert trial.rank == test.rank
    elif isinstance(trial, BasisFunction):
        recursive *= (trial.expr_rank() > 0)

    if recursive: # Use recursive algorithm for vector expressions of expr_rank > 0, e.g., inner(v, grad(u))
        if output_array is None and trial.argument == 2:
            output_array = Function(test.function_space())

        gij = test.function_space().coors.get_covariant_metric_tensor()
        if trial.argument == 2:
            # linear form
            w0 = np.zeros_like(output_array[0])
            for i, (te, x) in enumerate(zip(test, output_array)):
                for j, tr in enumerate(trial):
                    if gij[i, j] == 0:
                        continue
                    w0.fill(0)
                    x += inner(te*gij[i, j], tr, output_array=w0)
            return output_array

        result = []

        for i, te in enumerate(test):
            for j, tr in enumerate(trial):
                if gij[i, j] == 0:
                    continue
                l = inner(te*gij[i, j], tr, level=level)
                result += l if isinstance(l, list) else [l]
        return result[0] if len(result) == 1 else result

    if output_array is None and trial.argument == 2:
        output_array = Function(test.function_space())

    if trial.argument > 1:
        # Linear form
        assert isinstance(test, (Expr, BasisFunction))
        assert test.argument == 0
        space = test.function_space()
        if isinstance(trial, Array):
            if trial.rank == 0:
                output_array = space.scalar_product(trial, output_array)
                return output_array
            trial = trial.forward()

    assert isinstance(trial, (Expr, BasisFunction))
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(trial, BasisFunction):
        trial = Expr(trial)
    if isinstance(test, BasisFunction):
        test = Expr(test)

    assert test.expr_rank() == trial.expr_rank()
    testspace = test.base.function_space()
    trialspace = trial.base.function_space()
    test_scale = test.scales()
    trial_scale = trial.scales()

    uh = None
    if trial.argument == 2:
        uh = trial.base

    A = []
    gij = testspace.coors.get_covariant_metric_tensor()
    for vec_i, (base_test, test_ind) in enumerate(zip(test.terms(), test.indices())): # vector/scalar
        for vec_j, (base_trial, trial_ind) in enumerate(zip(trial.terms(), trial.indices())):
            g = 1 if len(test.terms()) == 1 else gij[vec_i, vec_j]
            if g == 0:
                continue
            for test_j, b0 in enumerate(base_test):              # second index test
                for trial_j, b1 in enumerate(base_trial):        # second index trial
                    dV = test_scale[vec_i][test_j]*trial_scale[vec_j][trial_j]*testspace.hi.prod()*g
                    assert len(b0) == len(b1)
                    trial_sp = trialspace
                    if isinstance(trialspace, (MixedTensorProductSpace, MixedBasis)): # could operate on a vector, e.g., div(u), where u is vector
                        trial_sp = trialspace.flatten()[trial_ind[trial_j]]
                    test_sp = testspace
                    if isinstance(testspace, (MixedTensorProductSpace, MixedBasis)):
                        test_sp = testspace.flatten()[test_ind[test_j]]
                    has_bcs = False
                    # Check if scale is zero
                    if dV == 0:
                        continue

                    for dv in split(dV):
                        sc = dv['coeff']
                        scb = dv['coeff']
                        M = []
                        DM = []
                        for i, (a, b) in enumerate(zip(b0, b1)): # Third index, one inner for each dimension
                            ts = trial_sp[i]
                            tt = test_sp[i]
                            msx = 'xyzrs'[i]
                            msi = dv[msx]

                            # assemble inner product
                            AA = inner_product((tt, a), (ts, b), msi)
                            M.append(AA)
                            if not abs(AA.scale-1.) < 1e-8:
                                sc *= AA.scale
                                AA.scale = 1.0

                            if ts.has_nonhomogeneous_bcs:
                                tsc = ts.get_bc_basis()
                                BB = inner_product((tt, a), (tsc, b), msi)
                                if not abs(BB.scale-1.) < 1e-8:
                                    scb *= BB.scale
                                    BB.scale = 1.0
                                if BB:
                                    DM.append(BB)
                                    has_bcs = True
                            else:
                                DM.append(AA)

                        sc = tt.broadcast_to_ndims(np.array([sc]))
                        if len(M) == 1: # 1D case
                            M[0].global_index = (test_ind[test_j], trial_ind[trial_j])
                            M[0].scale = sc[0]
                            M[0].mixedbase = testspace
                            A.append(M[0])
                        else:
                            A.append(TPMatrix(M, test_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace))
                        if has_bcs:
                            if len(DM) == 1: # 1D case
                                DM[0].global_index = (test_ind[test_j], trial_ind[trial_j])
                                DM[0].scale = scb
                                DM[0].mixedbase = testspace
                                A.append(DM[0])
                            else:
                                A.append(TPMatrix(DM, test_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace))

    # At this point A contains all matrices of the form. The length of A is
    # the number of inner products. For each index into A there are ndim 1D
    # inner products along, e.g., x, y and z-directions, or just x, y for 2D.
    # The outer product of these matrices is a tensorproduct matrix, and we
    # store the matrices using the TPMatrix class.
    #
    # Diagonal matrices can be eliminated and put in a scale array for the
    # non-diagonal matrices. E.g. for (v, div(grad(u))) in 2D
    #
    # Here A = [TPMatrix([(v[0], u[0]'')_x, (v[1], u[1])_y]),
    #           TPMatrix([(v[0], u[0])_x, (v[1], u[1]'')_y])]
    #
    # where v[0], v[1] are the test functions in x- and y-directions,
    # respectively. For example, v[0] could be a ShenDirichletBasis and v[1]
    # could be a FourierBasis. Same for u.
    #
    # There are now two possibilities, either a linear or a bilinear form.
    # A linear form has trial.argument == 2, whereas a bilinear form has
    # trial.argument == 1. A linear form should assemble to an array and
    # return this array. A bilinear form, on the other hand, should return
    # matrices. Which matrices, and how many, will of course depend on the
    # form and the number of terms.
    #
    # Considering again the tensor product space with ShenDirichlet and Fourier,
    # the list A will contain matrices as shown above. If Fourier is associated
    # with index 1, then (v[1], u[1])_y and (v[1], u[1]'')_y will be diagonal
    # whereas (v[0], u[0]'')_x and (v[0], u[0])_x will in general not. These
    # two matrices are usually termed the stiffness and mass matrices, and they
    # have been implemented in chebyshev/matrices.py or legendre/matrices.py,
    # where they are called ADDmat and BDDmat, respectively.

    if level == 2 and trial.argument == 1: # No processing of matrices
        return A

    for tpmat in A:
        if isinstance(tpmat, TPMatrix):
            tpmat.simplify_fourier_matrices()

    # Add equal matrices
    B = [A[0]]
    for a in A[1:]:
        found = False
        for b in B:
            if a == b:
                b += a
                found = True
        if not found:
            B.append(a)

    A = B

    # Bilinear form, return matrices
    if trial.argument == 1:
        return A[0] if len(A) == 1 else A

    # Linear form, return output_array
    wh = np.zeros_like(output_array)
    for b in A:
        if uh.rank > 0:
            wh = b.matvec(uh.v[b.global_index[1]], wh)
        else:
            wh = b.matvec(uh, wh)
        output_array += wh
        wh.fill(0)
    return output_array
