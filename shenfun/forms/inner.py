"""
This module contains the inner function that computes the
weighted inner product.
"""
from functools import reduce
from copy import copy
import numpy as np
from shenfun.spectralbase import inner_product
from shenfun.matrixbase import TPMatrix, SparseMatrix
from shenfun.la import DiagonalMatrix
from shenfun.tensorproductspace import MixedTensorProductSpace
from shenfun.matrixbase import BlockMatrix
from .arguments import Expr, Function, BasisFunction, Array

__all__ = ('inner',)

#pylint: disable=line-too-long,inconsistent-return-statements,too-many-return-statements

def inner(expr0, expr1, output_array=None, level=0):
    r"""
    Return weighted discrete inner product of linear or bilinear form

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
    expr0, expr1 : :class:`.Expr`, :class:`.BasisFunction` or :class:`.Array`
        Either one can be an expression involving a
        BasisFunction (:class:`.TestFunction`, :class:`.TrialFunction` or
        :class:`.Function`) or an Array. With expressions (Expr) on a
        BasisFunction we typically mean terms like div(u) or grad(u), where
        u is any one of the different types of BasisFunction.
        One of ``expr0`` or ``expr1`` need to be an expression on a
        TestFunction. If the second then involves a TrialFunction, a matrix is
        returned. If one of ``expr0``/``expr1`` involves a TestFunction and the
        other one is an expression on a Function, or a plain Array, then a
        linear form is assembled and a Function is returned.

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
    Function
        For linear forms involving one :class:`.TestFunction` and one
        :class:`.BasisFunction` or :class:`.Array`.

    SparseMatrix
        For bilinear forms involving both :class:`.TestFunction` and
        :class:`.TrialFunction`.

    dict
        For bilinear forms with many terms. Each item has a
        :class:`.SparseMatrix` as value.

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

    if test.rank() > 0 and test.expr_rank() > 0: # For vector expressions of rank > 0 use recursive algorithm
        ndim = test.function_space().num_components()

        if output_array is None and trial.argument == 2:
            output_array = Function(test.function_space())

        if trial.argument == 2:
            # linear form
            for ii in range(ndim):
                output_array[ii] = inner(test[ii], trial[ii],
                                         output_array=output_array[ii])
            return output_array

        result = []
        for ii in range(ndim):
            l = inner(test[ii], trial[ii], level=level)
            result += [l] if isinstance(l, TPMatrix) else l

        # Add equal TPMatrices
        if level == 0:
            B = [result[0]]
            for a in result[1:]:
                found = False
                for b in B:
                    if a == b:
                        b += a
                        found = True
                if not found:
                    B.append(a)
        return B[0] if len(B) == 1 else B

    if trial.argument > 1:
        # Linear form
        assert isinstance(test, (Expr, BasisFunction))
        assert test.argument == 0
        space = test.function_space()
        if isinstance(trial, Array):
            output_array = space.scalar_product(trial, output_array)
            return output_array

    # If trial is an Expr with terms, then compute using bilinear form and matvec

    assert isinstance(trial, (Expr, BasisFunction))
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(trial, BasisFunction):
        trial = Expr(trial)
    if isinstance(test, BasisFunction):
        test = Expr(test)

    space = test.function_space()
    base = test.basis().base
    trialspace = trial.function_space()
    test_scale = test.scales()
    trial_scale = trial.scales()
    trial_indices = trial.indices()
    test_indices = test.indices()

    uh = None
    if trial.argument == 2:
        uh = trial.base

    if output_array is None and trial.argument == 2:
        output_array = Function(test.function_space())

    A = []
    vec = 0
    for base_test, base_trial in zip(test.terms(), trial.terms()): # vector/scalar
        for test_j, b0 in enumerate(base_test):              # second index test
            for trial_j, b1 in enumerate(base_trial):        # second index trial
                sc = test_scale[vec, test_j]*trial_scale[vec, trial_j]
                M = []
                assert len(b0) == len(b1)
                trial_sp = trialspace
                if isinstance(trialspace, MixedTensorProductSpace): # could operate on a vector, e.g., div(u), where u is vector
                    trial_sp = trialspace[vec]
                test_sp = space
                if isinstance(space, MixedTensorProductSpace):
                    test_sp = space[vec]
                for i, (a, b) in enumerate(zip(b0, b1)): # Third index, one inner for each dimension
                    ts = trial_sp[i]
                    sp = test_sp[i]
                    AA = inner_product((sp, a), (ts, b))
                    M.append(AA)
                    # Take care of domains of not standard size
                    if not sp.domain_factor() == 1:
                        sc *= sp.domain_factor()**(a+b)
                sc = sp.broadcast_to_ndims(np.array([sc]))
                A.append(TPMatrix(M, test_sp, sc, (test_indices[0, test_j], trial_indices[0, trial_j]), base))
        vec += 1

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
    #
    # The inner product will return a dictionary of a type constructed from the
    # matrices in A. In this case:
    #
    # B = (v[0], u[0]'')_x
    # B.scale = (v[1], u[1])_y[local_shape]
    # B.axis = 0
    # C = (v[0], u[0])_x
    # C.scale = (v[1], u[1])_y[local_shape]
    # C.axis = 0
    # return {'ADDmat': B,
    #         'BDDmat': C}
    #
    # where the name 'ADDmat' is obtained from B.get_key()
    #
    # where local_shape is used to indicate that we are only returning the local
    # values of the scale arrays.

    if level == 2 and trial.argument == 1:
        return A

    for tpmat in A:
        tpmat.simplify_fourier_matrices()

    if level == 1 and trial.argument == 1:
        return A

    if np.all([f.all_identity() for f in A]): # No non-diagonal matrix
        if trial.argument == 1:
            if trial.rank() == 0:
                return reduce(lambda x, y: x+y, [s for s in A])
            else:
                return A

        # linear form
        if uh.rank() > 0:
            for i, b in enumerate(A):
                output_array += b.scale*uh[trial_indices[0, i]]
        else:
            f = reduce(lambda x, y: x+y, [s for s in A])
            output_array[:] = f.scale*uh
        return output_array

    elif np.any([isinstance(f.pmat, SparseMatrix) for f in A]):
        # One non-Fourier space
        B = [A[0]]
        for a in A[1:]:  # Add equal TPMatrices
            found = False
            for b in B:
                if a == b:
                    b += a
                    found = True
            if not found:
                B.append(a)
        if trial.argument == 1:  # bilinear form
            if len(B) == 1:
                if len(B[0].mats) == 1:
                    p = B[0].pmat
                    p.scale = p.scale*B[0].scale
                    return p
                return B[0]
            return B

        else: # linear form
            wh = np.empty_like(output_array)
            for i, b in enumerate(B):
                if uh.rank() > 0:
                    wh = b.matvec(uh[trial_indices[0, i]], wh)
                else:
                    wh = b.matvec(uh, wh)
                output_array += wh
            return output_array

    else:
        # Two non-Fourier spaces (experimental)
        if trial.argument == 1:  # bilinear form
            return A

        else: # linear form
            wh = np.empty_like(output_array)
            for i, b in enumerate(A):
                if uh.rank() > 0:
                    wh = b.matvec(uh[trial_indices[0, i]], wh)
                else:
                    wh = b.matvec(uh, wh)
                output_array += wh
            return output_array
