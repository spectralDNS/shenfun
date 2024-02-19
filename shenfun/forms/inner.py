from numbers import Number
import sympy as sp
import numpy as np
from shenfun.spectralbase import inner_product, SpectralBase, MixedFunctionSpace
from shenfun.matrixbase import TPMatrix
from shenfun.tensorproductspace import TensorProductSpace, CompositeSpace
from shenfun.utilities import dx, split, scalar_product, CachedArrayDict
from shenfun.config import config
from .arguments import Expr, Function, BasisFunction, Array, TestFunction

__all__ = ('inner', 'Inner')

#pylint: disable=line-too-long,inconsistent-return-statements,too-many-return-statements


def inner(expr0, expr1, output_array=None, assemble=None, kind=None, fixed_resolution=None, return_matrices=False):
    r"""
    Return (weighted or unweighted) discrete inner product

    .. math::

        (f, g)_w^N = \sum_{i\in\mathcal{I}}f(x_i) \overline{g}(x_i) w_i \approx \int_{\Omega} f\, \overline{g}\, w\, dx

    where :math:`\mathcal{I}=0, 1, \ldots, N-1, N \in \mathbb{Z}^+`, :math:`f`
    is a number or an expression linear in a :class:`.TestFunction`,
    and :math:`g` is an expression that is linear in :class:`.TrialFunction`
    or :class:`.Function`, or it is simply an :class:`.Array` (a solution interpolated on the
    quadrature mesh in physical space). :math:`w` is a weight associated with
    chosen basis, and :math:`w_i` are quadrature weights.

    Note
    ----
        If :math:`f` is a number (typically one) and :math:`g` an :class:`.Array`, then `inner`
        represents an unweighted, regular integral over the domain.

    If the expressions are created in a multidimensional :class:`.TensorProductSpace`,
    then the sum above is over all dimensions. In 2D it becomes:

    .. math::

        (f, g)_w^N = \sum_{i\in\mathcal{I}}\sum_{j\in\mathcal{J}} f(x_i, y_j) \overline{g}(x_i, y_j) w_j w_i

    where :math:`\mathcal{J}=0, 1, \ldots, M, M \in \mathbb{Z}^+`.

    Parameters
    ----------
    expr0, expr1 : :class:`.Expr`, :class:`.BasisFunction`, :class:`.Array`, number
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

    output_array : Function
        Optional return array for linear form.

    assemble : None or str, optional
        Determines how to perform the integration

        - 'quadrature' (default)
        - 'exact'
        - 'adaptive'

        Exact and adaptive should result in the same matrix. Exact computes the
        integral using `Sympy integrate <https://docs.sympy.org/latest/modules/integrals/integrals.html>`_,
        whereas adaptive makes use of adaptive quadrature through `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quadrature.html>`_.
        For most bilinear forms quadrature will also return the same matrix as exact.
        With non-constant measures (like for curvilinear coordinates) there is more
        likely to be differences.

    kind : None, str or dict, optional
        Alternative methods.

        For bilinear forms (matrices) kind is a string:

        - 'implemented' - Hardcoded implementations
        - 'stencil' - Use orthogonal bases and stencil-matrices
        - 'vandermonde' - Use Vandermonde matrix

        The default (for kind=None) is to first try to look for implemented kind,
        and if that fails try first 'stencil' and then finally fall back on
        vandermonde. Vandermonde creates a dense matrix of size NxN, so it should
        be avoided (e.g., by implementing the matrix) for large N.

        For linear forms (vectors) the kind keyword can be used to overload
        the default methods for transforms set in config['transforms']['kind'].
        Hence, kind is a dictionary with family as key and values either one of
        the possible methods

        - 'fast' - Use FFT (only Fourier and Chebyshev first and second kind)
        - 'recursive' - Low-memory implementation (only for Jacobi polynomials)
        - 'vandermonde' - Use Vandermonde matrix

        E.g., kind={'chebyshev': 'recursive'}.
        Note that for one-dimensional problems it is enough to use just the
        value of the dictionary.

    fixed_resolution : Number or sequence of integers, optional
        A fixed number of quadrature points used to compute the inner product
        along each dimension of the domain. If 'fixed_resolution' is set, then
        assemble is set to 'quadrature' and kind is set to 'vandermonde'.
        fixed_resolution is argument to :meth:`.TensorProductSpace.get_refined`.

    return_matrices : bool, optional
        For linear forms, whether to simply return the matrices that are used
        to compute the result with matrix-vector products.

    Note
    ----
    For most matrices all methods will lead to the same result. For bilinear
    forms with polynomial coefficients regular quadrature will become
    inaccurate and a 'fixed_resolution' higher than the regular number of
    quadrature points should be considered.

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

    >>> from shenfun import FunctionSpace, TensorProductSpace
    >>> from shenfun import TestFunction, TrialFunction, Array
    >>> SD = FunctionSpace(6, 'Chebyshev', bc=(0, 0))
    >>> u = TrialFunction(SD)
    >>> v = TestFunction(SD)
    >>> B = inner(v, u)
    >>> d = {-2: np.array([-np.pi/2]),
    ...       0: np.array([ 1.5*np.pi, np.pi, np.pi, np.pi]),
    ...       2: np.array([-np.pi/2])}
    >>> [np.all(abs(B[k]-v) < 1e-7) for k, v in d.items()]
    [True, True, True]

    # Compute unweighted integral
    >>> F = FunctionSpace(10, 'F', domain=(0, 2*np.pi))
    >>> T = TensorProductSpace(comm, (SD, F))
    >>> area = inner(1, Array(T, val=1))
    >>> print('Area of domain =', area)
    Area of domain = 12.56637061435917

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
        if isinstance(expr1, TestFunction):
            expr0 = sp.sympify(expr0)
        else:
            assert isinstance(expr1, (Array, Function))
            space = expr1.function_space()
            if isinstance(space, (TensorProductSpace, CompositeSpace)):
                df = np.prod(np.array([float(base.domain_factor()) for base in space.bases]))
            elif isinstance(space, SpectralBase):
                df = float(space.domain_factor())
            if isinstance(expr1, Function):
                #return (expr0/df)*dx(expr1.backward())
                expr1 = expr1.backward()
            if hasattr(space, 'hi'):
                if space.hi.prod() != 1:
                    expr1 = space.get_measured_array(expr1.copy())
            return (expr0/df)*dx(expr1)

    if isinstance(expr1, Number):
        if isinstance(expr0, TestFunction):
            expr1 = sp.sympify(expr1)
        else:
            assert isinstance(expr0, (Array, Function))
            space = expr0.function_space()
            if isinstance(space, (TensorProductSpace, CompositeSpace)):
                df = np.prod(np.array([float(base.domain_factor()) for base in space.bases]))
            elif isinstance(space, SpectralBase):
                df = float(space.domain_factor())
            if isinstance(expr0, Function):
                #return (expr1/df)*dx(expr0.backward())
                expr0 = expr0.backward()
            if hasattr(space, 'hi'):
                if space.hi.prod() != 1:
                    expr0 = space.get_measured_array(expr0.copy())
            return (expr1/df)*dx(expr0)

    if isinstance(expr0, tuple):
        assert isinstance(expr1, (Array, Function))
        space = expr1.function_space()
        assert isinstance(space, CompositeSpace)
        assert len(expr0) == len(space)
        result = 0.0
        for e0i, e1i in zip(expr0, expr1):
            result += inner(e0i, e1i)
        return result

    if isinstance(expr1, tuple):
        assert isinstance(expr0, (Array, Function))
        space = expr0.function_space()
        assert isinstance(space, CompositeSpace)
        assert len(expr1) == len(space)
        result = 0.0
        for e0i, e1i in zip(expr0, expr1):
            result += inner(e0i, e1i)
        return result

    if isinstance(expr0, sp.Expr) or isinstance(expr1, sp.Expr):
        # Special linear form using sympy expression
        expr_is_test = False
        if isinstance(expr0, sp.Expr):
            if isinstance(expr1, TestFunction):
                expr_is_test = True
            test = expr1
            trial = expr0
        else:
            if isinstance(expr0, TestFunction):
                expr_is_test = True
            test = expr0
            trial = expr1

        if expr_is_test:
            if output_array is None:
                output_array = Function(test.function_space())

            if assemble in ('exact', 'adaptive'):
                output_array = scalar_product(test, trial, output_array, assemble=assemble)

            else: # quadrature
                if fixed_resolution is not None:
                    from shenfun import comm
                    assert comm.Get_size() == 1
                    M = fixed_resolution
                    testM = test.function_space().get_refined(M)
                    outM = testM.scalar_product(Array(testM, buffer=trial), kind=kind)
                    output_array[test.function_space().slice()] = outM[test.function_space().slice()]
                else:
                    output_array = test.function_space().scalar_product(Array(test.function_space(), buffer=trial), output_array, kind=kind)
            return output_array
        else:
            expr0 = test
            expr1 = Function(test.function_space(), buffer=trial)

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

    recursive = test.function_space().is_composite_space
    if isinstance(trial, Array):
        assert Expr(trial).expr_rank() == test.expr_rank()
    elif isinstance(trial, BasisFunction):
        recursive *= (trial.expr_rank() > 0)
    if test.expr_rank() == 0:
        recursive = False

    if recursive: # Use recursive algorithm for vector expressions of expr_rank > 0, e.g., inner(v, grad(u))
        gij = test.function_space().coors.get_metric_tensor(config['basisvectors'])
        if trial.argument == 2 and not return_matrices:
            # linear form
            if output_array is None:
                output_array = Function(test.function_space())

            w0 = np.zeros(test.function_space().flatten()[0].shape(), dtype=output_array.dtype)
            if test.tensor_rank == 2:

                for i, (tei, xi) in enumerate(zip(test, output_array)):
                    for j, (teij, xij) in enumerate(zip(tei, xi)):
                        for k, trk in enumerate(trial):
                            if gij[i, k] == 0:
                                continue
                            for l, trkl in enumerate(trk):
                                if gij[j, l] == 0:
                                    continue

                                w0.fill(0)
                                xij += inner(teij*gij[i, k]*gij[j, l], trkl, output_array=w0, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution, return_matrices=return_matrices)

            elif test.tensor_rank == 1:
                for i, (te, x) in enumerate(zip(test, output_array)):
                    for j, tr in enumerate(trial):
                        if gij[i, j] == 0:
                            continue
                        w0.fill(0)
                        x += inner(te*gij[i, j], tr, output_array=w0, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution, return_matrices=return_matrices)

            return output_array

        result = []

        if test.tensor_rank == 2:
            for i, tei in enumerate(test):
                for j, teij in enumerate(tei):
                    for k, trk in enumerate(trial):
                        if gij[i, k] == 0:
                            continue
                        for l, trkl in enumerate(trk):
                            if gij[j, l] == 0:
                                continue
                            p = inner(teij*gij[i, k]*gij[j, l], trkl, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution, return_matrices=return_matrices)
                            result += p if isinstance(p, list) else [p]

        elif test.tensor_rank == 1:
            for i, te in enumerate(test):
                for j, tr in enumerate(trial):
                    if gij[i, j] == 0:
                        continue
                    l = inner(te, tr*gij[i, j], assemble=assemble, kind=kind, fixed_resolution=fixed_resolution, return_matrices=return_matrices)
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
            if trial.tensor_rank == 0 and isinstance(test, BasisFunction):
                output_array = space.scalar_product(trial, output_array, kind=kind)
                return output_array
            # project to orthogonal. Cannot use trial space, because the Array trial
            # may not fit with the boundary conditions of the trialspace
            orthogonal = trial.function_space().get_orthogonal()
            trial = Array(orthogonal, buffer=trial)
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

    gij = testspace.coors.get_metric_tensor(config['basisvectors'])

    A = []
    for vec_i, (base_test, test_ind) in enumerate(zip(test.terms(), test.indices())): # vector/scalar
        for vec_j, (base_trial, trial_ind) in enumerate(zip(trial.terms(), trial.indices())):
            g = 1 if len(test.terms()) == 1 else gij[vec_i, vec_j]
            if g == 0:
                continue
            for test_j, b0 in enumerate(base_test):              # second index test
                for trial_j, b1 in enumerate(base_trial):        # second index trial
                    dV = sp.simplify(test_scale[vec_i][test_j]*trial_scale[vec_j][trial_j]*testspace.coors.sg*g, measure=testspace.coors._measure)
                    dV = testspace.coors.refine(dV)

                    assert len(b0) == len(b1)
                    trial_sp = trialspace
                    if isinstance(trialspace, (CompositeSpace, MixedFunctionSpace)): # could operate on a vector, e.g., div(u), where u is vector
                        trial_sp = trialspace.flatten()[trial_ind[trial_j]]
                    test_sp = testspace
                    if isinstance(testspace, (CompositeSpace, MixedFunctionSpace)):
                        test_sp = testspace.flatten()[test_ind[test_j]]
                    has_bcs = False
                    # Check if scale is zero
                    if dV == 0:
                        continue
                    for dv in split(dV):
                        sc = dv['coeff']
                        M = []
                        DM = []
                        for i, (a, b) in enumerate(zip(b0, b1)): # Third index, one inner for each dimension
                            ts = trial_sp[i]
                            tt = test_sp[i]
                            msx = 'xyzrs'[i]
                            msi = dv[msx]

                            # assemble inner product
                            AA = inner_product((tt, a), (ts, b), msi, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
                            if len(AA) == 0:
                                sc = 0
                                continue

                            #if not abs(AA.scale-1.) < 1e-8:
                            #    AA.incorporate_scale()
                            M.append(AA)

                            if ts.has_nonhomogeneous_bcs:
                                tsc = ts.get_bc_space()
                                BB = inner_product((tt, a), (tsc, b), msi, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
                                #if not abs(BB.scale-1.) < 1e-8:
                                #    BB.incorporate_scale()
                                #if BB.diags('csr').nnz > 0:
                                if len(BB) > 0:
                                    DM.append(BB)
                                    has_bcs = True
                                else:
                                    DM.append(0)
                            else:
                                DM.append(AA)

                        if sc == 0:
                            continue
                        sc = tt.broadcast_to_ndims(np.array([sc]))

                        if len(M) == 1: # 1D case
                            M[0].global_index = (test_ind[test_j], trial_ind[trial_j])
                            M[0].scale = sc[0]
                            M[0].testbase = testspace
                            M[0].trialbase = trialspace
                            M[0].incorporate_scale()
                            A.append(M[0])
                        else:
                            A.append(TPMatrix(M, test_sp, trial_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace, trialspace))
                        if has_bcs:
                            if len(DM) == 1: # 1D case
                                DM[0].global_index = (test_ind[test_j], trial_ind[trial_j])
                                DM[0].scale = sc
                                DM[0].testbase = testspace
                                DM[0].trialbase = testspace
                                A.append(DM[0])
                            else:
                                if len(trial_sp.get_nonhomogeneous_axes()) == 1:
                                    A.append(TPMatrix(DM, test_sp, trial_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace, trialspace))
                                elif len(trial_sp.get_nonhomogeneous_axes()) == 2:
                                    if DM[1] != 0:
                                        A.append(TPMatrix([M[0], DM[1]], test_sp, trial_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace, trialspace))
                                    if DM[0] != 0:
                                        A.append(TPMatrix([DM[0], M[1]], test_sp, trial_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace, trialspace))
                                    if DM[0] != 0 and DM[1] != 0:
                                        A.append(TPMatrix(DM, test_sp, trial_sp, sc, (test_ind[test_j], trial_ind[trial_j]), testspace, trialspace))

    # At this point A contains all matrices of the form. The length of A is
    # the number of inner products. For each index into A there are ndim 1D
    # inner products along, e.g., x, y and z-directions, or just x, y for 2D.
    # The outer product of these matrices is a tensorproduct matrix, and we
    # store the matrices using the TPMatrix class.
    #
    # There are now two possibilities, either a linear or a bilinear form.
    # A linear form has trial.argument == 2, whereas a bilinear form has
    # trial.argument == 1. A linear form should assemble to an array and
    # return this array. A bilinear form, on the other hand, should return
    # matrices. Which matrices, and how many, will of course depend on the
    # form and the number of terms.

    # Bilinear form, return matrices
    if trial.argument == 1:
        return A[0] if len(A) == 1 else A

    if return_matrices:
        return A

    # Linear form, return output_array
    wh = np.zeros_like(output_array)
    for b in A:
        if uh.function_space().is_composite_space and wh.ndim == b.dimensions:
            wh = b.matvec(uh.v[b.global_index[1]], wh)
        elif uh.function_space().is_composite_space and wh.ndim > b.dimensions:
            wh[b.global_index[0]] = b.matvec(uh.v[b.global_index[1]], wh[b.global_index[0]])
        else:
            wh = b.matvec(uh, wh)
        output_array += wh
        wh.fill(0)
    return output_array

work = CachedArrayDict()

class Inner:
    """Return an instance of a class that can perform the inner product
    of the linear form efficiently through a matrix-vector product

    Parameters
    ----------
    v : :class:`.TestFunction`
    uh : Instance of either one of
        - :class:`.Expr`
        - :class:`.BasisFunction`

    Note
    ----
    This is an optimization only for linear forms, not bilinear.
    There is no need to use this class for regular scalar products, where `uh`
    is simply an Array.
    """
    def __init__(self, v, uh):
        from shenfun.matrixbase import get_simplified_tpmatrices
        assert isinstance(uh, (Expr, BasisFunction, Array))
        assert isinstance(v, TestFunction)
        assert uh.argument == 2
        self.uh = [uh]
        A = inner(v, uh, return_matrices=True)
        if isinstance(A[0], TPMatrix):
            A = get_simplified_tpmatrices(A)
        elif isinstance(A, Function):
            A = [A]
        self.A = [A]
        self.output_array = Function(v.function_space())

    def __call__(self):
        wh = work[(self.output_array, 0, True)]
        self.output_array.fill(0)
        for uh, A in zip(self.uh, self.A):
            uh = uh.base if uh.base is not None else uh
            for b in A:
                if isinstance(b, Function) and isinstance(uh, Array):
                    V = b.function_space()
                    wh = V.scalar_product(uh, wh)
                    self.output_array += wh
                elif uh.function_space().is_composite_space and wh.ndim == b.dimensions:
                    wh = b.matvec(uh.v[b.global_index[1]], wh)
                    self.output_array += wh
                elif uh.function_space().is_composite_space and wh.ndim > b.dimensions:
                    wh[b.global_index[0]] = b.matvec(uh.v[b.global_index[1]], wh[b.global_index[0]])
                    self.output_array.v[b.global_index[0]] += wh[b.global_index[0]]
                else:
                    wh = b.matvec(uh, wh)
                    self.output_array += wh
                wh.fill(0)
        return self.output_array

    def __add__(self, c):
        assert isinstance(c, Inner)
        assert c.output_array.function_space() == self.output_array.function_space()
        self.A += c.A
        self.uh += c.uh
        return self
