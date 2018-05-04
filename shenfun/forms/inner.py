"""
This module contains the inner function that computes the
weighted inner product.
"""
import numpy as np
from shenfun.fourier import FourierBase
from shenfun.spectralbase import inner_product
from shenfun.la import DiagonalMatrix
from shenfun.tensorproductspace import MixedTensorProductSpace
from .arguments import Expr, Function, BasisFunction, Array

__all__ = ('inner',)

#pylint: disable=line-too-long,inconsistent-return-statements,too-many-return-statements

def inner(expr0, expr1, output_array=None, uh_hat=None):
    """Return inner product of linear or bilinear form

    Parameters
    ----------
        expr0, expr1:  Expr
                       Test/trial function, or expression involving test/trial
                       function, e.g., div, grad. One of expr0 or expr1 need to
                       be an expression on a testfunction, and if the second
                       then involves a trial function, a matrix is returned.
                       If one of expr0/expr1 is a test function and the other
                       one is a Function/Array, then a linear form is assumed
                       and an assembled vector is returned

        output_array:  Numpy array
                       Optional return array for linear form.

        uh_hat:        Numpy array
                       The transform of the Function/Array used for linear forms.

    Example
    -------
        Compute mass matrix of Shen's Chebyshev Dirichlet basis:

        >>> from shenfun.chebyshev.bases import ShenDirichletBasis
        >>> from shenfun.forms.arguments import TestFunction, TrialFunction
        >>> import six
        >>> SD = ShenDirichletBasis(6)
        >>> u = TrialFunction(SD)
        >>> v = TestFunction(SD)
        >>> B = inner(v, u)
        >>> d = {-2: np.array([-np.pi/2]),
        ...       0: np.array([ 1.5*np.pi, np.pi, np.pi, np.pi]),
        ...       2: np.array([-np.pi/2])}
        >>> [np.all(abs(B[k]-v) < 1e-7) for k, v in six.iteritems(d)]
        [True, True, True]

    """
    # Wrap numpy array in Function
    if not hasattr(expr0, 'argument'):
        if isinstance(expr0, np.ndarray):
            try:
                expr0 = Function(expr1.function_space(), forward_output=False, buffer=expr0)
            except:
                raise RuntimeError

    if not hasattr(expr1, 'argument'):
        if isinstance(expr1, np.ndarray):
            try:
                expr1 = Function(expr0.function_space(), forward_output=False, buffer=expr1)
            except:
                raise RuntimeError

    t0 = expr0.argument()
    t1 = expr1.argument()
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

    if test.rank() == 2: # For vector spaces of rank 2 use recursive algorithm
        ndim = test.function_space().ndim()

        uh = uh_hat
        if uh is None and trial.argument() == 2:
            uh = Array(trial.function_space(), forward_output=True)
            basis = trial if isinstance(trial, np.ndarray) else trial.basis()
            uh = trial.function_space().forward(basis, uh)

        if output_array is None and trial.argument() == 2:
            output_array = Array(test.function_space())

        if trial.argument() == 2:
            # linear form
            for ii in range(ndim):
                output_array[ii] = inner(test[ii], trial[ii],
                                         output_array=output_array[ii],
                                         uh_hat=uh)
            return output_array

        result = []
        for ii in range(ndim):
            result.append(inner(test[ii], trial[ii], uh_hat=uh))
        return result


    if trial.argument() > 1:
        # Linear form
        assert isinstance(test, (Expr, BasisFunction))
        assert test.argument() == 0
        space = test.function_space()
        if isinstance(trial, np.ndarray):
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
    trialspace = trial.function_space()
    assert test.num_components() == trial.num_components()
    test_scale = test.scales()
    trial_scale = trial.scales()
    trial_indices = trial.indices()

    uh = uh_hat
    if uh is None and trial.argument() == 2:
        uh = Array(trialspace, forward_output=True)
        uh = trialspace.forward(trial.basis(), uh)

    if output_array is None and trial.argument() == 2:
        output_array = Array(trial.function_space())

    A = []
    S = []
    vec = 0
    for base_test, base_trial in zip(test.terms(), trial.terms()): # vector/scalar
        for test_j, b0 in enumerate(base_test):              # second index test
            for trial_j, b1 in enumerate(base_trial):        # second index trial
                sc = test_scale[vec, test_j]*trial_scale[vec, trial_j]
                A.append([])
                assert len(b0) == len(b1)
                for i, (a, b) in enumerate(zip(b0, b1)): # Third index, one inner for each dimension
                    ts = trialspace[i]
                    if isinstance(trialspace, MixedTensorProductSpace): # trial could operate on a vector, e.g., div(u), where u is vector
                        ts = ts[i]
                    AA = inner_product((space[i], a), (ts, b))
                    A[-1].append(AA)
                    # Take care of domains of not standard size
                    if not space[i].domain_factor() == 1:
                        sc *= space[i].domain_factor()**(a+b)
                S.append(np.array([sc]))

        vec += 1

    # At this point A contains all matrices of the form. The length of A is
    # the number of inner products. For each index into A there are ndim 1D
    # inner products along, e.g., x, y and z-directions, or just x, y for 2D.
    # The ndim matrices are multiplied with each other, and diagonal matrices
    # can be eliminated and put in a scale array for the non-diagonal matrices
    # E.g. (v, div(grad(u))) in 2D
    #
    # Here A = [[(v[0], u[0]'')_x, (v[1], u[1])_y,
    #            (v[0], u[0])_x, (v[1], u[1]'')_y ]]
    #
    # where v[0], v[1] are the test functions in x- and y-directions, respectively
    # For example, v[0] could be a ShenDirichletBasis and v[1] could be a
    # FourierBasis. Same for u.
    #
    # There are now two possibilities, either a linear or a bilinear form.
    # A linear form has trial.argument() == 2, whereas a bilinear form has
    # trial.argument() == 1. A linear form should assemble to an array and
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

    # Strip off diagonal matrices, put contribution in scale array
    B = []
    for sc, matrices in zip(S, A):
        scale = sc.reshape((1,)*space.ndim())
        nonperiodic = {}
        for axis, mat in enumerate(matrices):
            if isinstance(space[axis], FourierBase):
                mat = mat[0]    # get diagonal
                if np.ndim(mat):
                    mat = space[axis].broadcast_to_ndims(mat, space.ndim(), axis)

                scale = scale*mat

            else:
                mat.axis = axis
                nonperiodic[axis] = mat

        # Decomposition
        if hasattr(space, 'local_slice'):
            s = scale.shape
            ss = [slice(None)]*space.ndim()
            ls = space.local_slice()
            for axis, shape in enumerate(s):
                if shape > 1:
                    ss[axis] = ls[axis]
            scale = (scale[ss]).copy()

        if len(nonperiodic) is 0:
            # All diagonal matrices
            B.append(scale)

        else:
            nonperiodic['scale'] = scale
            B.append(nonperiodic)

    # At this point assembled matrices are in the B list. One item per term, same
    # as A. However, now the Fourier matrices have been contracted into the Numpy
    # array 'scale', of shape determined by decomposition.
    # The final step here is to add equal matrices together, and to compute the
    # output for linear forms.

    if np.all([isinstance(b, np.ndarray) for b in B]):

        # All Fourier
        if space.ndim() == 1:
            if trial.argument() == 1:
                A[0][0].axis = 0
                return A[0][0]

            output_array[:] = B[0]*uh
            return output_array

        else:
            if trial.argument() == 1:
                diagonal_array = B[0]
                for ci in B[1:]:
                    diagonal_array = diagonal_array + ci

                return DiagonalMatrix(diagonal_array)

            else:
                if uh.rank() == 2:
                    for i, b in enumerate(B):
                        output_array += b*uh[trial_indices[0, i]]

                else:
                    diagonal_array = B[0]
                    for ci in B[1:]:
                        diagonal_array = diagonal_array + ci
                    output_array[:] = diagonal_array*uh

                return output_array

    elif np.all([len(f) == 2 for f in B]):
        # Only one nonperiodic direction

        npaxis = [b for b in B[0].keys() if isinstance(b, int)][0]

        ## 1D case for itself, because it is simpler
        #if space.ndim() == 1:
            #from IPython import embed; embed()
            #if trial.argument() == 1:
                #if len(B) == 1:
                    #b = B[0][npaxis]
                    #b.scale = B[0]['scale']
                    #return b
                #else:

            #else:
                #mat = B[0][npaxis]
                #mat.scale = B[0]['scale']
                #output_array = mat.matvec(uh, output_array, axis=npaxis)
                #output_array *= mat.scale
                #return output_array

        if trial.argument() == 1:  # bilinear form
            b = B[0][npaxis]
            b.scale = B[0]['scale']
            C = {b.get_key(): b}
            for bb in B[1:]:
                b = bb[npaxis]
                b.scale = bb['scale']
                name = b.get_key()
                if name in C:
                    C[name].scale = C[name].scale + b.scale
                else:
                    C[name] = b

            if len(C) == 1:
                return C[b.get_key()]
            return C

        else: # linear form
            for i, bb in enumerate(B):
                b = bb[npaxis]
                if uh.rank() == 2:
                    sp = uh.function_space()
                    wh = Array(sp[npaxis], forward_output=True)
                    wh = b.matvec(uh[trial_indices[0, i]], wh, axis=b.axis)

                else:
                    wh = Array(trialspace, forward_output=True)
                    wh = b.matvec(uh, wh, axis=b.axis)
                output_array += wh*bb['scale']

            return output_array

    elif np.all([len(f) == 3 for f in B]):
        # Two nonperiodic directions

        if trial.argument() == 1:  # bilinear form
            return B

        else: # linear form
            npaxes = [b for b in B[0].keys() if isinstance(b, int)]

            pencilA = space.forward.output_pencil
            subcomms = [c.Get_size() for c in pencilA.subcomm]
            axis = pencilA.axis
            assert subcomms[axis] == 1
            npaxes.remove(axis)
            second_axis = npaxes[0]
            pencilB = pencilA.pencil(second_axis)
            transAB = pencilA.transfer(pencilB, 'd')

            # Output data is aligned in axis, but may be distributed in all other directions

            if uh.rank() == 2:
                sp = uh.function_space()
                wh = Array(sp[axis], forward_output=True)
                wc = Array(sp[axis], forward_output=True)

            else:
                wh = Array(trialspace, forward_output=True)
                wc = Array(trialspace, forward_output=True)

            whB = np.zeros(transAB.subshapeB)
            wcB = np.zeros(transAB.subshapeB)

            for i, bb in enumerate(B):
                if uh.rank() == 2:
                    wc[:] = uh[trial_indices[0, i]]
                else:
                    wc[:] = uh

                b = bb[axis]
                wh = b.matvec(wc, wh, axis=axis)

                # align in second non-periodic axis
                transAB.forward(wh, whB)
                b = bb[second_axis]
                wcB = b.matvec(whB, wcB, axis=second_axis)
                transAB.backward(wcB, wh)
                wh *= bb['scale']
                output_array += wh

            return output_array
