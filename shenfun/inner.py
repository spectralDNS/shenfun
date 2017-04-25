import numpy as np
import six
from shenfun.fourier import FourierBase
from .arguments import Expr
from .spectralbase import inner_product
from .arguments import TestFunction, TrialFunction, Function

__all__ = ('inner', 'project')


def inner(expr0, expr1, output_array=None, uh_hat=None):
    """Return inner product of linear or bilinear form

    args:
        expr0/expr1     Expr          Test/trial function, or expression
                                      involving test/trial function, e.g., div,
                                      grad. One of expr0 or expr1 need to be an
                                      expression on a testfunction, and if the
                                      second then involves a trial function, a
                                      matrix is returned

                                      If one of expr0/expr1 is a test function
                                      and the other one is a Function, then a
                                      linear form is assumed and an assembled
                                      vector is returned

    kwargs:
        output_array  Numpy array     Optional return array for linear form.

        uh_hat                        The transform of the Function used for
                                      linear forms.

    Example:
        Compute mass matrix of Shen's Chebyshev Dirichlet basis:

        >>> from shenfun.chebyshev.bases import ShenDirichletBasis
        >>> from shenfun.arguments import TestFunction, TrialFunction
        >>>
        >>> SD = ShenDirichletBasis(6)
        >>> u = TrialFunction(SD)
        >>> v = TestFunction(SD)
        >>> B = inner(v, u)
        >>> B
        {-2: array([-1.57079633]),
          0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265]),
          2: array([-1.57079633])}

    """
    if expr0.rank() > 1: # For vector spaces of rank 2 use recursive algorithm
        assert expr0.rank() == 2
        assert expr1.rank() == 2
        ndim = expr0.function_space().ndim()
        result = []
        for ii in range(ndim):
            result.append(inner(expr0[ii], expr1[ii]))
        if np.all([isinstance(a, np.ndarray) for a in result]):
            return Function(expr0.function_space(), buffer=np.array(result))
        else:
            return result

    if isinstance(expr0, np.ndarray) or isinstance(expr1, np.ndarray):
        # Linear form
        if isinstance(expr0, np.ndarray):
            fun = expr0
            test = expr1
        else:
            fun = expr1
            test = expr0
        assert isinstance(test, Expr)
        space = test.function_space()
        if not isinstance(fun, Expr):
            output_array = space.scalar_product(fun, output_array)
            return output_array
        else:
            ndim = space.ndim()
            if np.all(fun.terms() == np.zeros((1, 1, ndim), dtype=np.int)):
                output_array = space.scalar_product(fun, output_array)
                return output_array

        # If fun is an Expr with terms, then compute using bilinear form and matvec
        expr0 = test
        expr1 = fun

    assert isinstance(expr0, Expr)
    assert isinstance(expr1, Expr)
    t0 = expr0.argument()
    t1 = expr1.argument()
    if t0 == 0:
        assert t1 in (1, 2)
        test = expr0
        trial = expr1
    elif t0 == 1:
        assert t1 == 0
        test = expr1
        trial = expr0
    else:
        raise RuntimeError

    space = test.function_space()
    trialspace = trial.function_space()
    assert test.num_components() == trial.num_components()
    test_scale = test.scales()
    trial_scale = trial.scales()

    A = []
    S = []
    vec = 0
    for base_test, base_trial in zip(test.terms(), trial.terms()): # vector/scalar
        for test_j, b0 in enumerate(base_test):              # second index test
            for trial_j, b1 in enumerate(base_trial):        # second index trial
                sc = test_scale[vec, test_j]*trial_scale[vec, trial_j]
                A.append([])
                S.append(np.array([sc]))
                assert len(b0) == len(b1)
                for i, (a, b) in enumerate(zip(b0, b1)): # Third index, one inner for each dimension
                    AA = inner_product((space[i], a), (trialspace[i], b))
                    if isinstance(space[i], FourierBase):
                        d = AA[0]
                        if np.ndim(d):
                            d = space[i].broadcast_to_ndims(d, space.ndim(), i)
                    else:
                        d = AA
                    A[-1].append(d)
        vec += 1

    # Strip off diagonal matrices, put contribution in scale array
    B = []
    for sc, matrices in zip(S, A):
        scale = sc.reshape((1,)*space.ndim())
        nonperiodic = {}
        for axis, mat in enumerate(matrices):
            if isinstance(space[axis], FourierBase):
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

    # Get local matrices
    if np.all([isinstance(b, np.ndarray) for b in B]):

        # All Fourier
        if len(B) == 1:
            if trial.argument() == 1:
                return A[0][0]
            else:
                uh = uh_hat
                if uh is None:
                    uh = Function(trialspace, forward_output=True)
                    uh = trialspace.forward(trial, uh)
                return A[0][0]*uh

        else:
            diagonal_array = B[0]
            for ci in B[1:]:
                diagonal_array = diagonal_array + ci

            if trial.argument() == 1:
                diagonal_array = np.where(diagonal_array==0, 1, diagonal_array)
                return {'diagonal': diagonal_array}

            else:
                uh = uh_hat
                if uh is None:
                    uh = Function(trialspace, forward_output=True)
                    uh = trialspace.forward(trial, uh)
                return diagonal_array*uh

    else:

        # For now only allow one nonperiodic direction
        assert np.all([len(f) == 2 for f in B])
        if space.ndim() > 1:
            for axis, sp in enumerate(space):
                if not isinstance(sp, FourierBase):
                    npaxis = axis
        else:
            npaxis = 0

        if len(B) == 1:
            if trial.argument() == 1:
                b = B[0][npaxis]
                b.scale = B[0]['scale']
                return b
            else:
                uh = uh_hat
                if uh is None:
                    uh = Function(trialspace, forward_output=True)
                    uh = trialspace.forward(trial, uh)
                vh = Function(space, forward_output=True)
                mat = B[0][npaxis]
                mat.scale = B[0]['scale']
                vh = mat.matvec(uh, vh, axis=npaxis)
                vh *= mat.scale
                return vh

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

        if trial.argument() == 1:
            return C

        else:
            uh = uh_hat
            if uh is None:
                uh = Function(trialspace, forward_output=True)
                uh = trialspace.forward(trial, uh)

            vh = Function(space, forward_output=True)
            wh = Function(space, forward_output=True)
            for key, val in six.iteritems(C):
                wh = val.matvec(uh, wh, axis=val.axis)
                vh += wh*val.scale

            return vh


#def inner_vector(test, trial):
    #assert test.rank() == 2
    #assert test.rank() == trial.rank()

    #A = []
    #S = []
    #ndim = test.function_space().ndim()
    #for ii in range(ndim):
        #A.append([])
        #S.append([])
        #space = test[ii].function_space()
        #trialspace = trial[ii].function_space()
        #assert test[ii].num_components() == trial[ii].num_components()
        #test_scale = test.scales()
        #trial_scale = trial.scales()
        #vec = 0
        #for base_test, base_trial in zip(test.terms(), trial.terms()): # vector/scalar
            #for test_j, b0 in enumerate(base_test):              # second index test
                #for trial_j, b1 in enumerate(base_trial):        # second index trial
                    #sc = test_scale[vec, test_j]*trial_scale[vec, trial_j]
                    #A[-1].append([])
                    #S[-1].append(np.array([sc]))
                    #assert len(b0) == len(b1)
                    #for i, (a, b) in enumerate(zip(b0, b1)): # Third index, one inner for each dimension
                        #AA = inner_product((space[i], a), (trialspace[i], b))
                        #if isinstance(space[i], FourierBase):
                            #d = AA[0]
                            #if np.ndim(d):
                                #d = space[i].broadcast_to_ndims(d, space.ndim(), i)
                        #else:
                            #d = AA
                        #A[-1][-1].append(d)
            #vec += 1
    #return A


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

    assert uh.shape == T.forward.input_array.shape
    assert isinstance(uh, np.ndarray)
    assert isinstance(uh, Expr)

    if output_array is None:
        output_array = Function(T)

    if np.all(uh.terms() == np.zeros((1, 1, len(T)), dtype=np.int)):
        # Just regular forward transform
        output_array = T.forward(uh, output_array)
        return output_array

    v = TestFunction(T)
    u = TrialFunction(T)
    u_hat = inner(v, uh, uh_hat=uh_hat)
    B = inner(v, u)
    #from IPython import embed; embed()
    output_array = B.solve(u_hat, output_array, axis=B.axis)
    return output_array
