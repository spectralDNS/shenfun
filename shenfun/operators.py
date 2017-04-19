import numpy as np
from .arguments import Expr, Function

__all__ = ('div', 'grad', 'Dx')


# operators
def div(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    integrals = test.integrals()

    assert integrals.shape[0] == len(space)  # vector
    if len(space) == 1:      # 1D
        v = integrals.copy()
        v += 1
    else:
        ndim = len(space)
        v = integrals.copy()
        for i, s in enumerate(v):
            s[..., i] += 1
        v = v.reshape((1, np.prod(v.shape[:-1]), ndim))

    if test.argument() < 2:
        return Expr(space, v, test.argument())

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v.copy()
        return f


def grad(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    integrals = test.integrals()

    ndim = len(space)
    assert test.dim() == ndim
    assert test.rank() == 1       # can only take gradient of scalar (for now)
    ss = list(integrals.shape)
    ss[0] *= ndim
    v = np.broadcast_to(integrals, ss).copy()  # vector
    for i, s in enumerate(v):
        s[..., i%ndim] += 1

    if test.argument() < 2:
        return Expr(space, v, test.argument())

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v.copy()
        return f


def Dx(test, x, k):
    assert isinstance(test, Expr)
    space = test.function_space()
    integrals = test.integrals()

    v = integrals.copy()
    for comp in v:
        for i in comp:
            i[x] += k

    if test.argument() < 2:
        return Expr(space, v, test.argument())

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v
        return f

