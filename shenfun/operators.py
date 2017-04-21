import numpy as np
from .arguments import Expr, Function
from .tensorproductspace import VectorTensorProductSpace

__all__ = ('div', 'grad', 'Dx')


# operators
def div(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    v = test.integrals().copy()
    sc = test.scales().copy()

    #assert test.rank() == len(space)  # vector
    if len(space) == 1:      # 1D
        v += 1

    else:
        ndim = len(space)
        for i, s in enumerate(v):
            s[:, i%ndim] += 1
        v = v.reshape((v.shape[0]//ndim, v.shape[1]*ndim, ndim))
        sc = sc.reshape((sc.shape[0]//ndim, sc.shape[1]*ndim))

    if test.argument() < 2:
        return Expr(space, test.argument(), v, sc)

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v
        f._scales = sc
        return f


def grad(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    vspace = VectorTensorProductSpace(space.comm, space.bases,
                                      [a[0] for a in space.axes], space.dtype)

    integrals = test.integrals()
    sc = test.scales()

    ndim = len(space)
    assert test.dim() == ndim
    #assert test.rank() == 1       # allow only gradient of scalar
    v = np.repeat(integrals, ndim, axis=0) # Create vector
    sc = np.repeat(sc, ndim, axis=0)
    for i, s in enumerate(v):
        s[:, i%ndim] += 1

    if test.argument() < 2:
        return Expr(vspace, test.argument(), v, sc)

    elif test.argument() == 2:
        f = Function(vspace, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v
        f._scales = sc
        return f


def Dx(test, x, k):
    assert isinstance(test, Expr)
    space = test.function_space()
    v = test.integrals().copy()
    sc = test.scales().copy()

    v[:, :, x] += k

    if test.argument() < 2:
        return Expr(space, test.argument(), v, sc)

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v
        f._scales = sc
        return f


def curl(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    assert test.rank() == len(space)  # vector
    v = test.integrals().copy()
    sc = test.scales().copy()
    ndim = len(space)
    assert ndim > 1

    w0 = Dx(v[1], 2, 1) - Dx(v[2], 1, 1)
    w1 = Dx(v[2], 0, 1) - Dx(v[0], 2, 1)
    w2 = Dx(v[0], 1, 1) - Dx(v[1], 0, 1)
    v[0] = w0.integrals()[0]
    v[1] = w1.integrals()[0]
    v[2] = w2.integrals()[0]
    sc[0] = w0.scales()[0]
    sc[1] = w1.scales()[0]
    sc[2] = w2.scales()[0]

    if test.argument() < 2:
        return Expr(space, test.argument(), v, sc)

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._integrals = v
        f._scales = sc
        return f
