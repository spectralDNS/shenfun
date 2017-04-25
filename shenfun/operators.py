import numpy as np
from .arguments import Expr, Function
from .tensorproductspace import VectorTensorProductSpace

__all__ = ('div', 'grad', 'Dx', 'curl')


# operators
def div(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    v = test.terms().copy()
    sc = test.scales().copy()
    ind = test.indices().copy()

    ndim = space.ndim()
    if ndim == 1:      # 1D
        v += 1

    else:
        for i, s in enumerate(v):
            s[:, i%ndim] += 1
        v = v.reshape((v.shape[0]//ndim, v.shape[1]*ndim, ndim))
        sc = sc.reshape((sc.shape[0]//ndim, sc.shape[1]*ndim))
        ind = ind.reshape((ind.shape[0]//ndim, ind.shape[1]*ndim))

    if test.argument() < 2:
        return Expr(space, test.argument(), v, sc, ind)

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._terms = v
        f._scales = sc
        f._indices = ind
        return f


def grad(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    terms = test.terms()
    sc = test.scales()
    ind = test.indices()

    ndim = space.ndim()
    assert test.dim() == ndim
    #assert test.num_components() == 1       # allow only gradient of scalar
    v = np.repeat(terms, ndim, axis=0)       # Create vector
    sc = np.repeat(sc, ndim, axis=0)
    ind = np.repeat(ind, ndim, axis=0)
    for i, s in enumerate(v):
        s[:, i%ndim] += 1

    if test.argument() < 2:
        return Expr(space, test.argument(), v, sc, ind)

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._terms = v
        f._scales = sc
        f._indices = ind
        return f


def Dx(test, x, k):
    assert isinstance(test, Expr)
    space = test.function_space()
    v = test.terms().copy()
    sc = test.scales().copy()

    v[:, :, x] += k

    if test.argument() < 2:
        return Expr(space, test.argument(), v, sc, test.indices())

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._terms = v
        f._scales = sc
        f._indices = test.indices()
        return f


def curl(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    assert test.rank() == 2
    assert test.num_components() == test.dim()  # vector
    v = test.terms().copy()
    sc = test.scales().copy()
    ndim = space.ndim()
    assert ndim > 1

    w0 = Dx(test[1], 2, 1) - Dx(test[2], 1, 1)
    w1 = Dx(test[2], 0, 1) - Dx(test[0], 2, 1)
    w2 = Dx(test[0], 1, 1) - Dx(test[1], 0, 1)
    w = np.concatenate((w0.terms(), w1.terms(), w2.terms()), axis=0)
    sc = np.concatenate((w0.scales(), w1.scales(), w2.scales()), axis=0)
    indices = np.concatenate((w0.indices(), w1.indices(), w2.indices()), axis=0)

    if test.argument() < 2:
        return Expr(space, test.argument(), w, sc, indices)

    elif test.argument() == 2:
        f = Function(space, forward_output=space.is_forward_output(test))
        f[:] = test
        f._terms = w
        f._scales = sc
        f._indices = indices
        return f

