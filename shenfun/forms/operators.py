"""
This module contains the implementation of operators acting on arguments.
"""
import numpy as np
from .arguments import Expr, BasisFunction

__all__ = ('div', 'grad', 'Dx', 'curl')

#pylint: disable=protected-access

def div(test):
    """Return div(test)

    Parameters
    ----------
    test:  Expr or BasisFunction
           Must be rank > 0 (cannot take divergence of scalar)

    """
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(test, BasisFunction):
        test = Expr(test)

    v = test.terms().copy()
    sc = test.scales().copy()
    ind = test.indices().copy()

    ndim = test.dimensions
    if ndim == 1:      # 1D
        v += 1

    else:
        for i, s in enumerate(v):
            s[:, i%ndim] += 1
        v = v.reshape((v.shape[0]//ndim, v.shape[1]*ndim, ndim))
        sc = sc.reshape((sc.shape[0]//ndim, sc.shape[1]*ndim))
        ind = ind.reshape((ind.shape[0]//ndim, ind.shape[1]*ndim))

    test._terms = v
    test._scales = sc
    test._indices = ind
    return test


def grad(test):
    """Return grad(test)

    Parameters
    ----------
    test: Expr or BasisFunction
    """
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(test, BasisFunction):
        test = Expr(test)

    terms = test.terms()
    sc = test.scales()
    ind = test.indices()

    ndim = test.dimensions
    #assert test.num_components() == 1       # allow only gradient of scalar
    test._terms = np.repeat(terms, ndim, axis=0)       # Create vector
    test._scales = np.repeat(sc, ndim, axis=0)
    test._indices = np.repeat(ind, ndim, axis=0)
    for i, s in enumerate(test._terms):
        s[:, i%ndim] += 1

    return test


def Dx(test, x, k=1):
    """Return k'th order partial derivative in direction x

    Parameters
    ----------
    test: Expr or BasisFunction
    x:  int
        axis to take derivative over
    k:  int
        Number of derivatives
    """
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(test, BasisFunction):
        test = Expr(test)

    v = test.terms().copy()
    v[:, :, x] += k
    test._terms = v
    return test


def curl(test):
    """Return curl of test

    Parameters
    ----------
    test: Expr or BasisFunction

    """
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(test, BasisFunction):
        test = Expr(test)

    assert test.expr_rank() > 0
    assert test.num_components() == test.dimensions  # vector

    if test.dimensions == 3:
        w0 = Dx(test[2], 1, 1) - Dx(test[1], 2, 1)
        w1 = Dx(test[0], 2, 1) - Dx(test[2], 0, 1)
        w2 = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)
        test._terms = np.concatenate((w0.terms(), w1.terms(), w2.terms()), axis=0)
        test._scales = np.concatenate((w0.scales(), w1.scales(), w2.scales()), axis=0)
        test._indices = np.concatenate((w0.indices(), w1.indices(), w2.indices()), axis=0)
    else:
        assert test.dimensions == 2
        test = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)

    return test
