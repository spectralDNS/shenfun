"""
This module contains the implementation of operators acting on arguments.
"""
import numpy as np
import sympy as sp
import copy
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

    test = copy.copy(test)

    v = test.terms().copy()
    sc = test.scales().copy()
    ind = test.indices().copy()
    measures = test.measures().copy()
    ndim = test.dimensions
    hi = np.ones(1, dtype=np.int)
    if ndim > 1:
        hi = test.function_space().hi

    if hi.prod() == 1:
        # Cartesian
        if ndim == 1:      # 1D
            v += 1

        else:
            for i, s in enumerate(v):
                s[..., i%ndim] += 1
            v = v.reshape((v.shape[0]//ndim, v.shape[1]*ndim, ndim))
            sc = sc.reshape((sc.shape[0]//ndim, sc.shape[1]*ndim))
            ind = ind.reshape((ind.shape[0]//ndim, ind.shape[1]*ndim))
            measures = measures.reshape((measures.shape[0]//ndim, measures.shape[1]*ndim))

        test._terms = v
        test._scales = sc
        test._indices = ind
        test._measures = measures
        return test

    else:
        assert test.expr_rank() < 2, 'Cannot take divergence of higher order tensor in curvilinear coordinates'

        if ndim == 1:      # 1D
            v += 1

        else:
            v = np.repeat(v, 2, axis=1)
            sc = np.repeat(sc, 2, axis=1)
            ind = np.repeat(ind, 2, axis=1)
            measures = np.repeat(measures, 2, axis=1)
            psi = test.function_space().measures[0]

            for i, s in enumerate(v):
                ll = [k for k in range(ndim) if not k==i]
                for j in range(v.shape[1]):
                    if j%2 == 0:
                        s[j, i%ndim] += 1
                        measures[i, j] = measures[i, j] / hi[i%ndim]
                    else:
                        ms2 = measures[i, j]*np.take(hi, ll).prod()
                        measures[i, j] = ms2.diff(psi[i%ndim], 1) / hi.prod()

            v = v.reshape((v.shape[0]//ndim, v.shape[1]*ndim, ndim))
            sc = sc.reshape((sc.shape[0]//ndim, sc.shape[1]*ndim))
            ind = ind.reshape((ind.shape[0]//ndim, ind.shape[1]*ndim))
            measures = measures.reshape((measures.shape[0]//ndim, measures.shape[1]*ndim))
        test._terms = v
        test._scales = sc
        test._indices = ind
        test._measures = measures
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

    test = copy.copy(test)

    terms = test.terms().copy()
    sc = test.scales().copy()
    ind = test.indices().copy()
    measures = test.measures().copy()
    ndim = test.dimensions
    hi = np.ones(1, dtype=np.int)
    if ndim > 1:
        hi = test.function_space().hi

    if hi.prod() != 1:
        assert test.expr_rank() < 1, 'Cannot take gradient of tensor in curvilinear coordinates'

    if hi.prod() == 1:
        # Cartesian
        test._terms = np.repeat(terms, ndim, axis=0)
        test._scales = np.repeat(sc, ndim, axis=0)
        test._indices = np.repeat(ind, ndim, axis=0)
        test._measures = np.repeat(measures, ndim, axis=0)
        for i, s in enumerate(test._terms):
            s[..., i%ndim] += 1
        return test

    elif measures.flatten().prod() == 1:
        # If expr taken gradient of has measure 1

        test._terms = np.repeat(terms, ndim, axis=0)
        test._scales = np.repeat(sc, ndim, axis=0)
        test._indices = np.repeat(ind, ndim, axis=0)
        test._measures = np.repeat(measures, ndim, axis=0)
        for i, s in enumerate(test._terms):
            s[..., i%ndim] += 1

        for i, s in enumerate(test._measures):
            s[:] /= hi[i%ndim]
        return test

    else:
        # Expr taken gradient of has measures itself
        terms = np.repeat(terms, ndim, axis=0)
        test._terms = np.repeat(terms, 2, axis=1)
        N = test._terms.shape
        sc = np.repeat(sc, ndim, axis=0)
        test._scales = np.repeat(sc, 2, axis=1)
        ind = np.repeat(ind, ndim, axis=0)
        test._indices = np.repeat(ind, 2, axis=1)
        measures = np.repeat(measures, ndim, axis=0)
        test._measures = np.repeat(measures, 2, axis=1)
        psi = test.function_space().measures[0]

        for i, s in enumerate(test._terms):
            for j in range(test._terms.shape[1]):
                if j % 2 == 0:
                    s[j, i%ndim] += 1
                    test._measures[i, j] /= hi[i%ndim]
                else:
                    test._measures[i, j] = test._measures[i, j].diff(psi[i%ndim], 1) / hi[i%ndim]

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

    test = copy.copy(test)
    ndim = test.dimensions
    hi = np.ones(1, dtype=np.int)
    if ndim > 1:
        hi = test.function_space().hi

    if hi.prod() == 1:
        v = test.terms().copy()
        v[..., x] += k
        test._terms = v

    else:
        assert test.expr_rank() < 1, 'Cannot take derivative of tensor in curvilinear coordinates'
        v = test._terms = np.repeat(test.terms(), 2, axis=1)
        sc = test._scales = np.repeat(test.scales(), 2, axis=1)
        ind = test._indices = np.repeat(test.indices(), 2, axis=1)
        measures = test._measures = np.repeat(test.measures(), 2, axis=1)
        psi = test.function_space().measures[0]
        for i in range(v.shape[1]):
            if i % 2 == 0:
                v[:, i, x] += k
            else:
                measures[:, i] = sp.diff(measures[:, i], psi[x], 1)

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

    test = copy.copy(test)

    assert test.expr_rank() > 0
    assert test.num_components() == test.dimensions  # vector

    measures = test.measures().copy()
    hi = test.function_space().hi

    if hi.prod() == 1:
        # Cartesian
        if test.dimensions == 3:
            w0 = Dx(test[2], 1, 1) - Dx(test[1], 2, 1)
            w1 = Dx(test[0], 2, 1) - Dx(test[2], 0, 1)
            w2 = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)
            test._terms = np.concatenate((w0.terms(), w1.terms(), w2.terms()), axis=0)
            test._scales = np.concatenate((w0.scales(), w1.scales(), w2.scales()), axis=0)
            test._indices = np.concatenate((w0.indices(), w1.indices(), w2.indices()), axis=0)
            test._measures = np.concatenate((w0.measures(), w1.measures(), w2.measures()), axis=0)
        else:
            assert test.dimensions == 2
            test = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)

    else:
        assert test.expr_rank() < 2, 'Cannot take curl of higher order tensor in curvilinear coordinates'
        psi = test.function_space().measures[0]
        if test.dimensions == 3:
            w0 = (hi[2]*Dx(test[2], 1, 1) + test[2]*sp.diff(hi[2], psi[1], 1) - hi[1]*Dx(test[1], 2, 1) - test[1]*sp.diff(hi[1], psi[2], 1))*(1/(hi[1]*hi[2]))
            w1 = (hi[0]*Dx(test[0], 2, 1) + test[0]*sp.diff(hi[0], psi[2], 1) - hi[2]*Dx(test[2], 0, 1) - test[2]*sp.diff(hi[2], psi[0], 1))*(1/(hi[0]*hi[2]))
            w2 = (hi[1]*Dx(test[1], 0, 1) + test[1]*sp.diff(hi[1], psi[0], 1) - hi[0]*Dx(test[0], 1, 1) - test[0]*sp.diff(hi[0], psi[1], 1))*(1/(hi[0]*hi[1]))
            test._terms = np.concatenate((w0.terms(), w1.terms(), w2.terms()), axis=0)
            test._scales = np.concatenate((w0.scales(), w1.scales(), w2.scales()), axis=0)
            test._indices = np.concatenate((w0.indices(), w1.indices(), w2.indices()), axis=0)
            test._measures = np.concatenate((w0.measures(), w1.measures(), w2.measures()), axis=0)
        else:
            assert test.dimensions == 2
            test = (hi[1]*Dx(test[1], 0, 1) + test[1]*sp.diff(hi[1], psi[0], 1) - hi[0]*Dx(test[0], 1, 1) - test[0]*sp.diff(hi[0], psi[1], 1))*(1/(hi[0]*hi[1]))


    return test
