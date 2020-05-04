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
    ndim = test.dimensions
    hi = test.function_space().hi
    coors = test.function_space().coors

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

        test._terms = v
        test._scales = sc
        test._indices = ind
        return test

    else:
        assert test.expr_rank() < 2, 'Cannot (yet) take divergence of higher order tensor in curvilinear coordinates'

        v = np.repeat(v, 2, axis=1)
        sc = np.repeat(sc, 2, axis=1)
        ind = np.repeat(ind, 2, axis=1)
        psi = test.function_space().coors.coordinates[0]
        gt = coors.get_contravariant_metric_tensor()
        sg = coors.get_sqrt_g()

        for i, s in enumerate(v):
            ll = [k for k in range(ndim) if not k==i]
            for j in range(v.shape[1]):
                if j%2 == 0:
                    s[j, i%ndim] += 1
                else:
                    ms2 = sp.simplify(sc[i, j]*sg)
                    sc[i, j] = sp.simplify(ms2.diff(psi[i%ndim], 1) / sg)

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

    test = copy.copy(test)

    terms = test.terms().copy()
    sc = test.scales().copy()
    ind = test.indices().copy()
    ndim = test.dimensions
    hi = test.function_space().hi
    coors = test.function_space().coors

    if coors.is_cartesian:
        test._terms = np.repeat(terms, ndim, axis=0)
        test._scales = np.repeat(sc, ndim, axis=0)
        test._indices = np.repeat(ind, ndim, axis=0)
        for i, s in enumerate(test._terms):
            s[..., i%ndim] += 1
        return test

    elif sc.flatten().prod() == 1:
        # If expr taken gradient of has scale 1
        if test.expr_rank() == 0: # if gradient of scalar
            if coors.is_orthogonal:
                test._terms = np.repeat(terms, ndim, axis=0)
                test._scales = np.repeat(sc, ndim, axis=0)
                test._indices = np.repeat(ind, ndim, axis=0)
                gt = coors.get_contravariant_metric_tensor()
                for i, s in enumerate(test._terms):
                    s[..., i%ndim] += 1

                for i, s in enumerate(test._scales):
                    s[:] *= gt[i%ndim, i%ndim]
            else:
                test._terms = terms.repeat(ndim, axis=0).repeat(ndim, axis=1)
                test._scales = sc.repeat(ndim, axis=0).repeat(ndim, axis=1)
                test._indices = ind.repeat(ndim, axis=0).repeat(ndim, axis=1)
                g = coors.get_contravariant_metric_tensor()
                elms = test._terms.shape[1] // ndim
                # There are elms scalar elements that we take the gradient of
                # Each element gives ndim*ndim new terms
                for i in range(ndim):
                    for k in range(elms):
                        for j in range(ndim):
                            test._terms[i, k*ndim+j, j] += 1
                            test._scales[i, k*ndim+j] *= g[i%ndim, j] # g[i, j] = g[j, i]
                            test._scales[i, k*ndim+j] = sp.simplify(test._scales[i, k*ndim+j])
        else:
            raise NotImplementedError('Cannot (yet) take gradient of tensor in curvilinear coordinates')

        return test

    else:
        # Expr taken gradient of has measures itself
        if test.expr_rank() == 0: # if gradient of scalar
            if coors.is_orthogonal:
                test._terms = terms.repeat(ndim, axis=0).repeat(2, axis=1)
                test._scales = sc.repeat(ndim, axis=0).repeat(2, axis=1)
                test._indices = ind.repeat(ndim, axis=0).repeat(2, axis=1)
                psi = test.function_space().coors.coordinates[0]
                elms = test._terms.shape[1] // 2
                for i in range(ndim):
                    for j in range(test._terms.shape[1]):
                        if j % 2 == 0:
                            test._terms[i, j, i] += 1
                            test._scales[i, j] *= g[i, i]
                        else:
                            test._scales[i, j] = test._scales[i, j].diff(psi[i], 1) * g[i, i]

            else:
                test._terms = terms.repeat(ndim, axis=0).repeat(ndim*2, axis=1)
                test._scales = sc.repeat(ndim, axis=0).repeat(ndim*2, axis=1)
                test._indices = ind.repeat(ndim, axis=0).repeat(ndim*2, axis=1)
                psi = test.function_space().coors.coordinates[0]
                g = coors.get_contravariant_metric_tensor()
                elms = test._terms.shape[1] // (2*ndim)
                for i in range(ndim):
                    for k in range(elms):
                        for j in range(ndim):
                            if k % 2 == 0:
                                test._terms[i, k*ndim+j, j] += 1
                                test._scales[i, k*ndim+j] *= g[i, j]
                            else:
                                test._scales[i, k*ndim+j] = test._scales[i, k*ndim+j].diff(psi[j], 1) * g[i, j]
        else:
            raise NotImplementedError('Cannot (yet) take gradient of tensor in curvilinear coordinates')

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

    if test.function_space().coors.is_cartesian:
        v = test.terms().copy()
        v[..., x] += k
        test._terms = v

    else:
        assert test.expr_rank() < 1, 'Cannot (yet) take derivative of tensor in curvilinear coordinates'
        v = test._terms = np.repeat(test.terms(), 2, axis=1)
        sc = test._scales = np.repeat(test.scales(), 2, axis=1)
        test._indices = np.repeat(test.indices(), 2, axis=1)
        psi = test.function_space().coors.coordinates[0]
        for i in range(v.shape[1]):
            if i % 2 == 0:
                v[:, i, x] += k
            else:
                sc[:, i] = sp.diff(sc[:, i], psi[x], 1)

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

    coors = test.function_space().coors
    hi = coors.hi

    # Note - need to make curvilinear in terms of covariant vector

    if coors.is_cartesian:
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

    else:
        assert test.expr_rank() < 2, 'Cannot (yet) take curl of higher order tensor in curvilinear coordinates'
        psi = coors.psi
        sg = coors.get_sqrt_g()
        if coors.is_orthogonal:
            if test.dimensions == 3:
                w0 = (hi[2]**2*Dx(test[2], 1, 1) + test[2]*sp.diff(hi[2]**2, psi[1], 1) - hi[1]**3*Dx(test[1], 2, 1) - test[1]*sp.diff(hi[1]**2, psi[2], 1))/sg
                w1 = (hi[0]**2*Dx(test[0], 2, 1) + test[0]*sp.diff(hi[0]**2, psi[2], 1) - hi[2]**3*Dx(test[2], 0, 1) - test[2]*sp.diff(hi[2]**2, psi[0], 1))/sg
                w2 = (hi[1]**2*Dx(test[1], 0, 1) + test[1]*sp.diff(hi[1]**2, psi[0], 1) - hi[0]**3*Dx(test[0], 1, 1) - test[0]*sp.diff(hi[0]**2, psi[1], 1))/sg
                test._terms = np.concatenate((w0.terms(), w1.terms(), w2.terms()), axis=0)
                test._scales = np.concatenate((w0.scales(), w1.scales(), w2.scales()), axis=0)
                test._indices = np.concatenate((w0.indices(), w1.indices(), w2.indices()), axis=0)
            else:
                assert test.dimensions == 2
                test = (hi[1]**2*Dx(test[1], 0, 1) + test[1]*sp.diff(hi[1]**2, psi[0], 1) - hi[0]**2*Dx(test[0], 1, 1) - test[0]*sp.diff(hi[0]**2, psi[1], 1))/sg
        else:
            raise NotImplementedError

    return test
