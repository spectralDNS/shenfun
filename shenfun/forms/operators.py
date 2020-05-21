"""
This module contains the implementation of operators acting on arguments.
"""
import numpy as np
import sympy as sp
import copy
from .arguments import Expr, BasisFunction

__all__ = ('div', 'grad', 'Dx', 'curl')

#pylint: disable=protected-access

def _expr_from_vector_components(comp, basis):
    """Return Expr composed of vector components `comp`
    """
    terms, scales, indices = [], [], []
    for i in range(len(comp)):
        terms += comp[i]._terms
        scales += comp[i]._scales
        indices += comp[i]._indices
    return Expr(basis, terms, scales, indices)

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

    #test = copy.copy(test)
    ndim = test.dimensions
    coors = test.function_space().coors

    if coors.is_cartesian:

        if ndim == 1:      # 1D
            v = np.array(test.terms())
            v += 1
            test._terms = v.tolist()
            return test

        else:
            if test.num_components() == ndim**2: # second rank tensor
                dv = []
                for i in range(ndim):
                    dv.append(div(test[i]))

                return _expr_from_vector_components(dv, test.base)

            else: # vector
                d = Dx(test[0], 0, 1)
                for i in range(1, ndim):
                    d += Dx(test[i], i, 1)
                d.simplify()
                return d

    else:

        if test.num_components() == ndim**2:

            ct = coors.get_christoffel_second()
            d = []
            for i in range(ndim):
                di = []
                for j in range(ndim):
                    Sij = test[i][j]
                    di.append(Dx(Sij, j, 1))
                    for k in range(ndim):
                        Sik = test[i][k]
                        Sjk = test[j][k]
                        if not ct[i, k, j] == 0:
                            di.append(Sjk*ct[i, k, j])
                        if not ct[k, k, j] == 0:
                            di.append(Sij*ct[k, k, j])

                dj = di[0]
                for j in range(1, len(di)):
                    dj += di[j]
                dj.simplify()
                d.append(dj)
            return _expr_from_vector_components(d, test.base)

        else:
            sg = coors.get_sqrt_g()
            d = Dx(test[0]*sg, 0, 1)*(1/sg)
            for i in range(1, ndim):
                d += Dx(test[i]*sg, i, 1)*(1/sg)
        d.simplify()
        return d

def grad(test):
    """Return grad(test)

    Parameters
    ----------
    test: Expr or BasisFunction

    Note
    ----
    Increases the rank of Expr by one

    """
    assert isinstance(test, (Expr, BasisFunction))

    if isinstance(test, BasisFunction):
        test = Expr(test)

    #test = copy.copy(test)
    ndim = test.dimensions
    coors = test.function_space().coors

    if coors.is_cartesian:

        d = []
        if test.num_components() > 1:
            for i in range(test.num_components()):
                for j in range(ndim):
                    d.append(Dx(test[i], j, 1))
        else:
           for i in range(ndim):
               d.append(Dx(test, i, 1))

    else:
        #assert test.expr_rank() < 2, 'Cannot (yet) take gradient of higher order tensor in curvilinear coordinates'

        gt = coors.get_contravariant_metric_tensor()

        if test.num_components() == ndim:
            ct = coors.get_christoffel_second()
            d = []
            for i in range(ndim):
                vi = test[i]
                for j in range(ndim):
                    dj = []
                    for l in range(ndim):
                        sc = gt[l, j]
                        if not sc == 0:
                            dj.append(Dx(vi, l, 1)*sc)
                        for k in range(ndim):
                            if not sc*ct[i, k, l] == 0:
                                dj.append(test[k]*(sc*ct[i, j, k]))

                    di = dj[0]
                    for j in range(1, len(dj)):
                        di += dj[j]
                    d.append(di)

        else:
            d = []
            for i in range(ndim):
                dj = []
                for j in range(ndim):
                    sc = gt[j, i]
                    if not sc == 0:
                        dj.append(Dx(test, j, 1)*sc)
                di = dj[0]
                for j in range(1, len(dj)):
                    di += dj[j]
                d.append(di)

    dv = _expr_from_vector_components(d, test.base)
    dv.simplify()
    return dv

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
    coors = test.function_space().coors

    if coors.is_cartesian:
        v = np.array(test.terms())
        v[..., x] += k
        test._terms = v.tolist()

    else:
        assert test.expr_rank() < 1, 'Cannot (yet) take derivative of tensor in curvilinear coordinates'
        psi = coors.psi
        v = copy.deepcopy(test.terms())
        sc = copy.deepcopy(test.scales())
        ind = copy.deepcopy(test.indices())
        num_terms = test.num_terms()
        for i in range(test.num_components()):
            for j in range(num_terms[i]):
                sc0 = sp.simplify(sp.diff(sc[i][j], psi[x], k))
                if not sc0 == 0:
                    v[i].append(copy.deepcopy(v[i][j]))
                    sc[i].append(sc0)
                    ind[i].append(ind[i][j])
                v[i][j][x] += k
        test._terms = v
        test._scales = sc
        test._indices = ind

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
    assert test.num_components() == test.dimensions

    coors = test.function_space().coors

    if coors.is_cartesian:
        if test.dimensions == 3:
            w0 = Dx(test[2], 1, 1) - Dx(test[1], 2, 1)
            w1 = Dx(test[0], 2, 1) - Dx(test[2], 0, 1)
            w2 = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)
            test._terms = w0.terms()+w1.terms()+w2.terms()
            test._scales = w0.scales()+w1.scales()+w2.scales()
            test._indices = w0.indices()+w1.indices()+w2.indices()
        else:
            assert test.dimensions == 2
            test = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)

    else:
        assert test.expr_rank() < 2, 'Cannot (yet) take curl of higher order tensor in curvilinear coordinates'
        hi = coors.hi
        sg = coors.get_sqrt_g()
        if coors.is_orthogonal:
            if test.dimensions == 3:
                w0 = (Dx(test[2]*hi[2]**2, 1, 1) - Dx(test[1]*hi[1]**2, 2, 1))*(1/sg)
                w1 = (Dx(test[0]*hi[0]**2, 2, 1) - Dx(test[2]*hi[2]**2, 0, 1))*(1/sg)
                w2 = (Dx(test[1]*hi[1]**2, 0, 1) - Dx(test[0]*hi[0]**2, 1, 1))*(1/sg)
                test._terms = w0.terms()+w1.terms()+w2.terms()
                test._scales = w0.scales()+w1.scales()+w2.scales()
                test._indices = w0.indices()+w1.indices()+w2.indices()

            else:
                assert test.dimensions == 2
                test = (Dx(test[1]*hi[1]**2, 0, 1) - Dx(test[0]*hi[0]**2, 1, 1))*(1/sg)
        else:
            raise NotImplementedError

    return test
