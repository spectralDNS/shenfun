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

                return _expr_from_vector_components(dv, test.basis())

            else: # vector
                d = Dx(test[0], 0, 1)
                for i in range(1, ndim):
                    d += Dx(test[i], i, 1)
                d.simplify()
                return d

    else:

        if ndim == 1:      # 1D
            sg = coors.get_sqrt_det_g()
            d = Dx(test*sg, 0, 1)*(1/sg)
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
                            Skj = test[k][j]
                            if not ct[i, j, k] == 0:
                                di.append(Skj*ct[i, j, k])
                            if not ct[k, k, j] == 0:
                                di.append(Sij*ct[k, k, j])

                    dj = di[0]
                    for j in range(1, len(di)):
                        dj += di[j]
                    dj.simplify()
                    d.append(dj)
                return _expr_from_vector_components(d, test.basis())

            else:
                sg = coors.get_sqrt_det_g()
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

        gt = coors.get_contravariant_metric_tensor()

        if test.num_components() > 1:
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
                                dj.append(test[k]*(sc*ct[i, k, l]))

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

    dv = _expr_from_vector_components(d, test.basis())
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

    if k > 1:
        for l in range(k):
            test = Dx(test, x, 1)
        return test

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
                sc0 = sp.refine(sc0, coors._assumptions)

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
        sg = coors.get_sqrt_det_g()
        if coors.is_orthogonal:
            if test.dimensions == 3:
                w0 = (Dx(test[2]*hi[2]**2, 1, 1) - Dx(test[1]*hi[1]**2, 2, 1))*(1/sg)
                w1 = (Dx(test[0]*hi[0]**2, 2, 1) - Dx(test[2]*hi[2]**2, 0, 1))*(1/sg)
                w2 = (Dx(test[1]*hi[1]**2, 0, 1) - Dx(test[0]*hi[0]**2, 1, 1))*(1/sg)
                test = _expr_from_vector_components([w0, w1, w2], test.basis())
            else:
                assert test.dimensions == 2
                test = (Dx(test[1]*hi[1]**2, 0, 1) - Dx(test[0]*hi[0]**2, 1, 1))*(1/sg)
        else:
            g = coors.get_covariant_metric_tensor()

            if test.dimensions == 3:
                ct = coors.get_christoffel_second()
                w0 = np.sum([(Dx(test[i]*g[2, i], 1, 1) - Dx(test[i]*g[1, i], 2, 1))*(1/sg) for i in range(3)])
                w1 = np.sum([(Dx(test[i]*g[0, i], 2, 1) - Dx(test[i]*g[2, i], 0, 1))*(1/sg) for i in range(3)])
                w2 = np.sum([(Dx(test[i]*g[1, i], 0, 1) - Dx(test[i]*g[0, i], 1, 1))*(1/sg) for i in range(3)])
                # Don't think this double loop is needed due to symmetry of Christoffel?
                #for i in range(3):
                #    for k in range(3):
                #        w0 += (ct[i, 1, 2]*g[i, k]*test[k] - ct[i, 2, 1]*g[i, k]*test[k])*(1/sg)
                #        w1 += (ct[i, 2, 0]*g[i, k]*test[k] - ct[i, 0, 2]*g[i, k]*test[k])*(1/sg)
                #        w2 += (ct[i, 0, 1]*g[i, k]*test[k] - ct[i, 1, 0]*g[i, k]*test[k])*(1/sg)

                # This is an alternative (more complicated way):
                #gt = coors.get_contravariant_metric_tensor()
                #ww0 = grad(g[0, 0]*test[0] + g[0, 1]*test[1] + g[0, 2]*test[2])
                #ww1 = grad(g[1, 0]*test[0] + g[1, 1]*test[1] + g[1, 2]*test[2])
                #ww2 = grad(g[2, 0]*test[0] + g[2, 1]*test[1] + g[2, 2]*test[2])
                #d0 = sg*(ww0[1]*gt[0, 2] + ww1[1]*gt[1, 2] + ww2[1]*gt[2, 2] - ww0[2]*gt[0, 1] - ww1[2]*gt[1, 1] - ww2[2]*gt[2, 1])
                #d1 = sg*(ww0[2]*gt[0, 0] + ww1[2]*gt[1, 0] + ww2[2]*gt[2, 0] - ww0[0]*gt[0, 2] - ww1[0]*gt[1, 2] - ww2[0]*gt[2, 2])
                #d2 = sg*(ww0[0]*gt[0, 1] + ww1[0]*gt[1, 1] + ww2[0]*gt[2, 1] - ww0[1]*gt[0, 0] - ww1[1]*gt[1, 0] - ww2[1]*gt[2, 0])
                #w0 = d0*gt[0, 0] + d1*gt[1, 0] + d2*gt[2, 0]
                #w1 = d0*gt[0, 1] + d1*gt[1, 1] + d2*gt[2, 1]
                #w2 = d0*gt[0, 2] + d1*gt[1, 2] + d2*gt[2, 2]

                test = _expr_from_vector_components([w0, w1, w2], test.basis())
            else:
                assert test.dimensions == 2
                test = np.sum([(Dx(test[i]*g[1, i], 0, 1) - Dx(test[i]*g[0, i], 1, 1))*(1/sg) for i in range(2)])

    test.simplify()
    return test
