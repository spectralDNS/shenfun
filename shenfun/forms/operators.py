import copy
import numpy as np
import sympy as sp
from shenfun.config import config
from .arguments import Expr, BasisFunction

__all__ = ('div', 'grad', 'Dx', 'curl')

#pylint: disable=protected-access

def _expr_from_vector_components(comp, basis):
    """Return Expr composed of vector components `comp`
    """
    hi = basis.function_space().coors.hi
    terms, scales, indices = [], [], []
    ndim = len(comp[0].terms()[0][0])
    his = [1]*len(comp)
    if config['basisvectors'] == 'normal':
        if len(comp) == ndim:
            his = hi
        elif len(comp) == ndim**2:
            his = [hi[i]*hi[j] for i in range(ndim) for j in range(ndim)]
    for i, c in enumerate(comp):
        c *= his[i]
        terms += copy.deepcopy(c._terms)
        scales += copy.deepcopy(c._scales)
        indices += copy.deepcopy(c._indices)
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
            return Dx(test, 0, 1)

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
                d._basis = test._basis
                return d

    else:

        comp = test.get_contravariant_component
        if ndim == 1:      # 1D
            sg = coors.get_sqrt_det_g()
            d = Dx(comp(None)*sg, 0, 1)*(1/sg)
            return d

        else:

            if test.num_components() == ndim**2:
                ct = coors.get_christoffel_second()
                d = []
                for i in range(ndim):
                    di = []
                    for j in range(ndim):
                        Sij = comp(i, j)
                        di.append(Dx(Sij, j, 1))
                        for k in range(ndim):
                            Skj = comp(k, j)
                            #if not ct[i, j, k] == 0:
                            #    di.append(Skj*ct[i, j, k])
                            #if not ct[k, k, j] == 0:
                            #    di.append(Sij*ct[k, k, j])
                            Sik = comp(i, k)
                            if not ct[i, k, j] == 0:
                                di.append(Skj*ct[i, k, j])
                            if not ct[j, k, j] == 0:
                                di.append(Sik*ct[j, k, j])

                    dj = di[0]
                    for j in range(1, len(di)):
                        dj += di[j]
                    dj.simplify()
                    d.append(dj)
                return _expr_from_vector_components(d, test.basis())

            else:
                sg = coors.get_sqrt_det_g()
                d = Dx(comp(0)*sg, 0, 1)*(1/sg)
                for i in range(1, ndim):
                    d += Dx(comp(i)*sg, i, 1)*(1/sg)
            d.simplify()
            d._basis = test._basis
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
        comp = test.get_contravariant_component

        if test.num_components() > 1:
            ct = coors.get_christoffel_second()
            d = []
            for i in range(ndim):
                vi = comp(i)
                for j in range(ndim):
                    dj = []
                    for l in range(ndim):
                        sc = gt[l, j]
                        if not sc == 0:
                            dj.append(Dx(vi, l, 1)*sc)
                        for k in range(ndim):
                            if not sc*ct[i, k, l] == 0:
                                dj.append(comp(k)*(sc*ct[i, k, l]))

                    di = dj[0]
                    for m in range(1, len(dj)):
                        di += dj[m]
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

    if k == 0:
        return test

    if k > 1:
        for _ in range(k):
            test = Dx(test, x, 1)
        return test

    if isinstance(test, BasisFunction):
        test = Expr(test)

    dtest = Expr(test._basis,
                 copy.deepcopy(test._terms),
                 copy.deepcopy(test._scales),
                 copy.deepcopy(test._indices))

    # product rule
    # \frac{\partial scale(x) test(x)}{\partial x} = scale(x) \frac{\partial test(x)}{\partial x} + test(x) \frac{\partial scale(x)}{\partial x}

    coors = dtest.function_space().coors
    assert test.expr_rank() < 1 or coors.is_cartesian, 'Cannot (yet) take derivative of tensor in curvilinear coordinates'
    psi = coors.psi
    v = copy.deepcopy(dtest.terms())
    sc = copy.deepcopy(dtest.scales())
    ind = copy.deepcopy(dtest.indices())
    num_terms = dtest.num_terms()
    for i in range(dtest.num_components()):
        for j in range(num_terms[i]):
            sc0 = sp.simplify(sp.diff(sc[i][j], psi[x], 1), measure=coors._measure)
            sc0 = coors.refine(sc0)
            if not sc0 == 0:
                v[i].append(copy.deepcopy(v[i][j]))
                sc[i].append(sc0)
                ind[i].append(ind[i][j])
            v[i][j][x] += 1
    dtest._terms = v
    dtest._scales = sc
    dtest._indices = ind
    return dtest

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

    #assert test.expr_rank() > 0
    #assert test.num_components() == test.dimensions

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
            if test.expr_rank() > 0:
                test = Dx(test[1], 0, 1) - Dx(test[0], 1, 1)
            else:
                # Assume scalar test is vector test*k, where k is the unit basis vector in the z-direction
                test = _expr_from_vector_components([Dx(test, 1, 1), -Dx(test, 0, 1)], test.basis())

    else:
        assert test.expr_rank() < 2, 'Cannot (yet) take curl of higher order tensor in curvilinear coordinates'
        hi = coors.hi
        sg = coors.get_sqrt_det_g()
        comp = test.get_contravariant_component

        if coors.is_orthogonal:
            #p = 1 if config['basisvectors'] == 'normal' else 2
            if test.dimensions == 3:
                w0 = (Dx(comp(2)*hi[2]**2, 1, 1) - Dx(comp(1)*hi[1]**2, 2, 1))*(1/sg)
                w1 = (Dx(comp(0)*hi[0]**2, 2, 1) - Dx(comp(2)*hi[2]**2, 0, 1))*(1/sg)
                w2 = (Dx(comp(1)*hi[1]**2, 0, 1) - Dx(comp(0)*hi[0]**2, 1, 1))*(1/sg)
                test = _expr_from_vector_components([w0, w1, w2], test.basis())
            else:
                assert test.dimensions == 2
                test = (Dx(comp(1)*hi[1]**2, 0, 1) - Dx(comp(0)*hi[0]**2, 1, 1))*(1/sg)

        else:
            g = coors.get_covariant_metric_tensor()

            if test.dimensions == 3:
                w0 = np.sum([(Dx(comp(i)*g[2, i], 1, 1) - Dx(comp(i)*g[1, i], 2, 1))*(1/sg) for i in range(3)])
                w1 = np.sum([(Dx(comp(i)*g[0, i], 2, 1) - Dx(comp(i)*g[2, i], 0, 1))*(1/sg) for i in range(3)])
                w2 = np.sum([(Dx(comp(i)*g[1, i], 0, 1) - Dx(comp(i)*g[0, i], 1, 1))*(1/sg) for i in range(3)])

                test = _expr_from_vector_components([w0, w1, w2], test.basis())
            else:
                assert test.dimensions == 2
                test = np.sum([(Dx(comp(i)*g[1, i], 0, 1) - Dx(comp(i)*g[0, i], 1, 1))*(1/sg) for i in range(2)])

    test.simplify()
    return test
