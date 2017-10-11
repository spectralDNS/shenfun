import pytest
import numpy as np
from shenfun.chebyshev import bases as cbases
from shenfun.legendre import bases as lbases
from shenfun.fourier import bases as fbases
from mpi4py import MPI
import shenfun
from itertools import product

N = 8
comm = MPI.COMM_WORLD

V = cbases.Basis(N, plan=True)
u0 = shenfun.TrialFunction(V)

T = shenfun.TensorProductSpace(comm, (V, V))
u1 = shenfun.TrialFunction(V)

TT = shenfun.VectorTensorProductSpace([T, T])
u2 = shenfun.TrialFunction(TT)

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_mul(basis):
    e = shenfun.Expr(basis)
    e2 = 2*e
    assert np.allclose(e2.scales(), 2.)
    e2 = e*2
    assert np.allclose(e2.scales(), 2.)
    if e.expr_rank() == 2:
        a = tuple(range(e.dim()))
        e2 = a*e
        assert np.allclose(e2.scales()[:, 0], (0, 1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_imul(basis):
    e = shenfun.Expr(basis)
    e *= 2
    assert np.allclose(e.scales(), 2.)
    e *= 2
    assert np.allclose(e.scales(), 4.)
    if e.expr_rank() == 2:
        a = tuple(range(e.dim()))
        e *= a
        assert np.allclose(e.scales()[:, 0], (0, 4))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_add(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e3 = e + e2
    assert np.allclose(e3.terms(), np.concatenate((e.terms(), e2.terms()), axis=1))
    assert np.allclose(e3.scales(), np.concatenate((e.scales(), e2.scales()), axis=1))
    assert np.allclose(e3.indices(), np.concatenate((e.indices(), e2.indices()), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_iadd(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e += e2
    assert np.allclose(e.terms(), np.concatenate((e2.terms(), e2.terms()), axis=1))
    assert np.allclose(e.scales(), np.concatenate((e2.scales(), e2.scales()), axis=1))
    assert np.allclose(e.indices(), np.concatenate((e2.indices(), e2.indices()), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_sub(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e3 = e - e2
    assert np.allclose(e3.terms(), np.concatenate((e.terms(), e2.terms()), axis=1))
    assert np.allclose(e3.scales(), np.concatenate((e.scales(), -e2.scales()), axis=1))
    assert np.allclose(e3.indices(), np.concatenate((e.indices(), e2.indices()), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_isub(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e -= e2
    assert np.allclose(e.terms(), np.concatenate((e2.terms(), e2.terms()), axis=1))
    assert np.allclose(e.scales(), np.concatenate((e2.scales(), -e2.scales()), axis=1))
    assert np.allclose(e.indices(), np.concatenate((e2.indices(), e2.indices()), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_neg(basis):
    e = shenfun.Expr(basis)
    e2 = -e
    assert np.allclose(e.scales(), -e2.scales())

if __name__ == '__main__':
    test_mul(u2)
    test_imul(u2)
    test_add(u2)
    test_iadd(u2)
    test_sub(u2)
    test_isub(u2)
    test_neg(u2)
