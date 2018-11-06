import pytest
import numpy as np
from mpi4py import MPI
import shenfun

N = 8
comm = MPI.COMM_WORLD

V = shenfun.Basis(N, 'C')
u0 = shenfun.TrialFunction(V)

T = shenfun.TensorProductSpace(comm, (V, V))
u1 = shenfun.TrialFunction(V)

TT = shenfun.VectorTensorProductSpace(T)
u2 = shenfun.TrialFunction(TT)

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_mul(basis):
    e = shenfun.Expr(basis)
    e2 = 2*e
    assert np.allclose(e2.scales(), 2.)
    e2 = e*2
    assert np.allclose(e2.scales(), 2.)
    if e.expr_rank() == 1:
        a = tuple(range(e.dimensions()))
        e2 = a*e
        assert np.allclose(e2.scales()[:, 0], (0, 1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_imul(basis):
    e = shenfun.Expr(basis)
    e *= 2
    assert np.allclose(e.scales(), 2.)
    e *= 2
    assert np.allclose(e.scales(), 4.)
    if e.expr_rank() == 1:
        a = tuple(range(e.dimensions()))
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

K0 = shenfun.Basis(N, 'F', dtype='D')
K1 = shenfun.Basis(N, 'F', dtype='D')
K2 = shenfun.Basis(N, 'F', dtype='d')
K3 = shenfun.Basis(N, 'C', dtype='d')
T = shenfun.TensorProductSpace(comm, (K0, K1, K2))
C = shenfun.TensorProductSpace(comm, (K1, K2, K3))
TT = shenfun.VectorTensorProductSpace(T)
CC = shenfun.VectorTensorProductSpace(C)
KK = shenfun.MixedTensorProductSpace([T, T, C])
vf = shenfun.Function(TT)
va = shenfun.Array(TT)
cf = shenfun.Function(CC)
ca = shenfun.Array(CC)
df = shenfun.Function(KK)
da = shenfun.Array(KK)

@pytest.mark.parametrize('u', (va, vf, cf, ca, df, da))
def test_index(u):
    va0 = u[0]
    va1 = u[1]
    va2 = u[2]
    assert (va0.index(), va1.index(), va2.index()) == (0, 1, 2)
    assert va0.function_space() is u.function_space()[0]
    assert va1.function_space() is u.function_space()[1]
    assert va2.function_space() is u.function_space()[2]

if __name__ == '__main__':
    # test_mul(u2)
    # test_imul(u2)
    # test_add(u2)
    # test_iadd(u2)
    # test_sub(u2)
    # test_isub(u2)
    # test_neg(u2)
    test_index(vf)
