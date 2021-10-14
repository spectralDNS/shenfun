import pytest
import numpy as np
import sympy as sp
from mpi4py import MPI
import shenfun
from shenfun import inner, div, grad, curl

N = 8
comm = MPI.COMM_WORLD

V = shenfun.FunctionSpace(N, 'C')
u0 = shenfun.TrialFunction(V)

T = shenfun.TensorProductSpace(comm, (V, V))
u1 = shenfun.TrialFunction(V)

TT = shenfun.VectorSpace(T)
u2 = shenfun.TrialFunction(TT)

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_mul(basis):
    e = shenfun.Expr(basis)
    e2 = 2*e
    assert np.allclose(np.array(e2.scales()).astype(int), 2)
    e2 = e*2
    assert np.allclose(np.array(e2.scales()).astype(int), 2.)
    if e.expr_rank() == 1:
        a = tuple(range(e.dimensions))
        e2 = a*e
        assert np.allclose(np.array(e2.scales()).astype(int)[:, 0], (0, 1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_imul(basis):
    e = shenfun.Expr(basis)
    e *= 2
    assert np.allclose(np.array(e.scales()).astype(int), 2)
    e *= 2
    assert np.allclose(np.array(e.scales()).astype(int), 4)
    x = sp.symbols('x', real=True)
    e *= x
    assert np.alltrue(np.array(e.scales()) == 4*x)
    if e.expr_rank() == 1:
        a = tuple(range(e.dimensions))
        e *= a
        assert np.alltrue(np.array(e.scales())[:, 0] == (0, 4*x))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_add(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e3 = e + e2
    assert np.allclose(e3.terms(), np.concatenate((np.array(e.terms()), np.array(e2.terms())), axis=1))
    assert np.allclose(np.array(e3.scales()).astype(int),
                       np.concatenate((np.array(e.scales()).astype(int),
                                       np.array(e2.scales()).astype(int)), axis=1))
    assert np.allclose(e3.indices(), np.concatenate((np.array(e.indices()), np.array(e2.indices())), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_iadd(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e += e2
    assert np.allclose(e.terms(), np.concatenate((np.array(e2.terms()), np.array(e2.terms())), axis=1))
    assert np.allclose(np.array(e.scales()).astype(int),
                       np.concatenate((np.array(e2.scales()).astype(int),
                                       np.array(e2.scales()).astype(int)), axis=1))
    assert np.allclose(e.indices(), np.concatenate((np.array(e2.indices()), np.array(e2.indices())), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_sub(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e3 = e - e2
    assert np.allclose(e3.terms(), np.concatenate((np.array(e.terms()), np.array(e2.terms())), axis=1))
    assert np.allclose(np.array(e3.scales()).astype(int), np.concatenate((np.array(e.scales()).astype(int), -np.array(e2.scales()).astype(int)), axis=1))
    assert np.allclose(e3.indices(), np.concatenate((np.array(e.indices()), np.array(e2.indices())), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_isub(basis):
    e = shenfun.Expr(basis)
    e2 = shenfun.Expr(basis)
    e -= e2
    assert np.allclose(np.array(e.terms()), np.concatenate((np.array(e2.terms()), np.array(e2.terms())), axis=1))
    assert np.allclose(np.array(e.scales()).astype(int), np.concatenate((np.array(e2.scales()).astype(int), -np.array(e2.scales()).astype(int)), axis=1))
    assert np.allclose(np.array(e.indices()), np.concatenate((np.array(e2.indices()), np.array(e2.indices())), axis=1))

@pytest.mark.parametrize('basis', (u0, u1, u2))
def test_neg(basis):
    e0 = div(grad(basis))
    e1 = -e0
    sc0 = e0.scales()
    sc1 = e1.scales()
    s = sp.S(0)
    for sa, sb in zip(sc0, sc1):
        for sai, sbi in zip(sa, sb):
            s += sai+sbi
    assert sp.simplify(s) == 0

K0 = shenfun.FunctionSpace(N, 'F', dtype='D')
K1 = shenfun.FunctionSpace(N, 'F', dtype='D')
K2 = shenfun.FunctionSpace(N, 'F', dtype='d')
K3 = shenfun.FunctionSpace(N, 'C', dtype='d')
T = shenfun.TensorProductSpace(comm, (K0, K1, K2))
C = shenfun.TensorProductSpace(comm, (K1, K2, K3))
TT = shenfun.VectorSpace(T)
CC = shenfun.VectorSpace(C)
S  = shenfun.TensorSpace(T)
VT = shenfun.CompositeSpace([TT, T])
WT = shenfun.CompositeSpace([T, TT])
KK = shenfun.CompositeSpace([T, T, C])
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


def test_inner():
    v = shenfun.TestFunction(TT)
    u = shenfun.TrialFunction(TT)
    p = shenfun.TrialFunction(T)
    q = shenfun.TestFunction(T)
    inner(div(u), div(v))
    inner(grad(u), grad(v))
    inner(q, div(u))
    inner(curl(v), curl(u))
    inner(grad(q), grad(p))
    inner(v, grad(div(u)))
    inner(q, u[0])
    inner(q, u[1])
    inner(q, grad(p)[0])
    inner(q, grad(p)[1])
    inner(v, grad(u)[0])
    inner(v, grad(u)[1])
    wq = shenfun.TrialFunction(VT)
    w, q = wq
    hf = shenfun.TestFunction(VT)
    h, f = hf
    inner(h, div(grad(w)))
    inner(f, div(div(grad(w))))
    inner(h, curl(w))
    inner(curl(h), curl(w))
    inner(h, grad(div(w)))
    qw = shenfun.TrialFunction(WT)
    q, w = qw
    fh = shenfun.TestFunction(WT)
    f, h = fh
    inner(h, div(grad(w)))
    inner(f, div(div(grad(w))))
    inner(h, curl(w))
    inner(curl(h), curl(w))
    inner(h, grad(div(w)))

def test_tensor2():
    B0 = shenfun.FunctionSpace(8, 'C')
    T = shenfun.TensorProductSpace(comm, (B0, B0))
    x,y = sp.symbols('x,y')
    ue = x**2+y**2
    ua = shenfun.Array(T, buffer=ue)
    uh = ua.forward()
    M = shenfun.VectorSpace(T)
    gradu = shenfun.project(grad(uh), M)
    #V = shenfun.TensorSpace(T)
    #gradgradu = shenfun.project(grad(grad(uh)), V)
    #g = gradgradu.backward()
    #assert np.allclose(g.v[0], 2)
    #assert np.allclose(g.v[1], 0)
    #assert np.allclose(g.v[2], 0)
    #assert np.allclose(g.v[3], 2)

if __name__ == '__main__':
    # test_mul(u2)
    #test_imul(u2)
    # test_add(u2)
    # test_iadd(u2)
    # test_sub(u2)
    # test_isub(u2)
    test_neg(u2)
    # test_index(vf)
    #test_inner()
    #test_tensor2()
