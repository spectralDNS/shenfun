import pytest
import sympy as sp
import numpy as np
from shenfun import FunctionSpace, Function, project, TensorProductSpace, \
    comm

x, y = sp.symbols('x,y', real=True)
f = sp.sin(sp.cos(x))
h = sp.sin(sp.cos(x))*sp.atan2(y, x)
N = 16

def test_backward():
    T = FunctionSpace(N, 'C')
    L = FunctionSpace(N, 'L')

    uT = Function(T, buffer=f)
    uL = Function(L, buffer=f)

    uLT = uL.backward(kind=T)
    uT2 = project(uLT, T)
    assert np.linalg.norm(uT2-uT) < 1e-8

    uTL = uT.backward(kind=L)
    uL2 = project(uTL, L)
    assert np.linalg.norm(uL2-uL) < 1e-8

    T2 = FunctionSpace(N, 'C', bc=(f.subs(x, -1), f.subs(x, 1)))
    L = FunctionSpace(N, 'L')

    uT = Function(T2, buffer=f)
    uL = Function(L, buffer=f)

    uLT = uL.backward(kind=T2)
    uT2 = project(uLT, T2)
    assert np.linalg.norm(uT2-uT) < 1e-8

def test_backward2D():
    T = FunctionSpace(N, 'C')
    L = FunctionSpace(N, 'L')
    F = FunctionSpace(N, 'F', dtype='d')
    TT = TensorProductSpace(comm, (T, F))
    TL = TensorProductSpace(comm, (L, F))
    uT = Function(TT, buffer=f)
    uL = Function(TL, buffer=f)

    u2 = uL.backward(kind=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)

    TT = TensorProductSpace(comm, (F, T))
    TL = TensorProductSpace(comm, (F, L))
    uT = Function(TT, buffer=f)
    uL = Function(TL, buffer=f)

    u2 = uL.backward(kind=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)

def test_backward2ND():
    T0 = FunctionSpace(N, 'C')
    L0 = FunctionSpace(N, 'L')
    T1 = FunctionSpace(N, 'C')
    L1 = FunctionSpace(N, 'L')
    TT = TensorProductSpace(comm, (T0, T1))
    LL = TensorProductSpace(comm, (L0, L1))
    uT = Function(TT, buffer=h)
    uL = Function(LL, buffer=h)
    u2 = uL.backward(kind=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)

def test_backward3D():
    T = FunctionSpace(N, 'C')
    L = FunctionSpace(N, 'L')
    F0 = FunctionSpace(N, 'F', dtype='D')
    F1 = FunctionSpace(N, 'F', dtype='d')
    TT = TensorProductSpace(comm, (F0, T, F1))
    TL = TensorProductSpace(comm, (F0, L, F1))
    uT = Function(TT, buffer=h)
    uL = Function(TL, buffer=h)

    u2 = uL.backward(kind=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)

@pytest.mark.parametrize('family', 'CLJ')
def test_backward_uniform(family):
    T = FunctionSpace(N, family)
    uT = Function(T, buffer=f)
    ub = uT.backward(kind='uniform')
    xj = T.mesh(uniform=True)
    fj = sp.lambdify(x, f)(xj)
    assert np.linalg.norm(fj-ub) < 1e-8

if __name__ == '__main__':
    test_backward()

