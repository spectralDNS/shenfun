import pytest
import sympy as sp
import numpy as np
from shenfun import FunctionSpace, Function, project, TensorProductSpace, \
    comm, Array

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
    T = FunctionSpace(N, 'C', domain=(-2, 2))
    L = FunctionSpace(N, 'L', domain=(-2, 2))
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
    T0 = FunctionSpace(N, 'C', domain=(-2, 2))
    L0 = FunctionSpace(N, 'L', domain=(-2, 2))
    T1 = FunctionSpace(N, 'C', domain=(-2, 2))
    L1 = FunctionSpace(N, 'L', domain=(-2, 2))
    TT = TensorProductSpace(comm, (T0, T1))
    LL = TensorProductSpace(comm, (L0, L1))
    uT = Function(TT, buffer=h)
    uL = Function(LL, buffer=h)
    u2 = uL.backward(kind=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)

def test_backward3D():
    T = FunctionSpace(N, 'C', domain=(-2, 2))
    L = FunctionSpace(N, 'L', domain=(-2, 2))
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
    T = FunctionSpace(2*N, family, domain=(-2, 2))
    uT = Function(T, buffer=f)
    ub = uT.backward(kind='uniform')
    xj = T.mesh(uniform=True)
    fj = sp.lambdify(x, f)(xj)
    assert np.linalg.norm(fj-ub) < 1e-8

@pytest.mark.parametrize('family', 'CL')
def test_padding(family):
    N = 8
    B = FunctionSpace(N, family, bc=(-1, 1), domain=(-2, 2))
    Bp = B.get_dealiased(1.5)
    u = Function(B).set_boundary_dofs()
    #u[:(N-2)] = np.random.random(N-2)
    u[:(N-2)] = 1
    up = Array(Bp)
    up = Bp.backward(u, fast_transform=False)
    uf = Bp.forward(up, fast_transform=False)
    assert np.linalg.norm(uf-u) < 1e-12
    if family == 'C':
        up = Bp.backward(u)
        uf = Bp.forward(up)
        assert np.linalg.norm(uf-u) < 1e-12

    # Test padding 2D
    F = FunctionSpace(N, 'F', dtype='d')
    T = TensorProductSpace(comm, (B, F))
    Tp = T.get_dealiased(1.5)
    u = Function(T).set_boundary_dofs()
    u[:-2, :-1] = np.random.random(u[:-2, :-1].shape)
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

    # Test padding 3D
    F1 = FunctionSpace(N, 'F', dtype='D')
    T = TensorProductSpace(comm, (F1, F, B))
    Tp = T.get_dealiased(1.5)
    u = Function(T).set_boundary_dofs()
    u[:, :, :-2] = np.random.random(u[:, :, :-2].shape)
    u = u.backward().forward() # Clean
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

@pytest.mark.parametrize('family', 'CL')
def test_padding_neumann(family):
    N = 8
    B = FunctionSpace(N, family, bc={'left':('N', 0), 'right': ('N', 0)})
    Bp = B.get_dealiased(1.5)
    u = Function(B)
    u[1:-2] = np.random.random(N-3)
    up = Array(Bp)
    up = Bp.backward(u, fast_transform=False)
    uf = Bp.forward(up, fast_transform=False)
    assert np.linalg.norm(uf-u) < 1e-12
    if family == 'C':
        up = Bp.backward(u)
        uf = Bp.forward(up)
        assert np.linalg.norm(uf-u) < 1e-12

    # Test padding 2D
    F = FunctionSpace(N, 'F', dtype='d')
    T = TensorProductSpace(comm, (B, F))
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    u[1:-2, :-1] = np.random.random(u[1:-2, :-1].shape)
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

    # Test padding 3D
    F1 = FunctionSpace(N, 'F', dtype='D')
    T = TensorProductSpace(comm, (F1, F, B))
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    u[:, :, 1:-2] = np.random.random(u[:, :, 1:-2].shape)
    u = u.backward().forward() # Clean
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

@pytest.mark.parametrize('family', ('C','L','J','F','H','La'))
def test_padding_orthogonal(family):
    N = 8
    B = FunctionSpace(N, family)
    Bp = B.get_dealiased(1.5)
    u = Function(B)
    u[:] = np.random.random(u.shape)
    up = Array(Bp)
    if family != 'F':
        up = Bp.backward(u, fast_transform=False)
        uf = Bp.forward(up, fast_transform=False)
        assert np.linalg.norm(uf-u) < 1e-12
    if family in ('C', 'F'):
        up = Bp.backward(u, fast_transform=True)
        uf = Bp.forward(up, fast_transform=True)
        assert np.linalg.norm(uf-u) < 1e-12

    # Test padding 2D
    dtype = 'D' if family == 'F' else 'd'
    F = FunctionSpace(N, 'F', dtype=dtype)
    T = TensorProductSpace(comm, (F, B))
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    u[:] = np.random.random(u.shape)
    u = u.backward().forward()
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

    # Test padding 3D
    F1 = FunctionSpace(N, 'F', dtype='D')
    T = TensorProductSpace(comm, (F1, F, B))
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    u[:] = np.random.random(u.shape)
    u = u.backward().forward() # Clean
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

@pytest.mark.parametrize('family', 'CLJ')
def test_padding_biharmonic(family):
    N = 8
    B = FunctionSpace(N, family, bc=(0, 0, 0, 0))
    Bp = B.get_dealiased(1.5)
    u = Function(B)
    u[:(N-4)] = np.random.random(N-4)
    up = Array(Bp)
    up = Bp.backward(u, fast_transform=False)
    uf = Bp.forward(up, fast_transform=False)
    assert np.linalg.norm(uf-u) < 1e-12
    if family == 'C':
        up = Bp.backward(u)
        uf = Bp.forward(up)
        assert np.linalg.norm(uf-u) < 1e-12

    # Test padding 2D
    F = FunctionSpace(N, 'F', dtype='d')
    T = TensorProductSpace(comm, (B, F))
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    u[:-4, :-1] = np.random.random(u[:-4, :-1].shape)
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

    # Test padding 3D
    F1 = FunctionSpace(N, 'F', dtype='D')
    T = TensorProductSpace(comm, (F1, F, B))
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    u[:, :, :-4] = np.random.random(u[:, :, :-4].shape)
    u = u.backward().forward() # Clean
    up = Tp.backward(u)
    uc = Tp.forward(up)
    assert np.linalg.norm(u-uc) < 1e-8

if __name__ == '__main__':
    #test_backward()
    #test_backward2D()
    test_padding('L')
    #test_padding_biharmonic('C')
    #test_padding_neumann('C')
    #test_padding_orthogonal('F')
    #test_padding_orthogonal('C')

