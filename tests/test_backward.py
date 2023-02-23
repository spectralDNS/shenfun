import pytest
import sympy as sp
import numpy as np
from shenfun import FunctionSpace, Function, project, TensorProductSpace, \
    comm, Array, legendre
from shenfun.tensorproductspace import CompositeSpace

x, y, z = sp.symbols('x,y,z', real=True)
f = sp.sin(sp.cos(x))
ff = sp.sin(x)*sp.cos(y)
f3 = sp.sin(x)*sp.cos(y)*(1-z**2)
h = sp.sin(sp.cos(x))*sp.atan2(y, x)
N = 16

def test_backward():
    T = FunctionSpace(N, 'C')
    L = FunctionSpace(N, 'L')

    #uT = Function(T, buffer=f)
    #uL = Function(L, buffer=f)

    #uLT = uL.backward(mesh=T)
    #uT2 = project(uLT, T)
    #assert np.linalg.norm(uT2-uT) < 1e-8

    #uTL = uT.backward(mesh=L)
    #uL2 = project(uTL, L)
    #assert np.linalg.norm(uL2-uL) < 1e-8

    #T2 = FunctionSpace(N, 'C', bc=(f.subs(x, -1), f.subs(x, 1)))
    #L = FunctionSpace(N, 'L')

    #uT = Function(T2, buffer=f)
    #uL = Function(L, buffer=f)

    #uLT = uL.backward(mesh=T2)
    #uT2 = project(uLT, T2)
    #assert np.linalg.norm(uT2-uT) < 1e-8

def test_backward2D():
    T = FunctionSpace(N, 'C', domain=(-2, 2))
    L = FunctionSpace(N, 'L', domain=(-2, 2))
    F = FunctionSpace(N, 'F', dtype='d')
    TT = TensorProductSpace(comm, (T, F))
    TL = TensorProductSpace(comm, (L, F))
    uT = Function(TT, buffer=ff)
    uL = Function(TL, buffer=ff)
    u2 = uL.backward(mesh=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT) < 1e-8, np.linalg.norm(uT2-uT)
    TT.destroy()
    TL.destroy()
    TT = TensorProductSpace(comm, (F, T))
    TL = TensorProductSpace(comm, (F, L))
    uT = Function(TT, buffer=ff)
    uL = Function(TL, buffer=ff)
    u2 = uL.backward(mesh=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT) < 1e-8, np.linalg.norm(uT2-uT)
    TTC = CompositeSpace((TT, TT))
    TTL = CompositeSpace((TL, TL))
    uT = Function(TTC, buffer=(ff, ff))
    uL = Function(TTL, buffer=(ff, ff))
    u2 = uL.backward(mesh=TTC)
    uT2 = project(u2, TTC)
    assert np.linalg.norm(uT2-uT) < 1e-8
    TT.destroy()
    TL.destroy()

def test_backward2ND():
    T0 = FunctionSpace(N, 'C', domain=(-2, 2))
    L0 = FunctionSpace(N, 'L', domain=(-2, 2))
    T1 = FunctionSpace(N, 'C', domain=(-2, 2))
    L1 = FunctionSpace(N, 'L', domain=(-2, 2))
    TT = TensorProductSpace(comm, (T0, T1))
    LL = TensorProductSpace(comm, (L0, L1))
    uT = Function(TT, buffer=h)
    uL = Function(LL, buffer=h)
    u2 = uL.backward(mesh=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)
    TT.destroy()
    LL.destroy()

def test_backward3D():
    T = FunctionSpace(N, 'C', domain=(-2, 2))
    L = FunctionSpace(N, 'L', domain=(-2, 2))
    F0 = FunctionSpace(N, 'F', dtype='D')
    F1 = FunctionSpace(N, 'F', dtype='d')
    TT = TensorProductSpace(comm, (F0, T, F1))
    TL = TensorProductSpace(comm, (F0, L, F1))
    uT = Function(TT, buffer=h)
    uL = Function(TL, buffer=h)

    u2 = uL.backward(mesh=TT)
    uT2 = project(u2, TT)
    assert np.linalg.norm(uT2-uT)
    TT.destroy()
    TL.destroy()

@pytest.mark.parametrize('family', 'CLJ')
def test_backward_uniform(family):
    T = FunctionSpace(2*N, family, domain=(-2, 2))
    uT = Function(T, buffer=f)
    ub = uT.backward(mesh='uniform')
    xj = T.mesh(kind='uniform')
    fj = sp.lambdify(x, f)(xj)
    assert np.linalg.norm(fj-ub) < 1e-8

@pytest.mark.parametrize('family', 'CL')
def test_padding(family):
    N = 8
    B = FunctionSpace(N, family, bc=(0, 0), domain=(-1, 1))
    Bp = B.get_dealiased(1.5)
    #Bp = B
    u = Function(Bp)
    #u[:(N-2)] = np.random.random(N-2)
    u[:(N-2)] = 1
    up = Array(Bp)
    up = Bp.backward(u, kind="vandermonde")
    uf = Bp.forward(up, kind="vandermonde")
    assert np.linalg.norm(uf-u) < 1e-12
    uf = Function(Bp)
    up = Bp.backward(u, up)
    uf = Bp.forward(up, uf)
    assert np.linalg.norm(uf-u) < 1e-12, np.linalg.norm(uf-u)

    # Test padding 2D
    F = FunctionSpace(N, 'F', dtype='D')
    T = TensorProductSpace(comm, (B, F))
    Tp = T.get_dealiased(1.5)
    assert Tp.shape(False) == (12, 12)
    u = Function(T)
    u[:-2, :-1] = np.random.random(u[:-2, :-1].shape)
    up = Array(Tp)
    uc = Function(Tp)
    up = Tp.backward(u, up)
    uc = Tp.forward(up, uc)
    assert up.shape == (int(N*3/2), int(N*3/2))
    assert np.linalg.norm(u-uc) < 1e-8, np.linalg.norm(u-uc)
    T.destroy()
    Tp.destroy()

    # Test padding 3D
    F1 = FunctionSpace(N, 'F', dtype='D')
    T = TensorProductSpace(comm, (F1, F, B), dtype='D')
    Tp = T.get_dealiased(1.5)
    u = Function(T)
    #u[:-2, :-2, :-2] = np.random.random(u[:-2, :-2, :-2].shape)
    u[:-2, :-2, :-2] = 1
    up = Array(Tp)
    uc = Function(Tp)
    up = Tp.backward(u, up, kind={'legendre': 'fast'})
    uc = Tp.forward(up, uc, kind={'legendre': 'fast'})
    assert np.linalg.norm(u-uc) < 1e-8, np.linalg.norm(u-uc)
    T.destroy()
    Tp.destroy()

@pytest.mark.parametrize('family', 'CL')
def test_padding_neumann(family):
    N = 8
    B = FunctionSpace(N, family, bc={'left':('N', 0), 'right': ('N', 0)})
    Bp = B.get_dealiased(1.5)
    u = Function(B)
    u[1:-2] = np.random.random(N-3)
    up = Array(Bp)
    up = Bp.backward(u, kind="vandermonde")
    uf = Bp.forward(up, kind="vandermonde")
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
    assert up.shape == (int(N*3/2), int(N*3/2))
    assert np.linalg.norm(u-uc) < 1e-8
    T.destroy()
    Tp.destroy()

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
    T.destroy()
    Tp.destroy()

@pytest.mark.parametrize('family', ('C','L','J','Q','F','H','La'))
def test_padding_orthogonal(family):
    N = 8
    B = FunctionSpace(N, family)
    Bp = B.get_dealiased(1.5)
    u = Function(B)
    u[:] = np.random.random(u.shape)
    up = Array(Bp)
    if family != 'F':
        up = Bp.backward(u, kind="vandermonde")
        uf = Bp.forward(up, kind="vandermonde")
        assert np.linalg.norm(uf-u) < 1e-12
    if family in ('C', 'F'):
        up = Bp.backward(u, kind="fast")
        uf = Bp.forward(up, kind="fast")
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
    assert up.shape == (int(N*3/2), int(N*3/2))
    assert np.linalg.norm(u-uc) < 1e-8
    T.destroy()
    Tp.destroy()

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
    T.destroy()
    Tp.destroy()

@pytest.mark.parametrize('family', 'CLJ')
def test_padding_biharmonic(family):
    N = 8
    B = FunctionSpace(N, family, bc=(0, 0, 0, 0))
    Bp = B.get_dealiased(1.5)
    u = Function(B)
    u[:(N-4)] = np.random.random(N-4)
    up = Array(Bp)
    up = Bp.backward(u, kind="vandermonde")
    uf = Bp.forward(up, kind="vandermonde")
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
    T.destroy()
    Tp.destroy()

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
    T.destroy()
    Tp.destroy()

if __name__ == '__main__':
    test_backward()
    test_backward2D()
    #test_padding('L')
    #test_padding_biharmonic('J')
    #test_padding_neumann('C')
    #test_padding_orthogonal('F')
    #test_padding_orthogonal('C')
