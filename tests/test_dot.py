import pytest
import numpy as np
import sympy as sp
from shenfun import Function, Array, FunctionSpace, VectorSpace, \
    TensorSpace, TensorProductSpace, dot, project, comm, div, grad, config

x, y =sp.symbols('x,y', real=True)

def test_dot():
    N = 3
    D = FunctionSpace(N, 'C')
    T = TensorProductSpace(comm, (D, D))
    V = VectorSpace(T)
    S = TensorSpace(T)
    u = Function(V, buffer=(x, -y))
    bu = dot(u, u)
    assert np.linalg.norm(bu-Function(T, buffer=x**2+y**2)) < 1e-8
    gradu = project(grad(u), S)
    du = dot(gradu, u, forward_output=True)
    xy = Function(V, buffer=(x, y))
    assert np.linalg.norm(du - xy) < 1e-12
    du = dot(u, gradu, forward_output=True)
    assert np.linalg.norm(du - xy) < 1e-12
    gu = dot(gradu, gradu, forward_output=True)
    dd = Function(S, buffer=(1, 0, 0, 1))
    assert np.linalg.norm(gu - dd) < 1e-12
    T.destroy()

#@pytest.mark.skip('skipping')
def test_dot_curvilinear():
    # Define spherical coordinates without the poles
    N = 10
    basisvectors = config['basisvectors']
    config['basisvectors'] = 'normal'
    r, theta, phi = psi =sp.symbols('x,y,z', real=True, positive=True)
    rv = (r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta))
    L0 = FunctionSpace(N, 'L', domain=(0.5, 1))
    L1 = FunctionSpace(N, 'L', domain=(0.5, np.pi/2))
    F1 = FunctionSpace(N, 'F', dtype='d')
    T = TensorProductSpace(comm, (L0, L1, F1), coordinates=(psi, rv, sp.Q.positive(sp.sin(theta))))
    V = VectorSpace(T)
    S = TensorSpace(T)
    u = Function(V, buffer=(0, sp.sin(theta), sp.cos(phi)))
    u2 = dot(u, u)
    gij = T.coors.get_metric_tensor(config['basisvectors'])
    ue = Function(T, buffer=sp.sin(theta)**2*gij[1, 1]+sp.cos(phi)**2*gij[2, 2])
    assert np.linalg.norm(ue-u2) < 1e-6
    u = Function(V, buffer=(0, r, 0))
    gu = Function(S, buffer=(0, -1, 0, 1, 0, 0, 0, 0, 1/sp.tan(theta)))
    bu = dot(gu, u)
    but = Function(V, buffer=(-r, 0, 0))
    assert np.linalg.norm(bu-but) < 1e-6
    ub = dot(u, gu)
    ubt = Function(V, buffer=(r, 0, 0))
    assert np.linalg.norm(ub-ubt) < 1e-6
    gg = dot(gu, gu)
    ggt = Function(S, buffer=(-1, 0, 0, 0, -1, 0, 0, 0, 1/sp.tan(theta)**2))
    assert np.linalg.norm(gg-ggt) < 1e-4
    T.destroy()
    config['basisvectors'] = basisvectors

if __name__ == '__main__':
    import sys
    #test_dot()
    test_dot_curvilinear()
