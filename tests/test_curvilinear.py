import pytest
import numpy as np
import sympy as sp
from shenfun import Basis, TensorProductSpace, TrialFunction, div, grad, \
    curl, comm, VectorTensorProductSpace, TestFunction, Function, inner, \
    BlockMatrix

# Cylindrical

# Spherical

def get_function_space(space='cylinder'):
    if space == 'cylinder':
        r, theta, z = psi = sp.symbols('x,y,z', real=True, positive=True)
        rv = (r*sp.cos(theta), r*sp.sin(theta), z)
        N = 6
        F0 = Basis(N, 'F', dtype='D')
        F1 = Basis(N, 'F', dtype='d')
        L = Basis(N, 'L', domain=(0, 1))
        T = TensorProductSpace(comm, (L, F0, F1), coordinates=(psi, rv))

    elif space == 'sphere':
        r, theta, phi = psi = sp.symbols('x,y,z', real=True, positive=True)
        rv = (r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta))
        N = 6
        F = Basis(N, 'F', dtype='d')
        L0 = Basis(N, 'L', domain=(0, 1))
        L1 = Basis(N, 'L', domain=(0, np.pi))
        T = TensorProductSpace(comm, (L0, L1, F), coordinates=(psi, rv))

    return T

def test_cylinder():
    N = 6
    T = get_function_space('cylinder')
    u = TrialFunction(T)
    du = div(grad(u))
    assert du.tolatex() == '\\begin{equation*} \\frac{\\partial^2u}{\\partial^2x}+\\frac{1}{x}\\frac{\\partial u}{\\partial x}+\\frac{1}{x^{2}}\\frac{\\partial^2u}{\\partial^2y}+\\frac{\\partial^2u}{\\partial^2z} \\end{equation*}'
    V = VectorTensorProductSpace(T)
    u = TrialFunction(V)
    du = div(grad(u))
    assert du.tolatex() == '\\begin{equation*} \\left( \\frac{\\partial^2u^{x}}{\\partial^2x}+\\frac{1}{x}\\frac{\\partial u^{x}}{\\partial x}+\\frac{1}{x^{2}}\\frac{\\partial^2u^{x}}{\\partial^2y}- \\frac{2}{x}\\frac{\\partial u^{y}}{\\partial y}- \\frac{1}{x^{2}}u^{x}+\\frac{\\partial^2u^{x}}{\\partial^2z}\\right) \\mathbf{b}_{x} \\\\+\\left( \\frac{\\partial^2u^{y}}{\\partial^2x}+\\frac{3}{x}\\frac{\\partial u^{y}}{\\partial x}+\\frac{2}{x^{3}}\\frac{\\partial u^{x}}{\\partial y}+\\frac{1}{x^{2}}\\frac{\\partial^2u^{y}}{\\partial^2y}+\\frac{\\partial^2u^{y}}{\\partial^2z}\\right) \\mathbf{b}_{y} \\\\+\\left( \\frac{\\partial^2u^{z}}{\\partial^2x}+\\frac{1}{x}\\frac{\\partial u^{z}}{\\partial x}+\\frac{1}{x^{2}}\\frac{\\partial^2u^{z}}{\\partial^2y}+\\frac{\\partial^2u^{z}}{\\partial^2z}\\right) \\mathbf{b}_{z} \\\\ \\end{equation*}'

@pytest.mark.parametrize('space', ('cylinder', 'sphere'))
def test_vector_laplace(space):
    """Test that

    div(grad(u)) = grad(div(u)) - curl(curl(u))

    """
    T = get_function_space(space)
    V = VectorTensorProductSpace(T)
    u = TrialFunction(V)
    v = TestFunction(V)
    du = div(grad(u))
    dv = grad(div(u)) - curl(curl(u))
    u_hat = Function(V)
    u_hat[:] = np.random.random(u_hat.shape) + np.random.random(u_hat.shape)*1j
    A0 = inner(v, du)
    A1 = inner(v, dv)
    a0 = BlockMatrix(A0)
    a1 = BlockMatrix(A1)
    b0 = Function(V)
    b1 = Function(V)
    b0 = a0.matvec(u_hat, b0)
    b1 = a1.matvec(u_hat, b1)
    assert np.linalg.norm(b0-b1) < 1e-8

if __name__ == '__main__':
    #test_cylinder()
    test_vector_laplace('sphere')
