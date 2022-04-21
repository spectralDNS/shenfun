import pytest
import numpy as np
import sympy as sp
from shenfun import FunctionSpace, TensorProductSpace, TrialFunction, div, grad, \
    curl, comm, VectorSpace, Function, inner, \
    BlockMatrix, TestFunction as _TestFunction

def get_function_space(space='cylinder'):
    if space == 'cylinder':
        r, theta, z = psi = sp.symbols('x,y,z', real=True, positive=True)
        rv = (r*sp.cos(theta), r*sp.sin(theta), z)
        N = 6
        F0 = FunctionSpace(N, 'F', dtype='D')
        F1 = FunctionSpace(N, 'F', dtype='d')
        L = FunctionSpace(N, 'L', domain=(0, 1))
        T = TensorProductSpace(comm, (L, F0, F1), coordinates=(psi, rv))

    elif space == 'sphere':
        r, theta, phi = psi = sp.symbols('x,y,z', real=True, positive=True)
        rv = (r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta))
        N = 6
        F = FunctionSpace(N, 'F', dtype='d')
        L0 = FunctionSpace(N, 'L', domain=(0, 1))
        L1 = FunctionSpace(N, 'L', domain=(0, np.pi))
        T = TensorProductSpace(comm, (L0, L1, F), coordinates=(psi, rv, sp.Q.positive(sp.sin(theta))))

    return T

@pytest.mark.skip('skipping')
def test_cylinder():
    T = get_function_space('cylinder')
    u = TrialFunction(T)
    du = div(grad(u))
    assert du.tolatex() == '\\frac{\\partial^2 u}{\\partial x^2 }+\\frac{1}{x}\\frac{\\partial  u}{\\partial x  }+\\frac{1}{x^{2}}\\frac{\\partial^2 u}{\\partial y^2 }+\\frac{\\partial^2 u}{\\partial z^2 }'
    V = VectorSpace(T)
    u = TrialFunction(V)
    du = div(grad(u))
    assert du.tolatex() == '\\left( \\frac{\\partial^2 u^{x}}{\\partial x^2 }+\\frac{1}{x}\\frac{\\partial  u^{x}}{\\partial x  }+\\frac{1}{x^{2}}\\frac{\\partial^2 u^{x}}{\\partial y^2 }- \\frac{2}{x}\\frac{\\partial  u^{y}}{\\partial y  }- \\frac{1}{x^{2}}u^{x}+\\frac{\\partial^2 u^{x}}{\\partial z^2 }\\right) \\mathbf{b}_{x} \\\\+\\left( \\frac{\\partial^2 u^{y}}{\\partial x^2 }+\\frac{3}{x}\\frac{\\partial  u^{y}}{\\partial x  }+\\frac{2}{x^{3}}\\frac{\\partial  u^{x}}{\\partial y  }+\\frac{1}{x^{2}}\\frac{\\partial^2 u^{y}}{\\partial y^2 }+\\frac{\\partial^2 u^{y}}{\\partial z^2 }\\right) \\mathbf{b}_{y} \\\\+\\left( \\frac{\\partial^2 u^{z}}{\\partial x^2 }+\\frac{1}{x}\\frac{\\partial  u^{z}}{\\partial x  }+\\frac{1}{x^{2}}\\frac{\\partial^2 u^{z}}{\\partial y^2 }+\\frac{\\partial^2 u^{z}}{\\partial z^2 }\\right) \\mathbf{b}_{z} \\\\'

@pytest.mark.parametrize('space', ('cylinder', 'sphere'))
def test_vector_laplace(space):
    """Test that

    div(grad(u)) = grad(div(u)) - curl(curl(u))

    """
    T = get_function_space(space)
    V = VectorSpace(T)
    u = TrialFunction(V)
    v = _TestFunction(V)
    du = div(grad(u))
    dv = grad(div(u)) - curl(curl(u))
    u_hat = Function(V)
    u_hat[:] = np.random.random(u_hat.shape) + np.random.random(u_hat.shape)*1j
    g = T.coors.sg
    A0 = inner(v*g, du)
    A1 = inner(v*g, dv)
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
