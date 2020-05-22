import pytest
import sympy as sp
from shenfun import Basis, TensorProductSpace, TrialFunction, div, grad, \
    comm, VectorTensorProductSpace

r, theta, z = psi = sp.symbols('x,y,z', real=True, positive=True)
rv = (r*sp.cos(theta), r*sp.sin(theta), z)

def test_cylinder():
    N = 6
    F0 = Basis(N, 'F', dtype='D')
    F1 = Basis(N, 'F', dtype='d')
    L = Basis(N, 'L', bc='Dirichlet', domain=(0, 1))
    T = TensorProductSpace(comm, (L, F0, F1), coordinates=(psi, rv))
    u = TrialFunction(T)
    du = div(grad(u))
    assert du.latexprint() == '\\frac{\\partial^2u}{\\partial^2x}+\\frac{1}{x}\\frac{\\partial u}{\\partial x}+\\frac{1}{x^{2}}\\frac{\\partial^2u}{\\partial^2y}+\\frac{\\partial^2u}{\\partial^2z}'
    V = VectorTensorProductSpace(T)
    u = TrialFunction(V)
    du = div(grad(u))
    assert du.latexprint() == '\\left( \\frac{\\partial^2u_{x}}{\\partial^2x}+\\frac{1}{x}\\frac{\\partial u_{x}}{\\partial x}+\\frac{1}{x^{2}}\\frac{\\partial^2u_{x}}{\\partial^2y}- \\frac{2}{x}\\frac{\\partial u_{y}}{\\partial y}- \\frac{1}{x^{2}}u_{x}+\\frac{\\partial^2u_{x}}{\\partial^2z}\\right) \\mathbf{b}_{x}+\\left( \\frac{\\partial^2u_{y}}{\\partial^2x}+\\frac{3}{x}\\frac{\\partial u_{y}}{\\partial x}+\\frac{2}{x^{3}}\\frac{\\partial u_{x}}{\\partial y}+\\frac{1}{x^{2}}\\frac{\\partial^2u_{y}}{\\partial^2y}+\\frac{\\partial^2u_{y}}{\\partial^2z}\\right) \\mathbf{b}_{y}+\\left( \\frac{\\partial^2u_{z}}{\\partial^2x}+\\frac{1}{x}\\frac{\\partial u_{z}}{\\partial x}+\\frac{1}{x^{2}}\\frac{\\partial^2u_{z}}{\\partial^2y}+\\frac{\\partial^2u_{z}}{\\partial^2z}\\right) \\mathbf{b}_{z}'

if __name__ == '__main__':
    test_cylinder()