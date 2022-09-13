import pytest
import numpy as np
import sympy as sp
from shenfun import chebyshev, chebyshevu, legendre, fourier, hermite, \
    laguerre, jacobi, ultraspherical
import importlib

trialbases = []
testbases = []
bcbases = []
for family in ('chebyshev', 'chebyshevu', 'legendre', 'ultraspherical', 'jacobi', 'laguerre', 'hermite', 'fourier'):
    base = importlib.import_module('.'.join(('shenfun', family.lower())))
    trialbases += [base.bases.__dict__.get(b) for b in base.bases.bases if b != 'Generic']
    testbases += [base.bases.__dict__.get(b) for b in base.bases.testbases if b != 'Phi6']
    bcbases += [base.bases.__dict__.get(b) for b in base.bases.bcbases]

nonBC = (
    'Apply',
    'Periodic',
    'Biharmonic*2'
)

@pytest.mark.parametrize('base', trialbases+testbases)
def test_eval_basis(base):
    N = 8
    B = base(N)
    x = sp.symbols('x', real=True)
    b = [B]
    try:
        b += [B.get_bc_space()]
    except:
        pass

    for basis in b:
        M = basis.dim()
        for i in range(M):
            s = basis.basis_function(i, x)
            mesh = np.random.rand(3)
            f0 = sp.lambdify(x, s, 'numpy')(mesh)
            f1 = basis.evaluate_basis(mesh, i=i)
            assert np.allclose(f0, f1)


@pytest.mark.parametrize('base', trialbases+testbases)
def test_eval_basis_derivative(base):
    N = 8
    B = base(N)
    b = [B]
    try:
        b += [B.get_bc_space()]
    except:
        pass
    for basis in b:
        M = basis.dim()
        for i in range(M):
            x = sp.symbols('x', real=True)
            s = basis.basis_function(i, x)
            mesh = np.random.rand(3)
            for k in (1, 2, 3):
                f0 = sp.lambdify(x, s.diff(x, k), 'numpy')(mesh)
                f1 = basis.evaluate_basis_derivative(mesh, i=i, k=k)
                assert np.allclose(f0, f1)

if __name__ == '__main__':
    #test_eval_basis_derivative(chebyshev.Heinrichs)
    test_eval_basis(chebyshev.UpperDirichlet)
