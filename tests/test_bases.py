import pytest
import sympy as sp
from shenfun import *

bases = (chebyshev.Basis,
         chebyshev.ShenDirichletBasis,
         chebyshev.ShenNeumannBasis,
         chebyshev.ShenBiharmonicBasis,
         chebyshev.SecondNeumannBasis,
         chebyshev.BCBasis,
         legendre.Basis,
         legendre.ShenDirichletBasis,
         legendre.ShenNeumannBasis,
         legendre.ShenBiharmonicBasis,
         legendre.BCBasis,
         fourier.R2CBasis,
         fourier.C2CBasis,
         hermite.Basis,
         laguerre.Basis,
         laguerre.ShenDirichletBasis,
         jacobi.Basis,
         jacobi.ShenDirichletBasis,
         jacobi.ShenBiharmonicBasis,
         jacobi.ShenOrder6Basis
        )

@pytest.mark.parametrize('base', bases)
def test_eval_basis(base):
    B = base(8)
    i = 1
    s = B.sympy_basis(i=i)
    x = sp.symbols('x')
    mesh = B.points_and_weights()[0]
    f0 = sp.lambdify(x, s, 'numpy')(mesh)
    f1 = B.evaluate_basis(mesh, i=i)
    assert np.allclose(f0, f1)

@pytest.mark.parametrize('base', bases)
def test_eval_basis_derivative(base):
    B = base(8)
    i = 1
    s = B.sympy_basis(i=i)
    x = sp.symbols('x')
    mesh = B.points_and_weights()[0]
    for k in (1, 2, 3):
        f0 = sp.lambdify(x, s.diff(x, k), 'numpy')(mesh)
        f1 = B.evaluate_basis_derivative(mesh, i=i, k=k)
        assert np.allclose(f0, f1)

if __name__ == '__main__':
    #test_eval_basis(chebyshev.ShenDirichletBasis)
    test_eval_basis(legendre.ShenBiharmonicBasis)
