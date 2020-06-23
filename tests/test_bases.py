import pytest
import numpy as np
import sympy as sp
from shenfun import chebyshev, legendre, fourier, hermite, laguerre,\
    jacobi

bases = (chebyshev.Orthogonal,
         chebyshev.ShenDirichlet,
         chebyshev.ShenNeumann,
         chebyshev.ShenBiharmonic,
         chebyshev.SecondNeumann,
         chebyshev.ShenBiPolar,
         chebyshev.UpperDirichlet,
         chebyshev.DirichletNeumann,
         chebyshev.BCDirichlet,
         chebyshev.BCBiharmonic,
         legendre.Orthogonal,
         legendre.ShenDirichlet,
         legendre.ShenNeumann,
         legendre.ShenBiharmonic,
         legendre.UpperDirichlet,
         legendre.ShenBiPolar,
         legendre.ShenBiPolar0,
         legendre.NeumannDirichlet,
         legendre.DirichletNeumann,
         legendre.BCDirichlet,
         legendre.BCBiharmonic,
         fourier.R2C,
         fourier.C2C,
         hermite.Orthogonal,
         laguerre.Orthogonal,
         laguerre.ShenDirichlet,
         jacobi.Orthogonal,
         jacobi.ShenDirichlet,
         jacobi.ShenBiharmonic,
         jacobi.ShenOrder6
        )

@pytest.mark.parametrize('base', bases)
def test_eval_basis(base):
    B = base(8)
    i = 1
    x = sp.symbols('x')
    s = B.sympy_basis(i, x)
    mesh = B.points_and_weights()[0]
    f0 = sp.lambdify(x, s, 'numpy')(mesh)
    f1 = B.evaluate_basis(mesh, i=i)
    assert np.allclose(f0, f1)

@pytest.mark.parametrize('base', bases)
def test_eval_basis_derivative(base):
    B = base(8)
    i = 1
    x = sp.symbols('x')
    s = B.sympy_basis(i, x)
    mesh = B.points_and_weights()[0]
    for k in (1, 2, 3):
        f0 = sp.lambdify(x, s.diff(x, k), 'numpy')(mesh)
        f1 = B.evaluate_basis_derivative(mesh, i=i, k=k)
        assert np.allclose(f0, f1)

if __name__ == '__main__':
    test_eval_basis_derivative(legendre.BCBiharmonic)
    #test_eval_basis(legendre.ShenNeumannBasis)
