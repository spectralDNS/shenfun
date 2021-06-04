import pytest
import numpy as np
import sympy as sp
from shenfun import chebyshev, legendre, fourier, hermite, laguerre,\
    jacobi

bases = (
    chebyshev.Orthogonal,
    chebyshev.OrthogonalU,
    chebyshev.ShenDirichlet,
    chebyshev.DirichletU,
    chebyshev.Heinrichs,
    chebyshev.ShenNeumann,
    chebyshev.MikNeumann,
    chebyshev.CombinedShenNeumann,
    chebyshev.ShenBiharmonic,
    chebyshev.SecondNeumann,
    chebyshev.ShenBiPolar,
    chebyshev.UpperDirichlet,
    chebyshev.DirichletNeumann,
    chebyshev.NeumannDirichlet,
    chebyshev.UpperDirichletNeumann,
    legendre.Orthogonal,
    legendre.ShenDirichlet,
    legendre.ShenNeumann,
    legendre.ShenBiharmonic,
    legendre.UpperDirichlet,
    legendre.ShenBiPolar,
    legendre.ShenBiPolar0,
    legendre.NeumannDirichlet,
    legendre.DirichletNeumann,
    legendre.LowerDirichlet,
    legendre.UpperDirichletNeumann,
    legendre.DirichletNeumannDirichlet,
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

bcbases = (
    chebyshev.BCDirichlet,
    chebyshev.BCNeumann,
    chebyshev.BCBiharmonic,
    chebyshev.BCUpperDirichlet,
    chebyshev.BCNeumannDirichlet,
    chebyshev.BCDirichletNeumann,
    chebyshev.BCUpperDirichletNeumann,
    legendre.BCDirichlet,
    legendre.BCBiharmonic,
    legendre.BCNeumannDirichlet,
    legendre.BCDirichletNeumann,
    legendre.BCNeumann,
    legendre.BCBeamFixedFree,
    legendre.BCLowerDirichlet,
    legendre.BCUpperDirichlet,
    legendre.BCUpperDirichletNeumann,
    legendre.BCDirichletNeumannDirichlet,
    legendre.BCShenBiPolar0
)

nonBC = (
    'Apply',
    'Neumann2',
    'Periodic',
)

@pytest.mark.parametrize('base', bases+bcbases)
def test_eval_basis(base):
    N = 8
    B = base(N)
    x = sp.symbols('x', real=True)
    M = B.dim() if B.boundary_condition() in nonBC else N
    for i in range(M):
        s = B.sympy_basis(i, x)
        mesh = np.random.rand(3)
        f0 = sp.lambdify(x, s, 'numpy')(mesh)
        f1 = B.evaluate_basis(mesh, i=i)
        assert np.allclose(f0, f1)

@pytest.mark.parametrize('base', bases+bcbases)
def test_eval_basis_derivative(base):
    N = 8
    B = base(N)
    M = B.dim() if B.boundary_condition() in nonBC else N
    for i in range(M):
        x = sp.symbols('x', real=True)
        s = B.sympy_basis(i, x)
        mesh = np.random.rand(3)
        for k in (1, 2, 3):
            f0 = sp.lambdify(x, s.diff(x, k), 'numpy')(mesh)
            f1 = B.evaluate_basis_derivative(mesh, i=i, k=k)
            assert np.allclose(f0, f1)

if __name__ == '__main__':
    test_eval_basis_derivative(chebyshev.Heinrichs)
    test_eval_basis(chebyshev.Heinrichs)
