import pytest
from shenfun.chebyshev.la import PDMA
from shenfun.chebyshev.bases import ShenBiharmonicBasis
from shenfun import inner, TestFunction, TrialFunction, div, grad
from scipy.linalg import solve
import numpy as np
from itertools import product

N = 32

quads = ('GC', 'GL')

@pytest.mark.parametrize('quad', quads)
def test_PDMA(quad):
    SB = ShenBiharmonicBasis(N, quad=quad, plan=True)
    u = TrialFunction(SB)
    v = TestFunction(SB)
    points, weights = SB.points_and_weights(N)
    fj = np.random.randn(N)
    f_hat = np.zeros(N)
    f_hat = inner(v, fj, output_array=f_hat)

    A = inner(v, div(grad(u)))
    B = inner(v, u)
    s = SB.slice()

    H = A + B

    P = PDMA(A, B, A.scale, B.scale, solver='cython')

    u_hat = np.zeros_like(f_hat)
    u_hat[s] = solve(H.diags().toarray()[s, s], f_hat[s])

    u_hat2 = np.zeros_like(f_hat)
    u_hat2 = P(u_hat2, f_hat)
    #from IPython import embed; embed()

    assert np.allclose(u_hat2, u_hat)
