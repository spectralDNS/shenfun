import numpy as np
from scipy.linalg import solve
import pytest
from shenfun.chebyshev.la import PDMA, FDMA, TwoDMA
from shenfun import inner, TestFunction, TrialFunction, div, grad, \
    SparseMatrix, FunctionSpace, Function, Array, la
np.warnings.filterwarnings('ignore')

N = 32

quads = ('GC', 'GL')

@pytest.mark.parametrize('quad', quads)
def test_PDMA(quad):
    SB = FunctionSpace(N, 'C', bc=(0, 0, 0, 0), quad=quad)
    u = TrialFunction(SB)
    v = TestFunction(SB)
    points, weights = SB.points_and_weights(N)
    fj = Array(SB, buffer=np.random.randn(N))
    f_hat = Function(SB)
    f_hat = inner(v, fj, output_array=f_hat)
    A = inner(v, div(grad(u)))
    B = inner(v, u)
    s = SB.slice()
    H = A + B
    P = PDMA(A, B, A.scale, B.scale, solver='cython')
    u_hat = Function(SB)
    u_hat[s] = solve(H.diags().toarray()[s, s], f_hat[s])
    u_hat2 = Function(SB)
    u_hat2 = P(f_hat, u_hat2)
    assert np.allclose(u_hat2, u_hat)

def test_FDMA():
    N = 12
    SD = FunctionSpace(N, 'C', basis='ShenDirichlet')
    HH = FunctionSpace(N, 'C', basis='Heinrichs')
    u = TrialFunction(HH)
    v = TestFunction(SD)
    points, weights = SD.points_and_weights(N)
    fj = Array(SD, buffer=np.random.randn(N))
    f_hat = Function(SD)
    f_hat = inner(v, fj, output_array=f_hat)
    A = inner(v, div(grad(u)))
    B = inner(v, u)
    H = B-A
    sol = FDMA(H)
    u_hat = Function(HH)
    u_hat = sol(f_hat, u_hat)
    sol2 = la.Solve(H)
    u_hat2 = Function(HH)
    u_hat2 = sol2(f_hat, u_hat2)
    assert np.allclose(u_hat2, u_hat)

def test_TwoDMA():
    N = 12
    SD = FunctionSpace(N, 'C', basis='ShenDirichlet')
    HH = FunctionSpace(N, 'C', basis='Heinrichs')
    u = TrialFunction(HH)
    v = TestFunction(SD)
    points, weights = SD.points_and_weights(N)
    fj = Array(SD, buffer=np.random.randn(N))
    f_hat = Function(SD)
    f_hat = inner(v, fj, output_array=f_hat)
    A = inner(v, div(grad(u)))
    sol = TwoDMA(A)
    u_hat = Function(HH)
    u_hat = sol(f_hat, u_hat)
    sol2 = la.Solve(A)
    u_hat2 = Function(HH)
    u_hat2 = sol2(f_hat, u_hat2)
    assert np.allclose(u_hat2, u_hat)

@pytest.mark.parametrize('quad', quads)
def test_solve(quad):
    SD = FunctionSpace(N, 'C', bc=(0, 0), quad=quad)
    u = TrialFunction(SD)
    v = TestFunction(SD)
    A = inner(div(grad(u)), v)
    b = np.ones(N)
    u_hat = Function(SD)
    u_hat = A.solve(b, u=u_hat)

    w_hat = Function(SD)
    B = SparseMatrix(dict(A), (N-2, N-2))
    w_hat = B.solve(b, w_hat)
    assert np.all(abs(w_hat-u_hat) < 1e-8)

    ww = w_hat.repeat(N).reshape((N, N))
    bb = b.repeat(N).reshape((N, N))
    ww = B.solve(bb, ww, axis=0)
    assert np.all(abs(ww-u_hat.repeat(N).reshape((N, N))) < 1e-8)

    bb = bb.transpose()
    ww = B.solve(bb, ww, axis=1)
    assert np.all(abs(ww[:-2, :-2]-u_hat[:-2].repeat(N-2).reshape((N-2, N-2)).transpose()) < 1e-8)

if __name__ == "__main__":
    #test_solve('GC')
    test_PDMA('GC')
    #test_FDMA()
    #test_TwoDMA()
