import numpy as np
import pytest
from shenfun import SparseMatrix, la
import warnings

warnings.filterwarnings('ignore')

N = 10
d = [
    {0: np.arange(N)+1},
    {0: -2, 2: 1},
    {-1: 1, 0: -2, 1: 1},
    {-2: 1, 0: -2, 2: 1},
    {-2: 1, 0: -2, 2: 1, 4: 0.1},
    {-4: 0.1, -2: 1, 0: -2, 2: 1, 4: 0.1}
    ]

@pytest.mark.parametrize('di', d)
def test_XDMA(di):
    """Testing

        - DiagMA
        - TwoDMA
        - TDMA
        - TDMA_O
        - FDMA
        - PDMA
    """
    M = SparseMatrix(di, (N, N))
    sol = la.Solver(M)
    sol2 = la.Solve(M)
    b = np.ones(N)
    u_hat = np.zeros_like(b)
    u_hat = sol(b, u_hat)
    u_hat2 = np.zeros_like(b)
    u_hat2 = sol2(b, u_hat2)
    assert np.allclose(u_hat2, u_hat)
    bh = np.ones((N, N))
    uh = np.zeros_like(bh)
    uh2 = np.zeros_like(bh)
    uh = sol(bh, uh, axis=1)
    uh2 = sol(bh, uh2, axis=1)
    assert np.allclose(uh2, uh)
    assert np.allclose(uh[0], u_hat)
    uh = sol(bh, uh, axis=0)
    uh2 = sol(bh, uh2, axis=0)
    assert np.allclose(uh2, uh)
    assert np.allclose(uh[:, 0], u_hat)


if __name__ == "__main__":
    #test_solve('GC')
    #test_TDMA()
    #test_TDMA_O()
    #test_DiagMA()
    #test_PDMA('GC')
    #test_FDMA()
    #test_TwoDMA()
    test_XDMA(d[1])
