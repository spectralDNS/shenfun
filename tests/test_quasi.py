import pytest
import sympy as sp
import scipy.sparse as scp
from shenfun import FunctionSpace, TrialFunction, TestFunction, \
    chebyshev, inner, div, grad, la, Array, Function, np
np.warnings.filterwarnings('ignore')

x = sp.Symbol('x', real=True)
ue = sp.sin(sp.pi*sp.cos(sp.pi*x))
fe = ue-ue.diff(x, 2)

@pytest.mark.parametrize('basis', ('ShenDirichlet', 'ShenNeumann'))
def test_quasiGalerkin(basis):
    N = 40
    T = FunctionSpace(N, 'C')
    S = FunctionSpace(N, 'C', basis=basis)
    u = TrialFunction(S)
    v = TestFunction(T)
    A = inner(v, div(grad(u)))
    B = inner(v, u)
    Q = chebyshev.quasi.QIGmat(N)
    A = Q*A
    B = Q*B
    M = B-A
    sol = la.Solve(M, S)
    f_hat = inner(v, Array(T, buffer=fe))
    f_hat[:-2] = Q.diags('csc')*f_hat[:-2]
    u_hat = Function(S)
    u_hat = sol(f_hat, u_hat)
    uj = u_hat.backward()
    ua = Array(S, buffer=ue)
    bb = Q*np.ones(N)
    assert abs(np.sum(bb[:8])) < 1e-8
    if S.boundary_condition().lower() == 'neumann':
        xj, wj = S.points_and_weights()
        ua -= np.sum(ua*wj)/np.pi # normalize
        uj -= np.sum(uj*wj)/np.pi # normalize
    assert np.sqrt(inner(1, (uj-ua)**2)) < 1e-5

@pytest.mark.parametrize('bc', ('Dirichlet', 'Neumann'))
def test_quasiTau(bc):
    N = 40
    T = FunctionSpace(N, 'C')
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(v, div(grad(u)))
    B = inner(v, u)
    Q = chebyshev.quasi.QITmat(N)
    A = Q*A
    B = Q*B
    bb = Q*np.ones(N)
    assert abs(np.sum(bb[:8])) < 1e-8
    M = B-A
    M0 = M.diags().tolil()
    if bc == 'Dirichlet':
        M0[0] = np.ones(N)
        nn = np.ones(N)
        nn[1::2] = -1
        M0[1] = nn
    elif bc == 'Neumann':
        nn = np.arange(N)**2
        M0[0] = nn.copy()
        nn[1::2] *= -1
        M0[1] = nn
    M0 = M0.tocsc()
    f_hat = inner(v, Array(T, buffer=fe))
    gh = Q.diags('csc')*f_hat
    gh[:2] = 0
    u_hat = Function(T)
    u_hat[:] = scp.linalg.spsolve(M0, gh)
    uj = u_hat.backward()
    ua = Array(T, buffer=ue)

    if bc == 'Neumann':
        xj, wj = T.points_and_weights()
        ua -= np.sum(ua*wj)/np.pi # normalize
        uj -= np.sum(uj*wj)/np.pi # normalize
    assert np.sqrt(inner(1, (uj-ua)**2)) < 1e-5

if __name__ == '__main__':
    test_quasiGalerkin('ShenDirichlet')
    test_quasiGalerkin('ShenNeumann')
    test_quasiTau('Dirichlet')
    test_quasiTau('Neumann')
