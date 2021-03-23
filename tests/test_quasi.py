import pytest
import sympy as sp
from shenfun import FunctionSpace, TrialFunction, TestFunction, \
    chebyshev, inner, div, grad, la, Array, Function, np
np.warnings.filterwarnings('ignore')

x = sp.Symbol('x', real=True)
ue = sp.sin(sp.pi*sp.cos(sp.pi*x))
fe = ue-ue.diff(x, 2)

@pytest.mark.parametrize('basis', ('ShenDirichlet', 'ShenNeumann'))
def test_quasi(basis):
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
    if S.boundary_condition().lower() == 'neumann':
        xj, wj = S.points_and_weights()
        ua -= np.sum(ua*wj)/np.pi # normalize
        uj -= np.sum(uj*wj)/np.pi # normalize
    assert np.sqrt(inner(1, (uj-ua)**2)) < 1e-5

if __name__ == '__main__':
    test_quasi('ShenDirichlet')
    test_quasi('ShenNeumann')
