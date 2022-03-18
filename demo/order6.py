r"""
Solve 6th order equation in 1D

    u(x)^(6) - a(x)u(x) = f(x), for x in [-1, 1]

where a(x) and f(x) are given. There are 6 boundary conditions
u(-1) = a, u'(-1) = b,  u''(-1) = c, u(1) = d, u'(1) = e,  u''(1) = f,
which are computed from the manufactured solution.

"""
import os
import sympy as sp
import numpy as np
from shenfun import *

# Manufactured solution that satisfies boundary conditions
x = sp.Symbol("x", real=True)

def main(N, family, sol=0, alpha=0, beta=0):

    if sol == 0:
        domain = (-1, 1)
        ue = sp.sin(sp.pi*x*2)*sp.exp(-x/2)
        measure = -1

    elif sol == 1:
        domain = (0, 1.)
        ue = x**3*(1-x)**3
        measure = -sp.exp(-x)
    fe = ue.diff(x, 6) + measure*ue

    bc = {'left': {'D': ue.subs(x, domain[0]).n(),
                   'N': ue.diff(x, 1).subs(x, domain[0]).n(),
                   'N2': ue.diff(x, 2).subs(x, domain[0]).n()},
          'right': {'D': ue.subs(x, domain[1]).n(),
                    'N': ue.diff(x, 1).subs(x, domain[1]).n(),
                    'N2': ue.diff(x, 2).subs(x, domain[1]).n()}}

    SD = FunctionSpace(N, family, domain=domain, bc=bc, alpha=alpha, beta=beta)

    X = SD.mesh()
    u = TrialFunction(SD)
    v = TestFunction(SD)

    # Note - integration by parts only valid for alpha=beta=0
    #S = inner(Dx(v, 0, 3), -Dx(u, 0, 3))
    S = inner(v, Dx(u, 0, 6))
    B = inner(v*measure, u)
    if sol == 0:
        M = B + [S]
    else:
        M = B + S

    sol = la.Solver(M)

    # Get f on quad points
    fj = Array(SD, buffer=fe)
    f_hat = inner(v, fj)

    u_hat = Function(SD)
    u_hat = sol(f_hat, u_hat)

    uq = Array(SD, buffer=ue)
    print(np.sqrt(inner(1, (u_hat.backward()-uq)**2)))
    if 'pytest 'in os.environ:
        assert np.sqrt(inner(1, (u_hat.backward()-uq)**2)) < 1e-6

if __name__ == '__main__':
    N = 24
    for sol in (0, 1):
        for family in ('legendre', 'chebyshev', 'chebyshevu', 'jacobi'):
            main(N, family, sol)
