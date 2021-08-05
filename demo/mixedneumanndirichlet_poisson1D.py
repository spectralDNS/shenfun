r"""
Solve Poisson equation in 1D with mixed Dirichlet and Neumann bcs

    \nabla^2 u = f,

The equation to solve is

     (\nabla^2 u, v) = (f, v)

Use any combination of Dirichlet and Neumann boundary conditions.

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, dx, legendre, extract_bc_matrices, la

# Use sympy to compute a rhs, given an analytical solution
# Choose a solution with non-zero values

domain = (-1, 1)
x = sp.symbols("x", real=True)
ue = sp.cos(5*sp.pi*(x+0.1)/2)
fe = ue.diff(x, 2)

# 5 different types of boundary conditions
bcs = [
    {'left': ('N', ue.diff(x, 1).subs(x, domain[0])), 'right': ('N', ue.diff(x, 1).subs(x, domain[1]))},
    {'left': ('D', ue.subs(x, domain[0])), 'right': ('D', ue.subs(x, domain[1]))},
    {'left': ('N', ue.diff(x, 1).subs(x, domain[0])), 'right': ('D', ue.subs(x, domain[1]))},
    {'left': ('D', ue.subs(x, domain[0])), 'right': ('N', ue.diff(x, 1).subs(x, domain[1]))},
    {'right': (('D', ue.subs(x, domain[1])), ('N', ue.diff(x, 1).subs(x, domain[1])))}
]

def main(N, family, bci, plot=False):
    bc = bcs[bci]
    SD = FunctionSpace(N, family=family, bc=bc, domain=domain)

    u = TrialFunction(SD)
    v = TestFunction(SD)

    # Get f on quad points
    fj = Array(SD, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = Function(SD)
    f_hat = inner(v, fj, output_array=f_hat)

    # Get left hand side of Poisson equation
    A = inner(v, div(grad(u)))

    u_hat = Function(SD).set_boundary_dofs()

    constraint = ()
    if bci == 0:
        mean = dx(Array(SD, buffer=ue), weighted=True)/dx(Array(SD, val=1), weighted=True)
        constraint = ((0, mean),)

    sol = la.Solve(A)
    u_hat = sol(f_hat, u_hat, constraints=constraint)

    uj = u_hat.backward()
    uh = uj.forward()

    # Compare with analytical solution
    ua = Array(SD, buffer=ue)
    assert np.allclose(uj, ua), np.linalg.norm(uj-ua)
    if plot:
        print("Error=%2.16e" %(np.sqrt(dx((uj-ua)**2))))
        import matplotlib.pyplot as plt
        plt.plot(SD.mesh(), uj, 'b', SD.mesh(), ua, 'r')
        plt.show()

if __name__ == '__main__':
    import sys
    N = int(sys.argv[-1]) if len(sys.argv) == 2 else 36
    for family in ('C', 'L'):
        for bci in range(5):
            main(N, family, bci)
