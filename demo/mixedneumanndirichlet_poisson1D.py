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
    Array, Function, FunctionSpace, dx, legendre, extract_bc_matrices

# Use sympy to compute a rhs, given an analytical solution
# Choose a solution with non-zero values

domain = (-2, 1)
x = sp.symbols("x", real=True)
ue = sp.cos(5*sp.pi*(x+0.1)/2)
fe = ue.diff(x, 2)

# The pure Neumann requires the value of the mean
x_map = -1 + (x-domain[0])*2/(domain[1]-domain[0])
mean = {
    'c': sp.integrate(ue/sp.sqrt(1-x_map**2), (x, domain[0], domain[1])).evalf(),
    'l': sp.integrate(ue, (x, domain[0], domain[1])).evalf()
}

# 5 different types of boundary conditions
bcs = [
    {'left': ('N', ue.diff(x, 1).subs(x, domain[0])), 'right': ('N', ue.diff(x, 1).subs(x, domain[1]))},
    {'left': ('D', ue.subs(x, domain[0])), 'right': ('D', ue.subs(x, domain[1]))},
    {'left': ('N', ue.diff(x, 1).subs(x, domain[0])), 'right': ('D', ue.subs(x, domain[1]))},
    {'left': ('D', ue.subs(x, domain[0])), 'right': ('N', ue.diff(x, 1).subs(x, domain[1]))},
    {'right': (('D', ue.subs(x, domain[1])), ('N', ue.diff(x, 1).subs(x, domain[1])))}
]

def main(N, family, bci):
    bc = bcs[bci]
    if bci == 0:
        SD = FunctionSpace(N, family=family, bc=bc, domain=domain, mean=mean[family.lower()])
    else:
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
    if isinstance(A, list):
        bc_mat = extract_bc_matrices([A])
        A = A[0]
        f_hat -= bc_mat[0].matvec(u_hat, Function(SD))

    u_hat = A.solve(f_hat, u_hat)
    uj = u_hat.backward()
    uh = uj.forward()

    # Compare with analytical solution
    ua = Array(SD, buffer=ue)
    assert np.allclose(uj, ua), np.linalg.norm(uj-ua)
    if 'pytest' not in os.environ:
        print("Error=%2.16e" %(np.sqrt(dx((uj-ua)**2))))
        import matplotlib.pyplot as plt
        plt.plot(SD.mesh(), uj, 'b', SD.mesh(), ua, 'r')
        #plt.show()

if __name__ == '__main__':
    import sys
    N = int(sys.argv[-1]) if len(sys.argv) == 2 else 36
    for family in ('C', 'L'):
        for bci in range(5):
            main(N, family, bci)
