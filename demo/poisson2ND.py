r"""
Solve Poisson equation in 2D with mixed Dirichlet and Neumann bcs

    \nabla^2 u = f,

The equation to solve is

     (\nabla^2 u, v)_w = (f, v)

Use any combination of Dirichlet and Neumann boundary conditions.

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, dx, TensorProductSpace, comm, la

# Use sympy to compute a rhs, given an analytical solution
# Choose a solution with non-zero values

xdomain = (-2, 2)
ydomain = (-2, 2)
x, y = sp.symbols("x,y", real=True)
ue = sp.chebyshevt(4, x)*sp.chebyshevt(4, y)+1
#ue = (1-y**2)*sp.sin(2*sp.pi*x)
fe = - ue.diff(x, 2) - ue.diff(y, 2)

# different types of boundary conditions
bcx = [
    {'left': ('D', ue.subs(x, xdomain[0])), 'right': ('D', ue.subs(x, xdomain[1]))},
    {'left': ('N', ue.diff(x, 1).subs(x, xdomain[0])), 'right': ('N', ue.diff(x, 1).subs(x, xdomain[1]))},
    {'left': ('N', ue.diff(x, 1).subs(x, xdomain[0])), 'right': ('D', ue.subs(x, xdomain[1]))},
    {'left': ('D', ue.subs(x, xdomain[0])), 'right': ('N', ue.diff(x, 1).subs(x, xdomain[1]))}
]

bcy = [
    {'left': ('D', ue.subs(y, ydomain[0])), 'right': ('D', ue.subs(y, ydomain[1]))},
    {'left': ('N', ue.diff(y, 1).subs(y, ydomain[0])), 'right': ('D', ue.subs(y, ydomain[1]))},
    {'left': ('D', ue.subs(y, ydomain[0])), 'right': ('N', ue.diff(y, 1).subs(y, ydomain[1]))},
    {'left': ('N', ue.diff(y, 1).subs(y, ydomain[0])), 'right': ('N', ue.diff(y, 1).subs(y, ydomain[1]))}
]

def main(N, family, bci, bcj, plotting=False):
    global fe, ue
    BX = FunctionSpace(N, family=family, bc=bcx[bci], domain=xdomain)
    BY = FunctionSpace(N, family=family, bc=bcy[bcj], domain=ydomain)
    trialspace = TensorProductSpace(comm, (BX, BY))
    testspace = trialspace.get_testspace(kind='PG')
    u = TrialFunction(trialspace)
    v = TestFunction(testspace)

    # Get f on quad points
    fj = Array(testspace, buffer=fe)

    # Compare with analytical solution
    ua = Array(trialspace, buffer=ue)

    constraint = ()
    if trialspace.use_fixed_gauge:
        mean = dx(ua, weighted=True) / dx(Array(trialspace, val=1), weighted=True)
        constraint = ((0, mean),)

    # Compute right hand side of Poisson equation
    f_hat = Function(testspace)
    f_hat = inner(v, fj, output_array=f_hat)

    # Get left hand side of Poisson equation
    A = inner(v, -div(grad(u)))

    u_hat = Function(trialspace)

    sol = la.Solver2D(A)
    u_hat = sol(f_hat, u_hat, constraints=constraint)
    uj = u_hat.backward()

    if 'pytest' not in os.environ and plotting is True:
        import matplotlib.pyplot as plt
        X, Y = trialspace.local_mesh(True)
        plt.contourf(X, Y, uj, 100);plt.colorbar()
        plt.figure()
        plt.contourf(X, Y, ua, 100);plt.colorbar()
        plt.figure()
        plt.contourf(X, Y, ua-uj, 100)
        plt.colorbar()
        plt.show()
    else:
        error = np.sqrt(inner(1, (uj-ua)**2))
        bx = "x: L-" + "".join(list(BX.bc.bc["left"].keys())) + " R-" + "".join(list(BX.bc.bc["right"].keys()))
        bx += " :: y: L-" + "".join(list(BY.bc.bc["left"].keys())) + " R-" + "".join(list(BY.bc.bc["right"].keys()))
        print(f"poisson2ND {BX.family():14s} {bx} L2 error = {error:2.6e}")
        assert error < 1e-6

if __name__ == '__main__':
    # Note - some are slower since the basis stencil is not precomputed
    for family in ('C', 'L', 'U', 'Q', 'J'):
        for bci in range(4):
            for bcj in range(4):
                main(16, family, bci, bcj)