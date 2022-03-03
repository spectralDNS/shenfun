r"""
Solve Poisson equation in 1D with homogeneous Neumann bcs

    \nabla^2 u = f

Use Shen's Neumann basis

The equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, FunctionSpace, \
    Array, Function, legendre, chebyshev, extract_bc_matrices, la, SpectralMatrix, dx

# Collect basis from either Chebyshev or Legendre submodules
assert len(sys.argv) == 3, "Call with two command-line arguments"
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'chebyshevu')
assert isinstance(int(sys.argv[-2]), int)
family = sys.argv[-1].lower()
Solver = chebyshev.la.Helmholtz if family == 'chebyshev' else la.SolverGeneric1ND

# Use sympy to compute a rhs, given an analytical solution
x = sp.symbols("x", real=True)
alpha = 0
ue = sp.cos(2*sp.pi*x)
#ue = sp.cos(5*sp.pi*(x+0.1)/2)
fe = -ue.diff(x, 2)+alpha*ue

# Size of discretization
N = int(sys.argv[-2])

bc = {'left': ('N', ue.diff(x, 1).subs(x, -1)), 'right': ('N', ue.diff(x, 1).subs(x, 1))}
SD = FunctionSpace(N, family=family, bc=bc)
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)

# Get left hand side of Poisson equation
A0 = inner(v, div(grad(u)))
B0 = inner(v, u)

# Solve
u_hat = Function(SD).set_boundary_dofs()
M = -A0
if alpha != 0:
    M += alpha*B0

# The coefficient matrix is singular if alpha=0. In that case add constraint
constraint = ((0, dx(Array(SD, buffer=ue), weighted=True)/dx(Array(SD, val=1), weighted=True)),) if alpha == 0 else ()

if alpha != 0:
    # Use Helmholtz solver
    sol = Solver(A0, B0, -1, alpha)
    u_hat = sol(f_hat, u_hat, constraints=constraint)
else:
    u_hat = M.solve(f_hat, u_hat, constraints=constraint)

# Transform to real space
uj = u_hat.backward()

# Compare with analytical solution
ua = Array(SD, buffer=ue)
print(abs(uj-ua).max())
assert np.allclose(uj, ua)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = SD.mesh()
    plt.plot(X, uj)

    plt.figure()
    plt.plot(X, ua)

    plt.figure()
    plt.plot(X, ua-uj)
    plt.title('Error')
    #plt.show()
