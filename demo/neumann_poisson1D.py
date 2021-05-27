r"""
Solve Poisson equation in 1D with homogeneous Neumann bcs

    \nabla^2 u = f

Use Shen's Neumann basis

The equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import importlib
from sympy import symbols, sin, cos, pi, chebyshevt
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, FunctionSpace, \
    Array, Function, legendre, chebyshev, extract_bc_matrices, la, SpectralMatrix

# Collect basis from either Chebyshev or Legendre submodules
assert len(sys.argv) == 3, "Call with two command-line arguments"
assert sys.argv[-1].lower() in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)
family = sys.argv[-1].lower()

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x", real=True)
alpha = 0
ue = sin(pi*x)*(1-x**2)
fe = -ue.diff(x, 2)+alpha*ue

# Size of discretization
N = int(sys.argv[-2])

# alpha=0 requires a fixed gauge, but not alpha!=0 -> mean
SD = FunctionSpace(N, family=family, bc={'left': ('N', 0),
                                         'right': ('N', 0)},
                                         mean=0 if alpha==0 else None,
                                         basis='ShenNeumann')
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
u_hat = Function(SD)
M = alpha*B0-A0
sol = la.Solve(M, SD)
u_hat = sol(f_hat, u_hat)

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
