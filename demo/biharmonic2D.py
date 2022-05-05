r"""
Solve Biharmonic equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet and Neumann in the other

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import sys
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, FunctionSpace, comm, la, chebyshev

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
BiharmonicSolver = chebyshev.la.Biharmonic if family == 'chebyshev' else la.SolverGeneric1ND

# Use sympy to compute a rhs, given an analytical solution
x, y = sp.symbols("x,y", real=True)
ue = x**4*sp.sin(2*y)
fe = ue.diff(x, 4) + ue.diff(y, 4) + 2*ue.diff(x, 2, y, 2)

# Size of discretization
N = (30, 30)

if family == 'chebyshev':
    assert N[0] % 2 == 0, "Biharmonic solver only implemented for even numbers"

bcs = {'left': {'D': ue.subs(x, -1), 'N': ue.diff(x, 1).subs(x, -1)},
       'right': {'D': ue.subs(x, 1), 'N': ue.diff(x, 1).subs(x, 1)}}
SD = FunctionSpace(N[0], family=family, bc=bcs)
K1 = FunctionSpace(N[1], family='F')
T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))

u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation
matrices = inner(v, div(grad(div(grad(u)))))

u_hat = Function(T) # Solution spectral space

# Create linear algebra solver
H = BiharmonicSolver(matrices)

# Solve and transform to real space
u_hat = H(f_hat, u_hat)       # Solve

#H = la.SolverGeneric1ND(matrices)
#u_hat = H(f_hat, u_hat)

uq = u_hat.backward()

# Compare with analytical solution
uj = Array(T, buffer=ue)
print(abs(uj-uq).max())
assert np.allclose(uj, uq, 1e-8)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
    plt.contourf(X[0], X[1], uq)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uj)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uq-uj)
    plt.colorbar()
    plt.title('Error')
    #plt.show()
