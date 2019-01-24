r"""
Solve Biharmonic equation in 2D with homogeneous Dirichlet and
Neumann boundary conditions in both directions

    \nabla^4 u = f,

Use Shen's Biharmonic basis for both directions.

Note that we are solving

    (v, \nabla^4 u) = (v, f)

with the Chebyshev basis, and

    (div(grad(v), div(grad(u)) = -(v, f)

for the Legendre basis.

"""
import sys
import os
from sympy import symbols, sin, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, Basis
from shenfun.la import SolverGeneric2NP

comm = MPI.COMM_WORLD

assert comm.Get_size() == 1, "Two non-periodic directions only have solver implemented for serial"

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
ue = (sin(2*np.pi*x)*sin(4*np.pi*y))*(1-x**2)*(1-y**2)
fe = ue.diff(x, 4) + ue.diff(y, 4) + 2*ue.diff(x, 2, y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (36, 36)

S0 = Basis(N[0], family=family, bc='Biharmonic')
S1 = Basis(N[1], family=family, bc='Biharmonic')
T = TensorProductSpace(comm, (S0, S1), axes=(0, 1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fl(*X))

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation
if family == 'chebyshev': # No integration by parts due to weights
    matrices = inner(v, div(grad(div(grad(u)))))
else: # Use form with integration by parts.
    matrices = inner(div(grad(v)), div(grad(u)))

# Create linear algebra solver
H = SolverGeneric2NP(matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(f_hat, u_hat)       # Solve
uq = u_hat.backward()

# Compare with analytical solution
uj = ul(*X)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

points = np.array([[0.2, 0.3], [0.1, 0.5]])
p = T.eval(points, u_hat, method=2)
assert np.allclose(p, ul(*points))

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0], X[1], uq)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uj)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uq-uj)
    plt.colorbar()
    plt.title('Error')
    plt.show()
