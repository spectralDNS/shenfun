r"""
Solve Helmholtz equation in 2D with periodic bcs in one direction
and Dirichlet in the other

   alpha u - \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Dirichlet basis for the
non-periodic direction.

The equation to solve is

     alpha (u, v) - (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import importlib
from sympy import symbols, cos, sin, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Basis, \
    Array, Function, TensorProductSpace, dx

comm = MPI.COMM_WORLD

assert len(sys.argv) == 3, "Call with two command-line arguments"
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi')
assert isinstance(int(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1]
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
alpha = 2.
x, y = symbols("x,y")
ue = (cos(4*np.pi*x) + sin(2*y))*(1-x**2)
fe = alpha*ue - ue.diff(x, 2) - ue.diff(y, 2)

# Size of discretization
N = (int(sys.argv[-2]),)*2

SD = Basis(N[0], family, bc=(0, 0), scaled=True)
K1 = Basis(N[1], 'F', dtype='d')
T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Helmholtz equation
matrices = inner(v, alpha*u - div(grad(u)))

# Create Helmholtz linear algebra solver
H = Solver(*matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
uq = Array(T)
uq = T.backward(u_hat, uq)

# Compare with analytical solution
uj = Array(T, buffer=ue)
print("Error=%2.16e" %(np.sqrt(dx(uj-uq)**2)))
assert np.allclose(uj, uq)

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
