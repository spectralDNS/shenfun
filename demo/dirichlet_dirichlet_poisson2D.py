r"""
Solve Helmholtz equation in 2D with homogeneous Dirichlet boundary conditions

    au - \nabla^2 u = f,

Use Shen's Dirichlet basis for either Chebyshev, Legendre or Jacobi polynomials.

"""
import os
import sys
import importlib
from sympy import symbols, cos, sin
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, FunctionSpace, \
    TensorProductSpace, Array
from shenfun.la import SolverGeneric2ND

comm = MPI.COMM_WORLD

assert len(sys.argv) == 4, "Call with three command-line arguments: N[0], N[1] and family (Chebyshev/Legendre)"
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi')
assert isinstance(int(sys.argv[-2]), int)
assert isinstance(int(sys.argv[-3]), int)
family = sys.argv[-1].lower()
if family == 'legendre':
    base = importlib.import_module('.'.join(('shenfun', family)))
    Solver = base.la.Helmholtz_2dirichlet
else:
    Solver = SolverGeneric2ND

# Use sympy to compute a rhs, given an analytical solution
a = 1.
x, y = symbols("x,y", real=True)
ue = (cos(4*y)*sin(2*x))*(1-x**2)*(1-y**2)
fe = a*ue - ue.diff(x, 2) - ue.diff(y, 2)

# Size of discretization
N = (int(sys.argv[-3]), int(sys.argv[-2]))

SD0 = FunctionSpace(N[0], family, bc=(0, 0), scaled=True)
SD1 = FunctionSpace(N[1], family, bc=(0, 0), scaled=True)

T = TensorProductSpace(comm, (SD0, SD1), axes=(1, 0))
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
matrices = inner(v, -div(grad(u)))
matrices += [inner(v, a*u)]

# Create Helmholtz linear algebra solver
H = Solver(matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(f_hat, u_hat)    # Solve

uq = Array(T)
uq = T.backward(u_hat, uq)

# Compare with analytical solution
uj = Array(T, buffer=ue)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh(True)
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
