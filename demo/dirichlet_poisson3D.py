r"""
Solve Poisson equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet in the third

    \nabla^2 u = f,

Use Fourier basis for the periodic directions and Shen's Dirichlet basis for the
remaining non-periodic direction. Discretization leads to a Holmholtz problem.

Note that the equation to solve for Legendre basis is

     (\nabla u, \nabla v) = -(f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys, os
import importlib
from sympy import symbols, cos, sin, lambdify
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, Basis, TensorProductSpace
import time
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

comm = MPI.COMM_WORLD

assert len(sys.argv) == 3
assert sys.argv[-1].lower() in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower()
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz
regtest = True

# Use sympy to compute a rhs, given an analytical solution
a = 0
b = 1
x, y, z = symbols("x,y,z")
ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-x**2) + a*(1 + x)/2. + b*(1 - x)/2.
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = int(sys.argv[-2])
N = [N, N+1, N+2]
#N = (14, 15, 16)

SD = Basis(N[0], family=family, bc=(a, b))
K1 = Basis(N[1], family='F', dtype='D')
K2 = Basis(N[2], family='F', dtype='d')
T = TensorProductSpace(comm, (SD, K1, K2), axes=(0, 1, 2), slab=True)
X = T.local_mesh()
u = TrialFunction(T)
v = TestFunction(T)

K = T.local_wavenumbers()

# Get f on quad points
fj = Array(T, buffer=fl(*X))

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)
if family == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if family == 'chebyshev':
    matrices = inner(v, div(grad(u)))
else:
    matrices = inner(grad(v), grad(u))

# Create Helmholtz linear algebra solver
H = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
t0 = time.time()
u_hat = H(u_hat, f_hat)       # Solve
uq = T.backward(u_hat, fast_transform=True)

# Compare with analytical solution
uj = ul(*X)
error = comm.reduce(np.linalg.norm(uj-uq)**2)
if comm.Get_rank() == 0 and regtest == True:
    print("Error=%2.16e" %(np.sqrt(error)))
#assert np.allclose(uj, uq)

if plt is not None and not 'pytest' in os.environ:
    plt.figure()
    plt.contourf(X[2][0, 0, :], X[0][:, 0, 0], uq[:, 2, :])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[2][0, 0, :], X[0][:, 0, 0], uj[:, 2, :])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[2][0, 0, :], X[0][:, 0, 0], uq[:, 2, :]-uj[:, 2, :])
    plt.colorbar()
    plt.title('Error')

    #plt.show()
