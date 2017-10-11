r"""
Solve Poisson equation in 3D with periodic bcs in two directions
and homogeneous Neumann in the third

    \nabla^2 u = f,

Use Fourier basis for the periodic directions and Shen's Neumann basis for the
non-periodic direction.

The equation to solve for the Legendre basis is

    -(\nabla u, \nabla v) = (f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys, os
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project, Dx, TensorProductSpace
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except:
    plt = None

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenNeumannBasis
Solver = shen.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue =  sin(6*z)*cos(4*y)*sin(2*np.pi*x)*(1-x**2)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = (32, 32, 32)

SD = Basis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
T = TensorProductSpace(comm, (SD, K1, K2), dtype='d')
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if basis == 'chebyshev':
    matrices = inner(v, div(grad(u)))
else:
    matrices = inner(grad(v), grad(u))

# Create Helmholtz linear algebra solver
H = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
u = T.backward(u_hat)

# Compare with analytical solution
uj = ul(*X)
print(abs(uj-u).max())
assert np.allclose(uj, u)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], u[:, :, 2])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uj[:, :, 2])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], u[:, :, 2]-uj[:, :, 2])
    plt.colorbar()
    plt.title('Error')

    plt.show()
