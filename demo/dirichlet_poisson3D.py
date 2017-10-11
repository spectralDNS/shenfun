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
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project, Dx
import time
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except:
    plt = None

comm = MPI.COMM_WORLD

assert len(sys.argv) == 3
assert sys.argv[-1] in ('legendre', 'chebyshev')
assert isinstance(eval(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1]
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis
Solver = shen.la.Helmholtz
regtest = False

# Use sympy to compute a rhs, given an analytical solution
a = 0
b = 0
x, y, z = symbols("x,y,z")
ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-x**2) + a*(1 + x)/2. + b*(1 - x)/2.
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = eval(sys.argv[-2])
N = [N, N+1, N+2]
#N = (14, 15, 16)

SD = Basis(N[0], bc=(a, b))
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
T = TensorProductSpace(comm, (SD, K1, K2), axes=(0,1,2), slab=True)
X = T.local_mesh() # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

K = T.local_wavenumbers()

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
t0 = time.time()
u_hat = H(u_hat, f_hat)       # Solve
uq = T.backward(u_hat, fast_transform=False)

# Compare with analytical solution
uj = ul(*X)
error = comm.reduce(np.linalg.norm(uj-uq)**2)
if comm.Get_rank() == 0 and regtest == True:
    print("Error=%2.16e" %(np.sqrt(error)))
assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.contourf(X[1][0, :, 0], X[0][:, 0, 0], uq[:, :, 2])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[1][0, :, 0], X[0][:, 0, 0], uj[:, :, 2])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[1][0, :, 0], X[0][:, 0, 0], uq[:, :, 2]-uj[:, :, 2])
    plt.colorbar()
    plt.title('Error')

    plt.show()
