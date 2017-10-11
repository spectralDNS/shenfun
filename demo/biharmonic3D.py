r"""
Solve Biharmonic equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet and Neumann in the remaining third

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

Note that we are solving

    (v, \nabla^4 u) = (v, f)

with the Chebyshev basis, and

    (div(grad(v), div(grad(u)) = -(v, f)

for the Legendre basis.

"""
import sys, os
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, Dx, project
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except:
    plt = None

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
BiharmonicBasis = shen.bases.ShenBiharmonicBasis
BiharmonicSolver = shen.la.Biharmonic

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = (sin(4*np.pi*z)*sin(2*y)*cos(4*x))*(1-z**2)
fe = ue.diff(x, 4) + ue.diff(y, 4) + ue.diff(z, 4) + 2*ue.diff(x, 2, y, 2) + 2*ue.diff(x, 2, z, 2) + 2*ue.diff(y, 2, z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = (64, 64, 64)

SD = BiharmonicBasis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
T = TensorProductSpace(comm, (K1, K2, SD), axes=(2,0,1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)

# Get left hand side of Poisson equation
if basis == 'chebyshev': # No integration by parts due to weights
    matrices = inner(v, div(grad(div(grad(u)))))
else: # Use form with integration by parts. Note that BiharmonicOperator also works for Legendre though
    matrices = inner(div(grad(v)), div(grad(u)))

# Create Helmholtz linear algebra solver
H = BiharmonicSolver(**matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
uq = T.backward(u_hat)

# Compare with analytical solution
uj = ul(*X)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 8])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uj[:, :, 8])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 8]-uj[:, :, 8])
    plt.colorbar()
    plt.title('Error')

    plt.show()


