r"""
Solve Biharmonic equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet and Neumann in the other

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis
from shenfun.tensorproductspace import TensorProductSpace, Function,\
     inner_product
from shenfun.operators import BiharmonicOperator, Laplace, div, grad
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
BiharmonicBasis = shen.bases.ShenBiharmonicBasis
BiharmonicSolver = shen.la.Biharmonic

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
u = (sin(4*np.pi*x)*cos(4*y))*(1-x**2)
f = u.diff(x, 4) + u.diff(y, 4) + 2*u.diff(x, 2, y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), u, 'numpy')
fl = lambdify((x, y), f, 'numpy')

# Size of discretization
N = (64, 64)

SD = BiharmonicBasis(N[0])
K1 = R2CBasis(N[1])
T = TensorProductSpace(comm, (SD, K1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = T.scalar_product(fj, f_hat)

# Get left hand side of Poisson equation
v = T.test_function()
if basis == 'chebyshev': # No integration by parts due to weights
    matrices = inner_product(v, div(grad(div(grad(v)))))
else: # Use form with integration by parts. Note that BiharmonicOperator also works for Legendre though
    matrices = inner_product(div(grad(v)), div(grad(v)))

# Create Helmholtz linear algebra solver
H = BiharmonicSolver(**matrices, local_shape=T.local_shape())

# Solve and transform to real space
u = Function(T, False)        # Solution real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
u = T.backward(u_hat, u)

# Compare with analytical solution
uj = ul(X[0], X[1])
print(abs(uj-u).max())
assert np.allclose(uj, u)

plt.figure()
plt.contourf(X[0], X[1], u)
plt.colorbar()

plt.figure()
plt.contourf(X[0], X[1], uj)
plt.colorbar()

plt.figure()
plt.contourf(X[0], X[1], u-uj)
plt.colorbar()
plt.title('Error')
#plt.show()

