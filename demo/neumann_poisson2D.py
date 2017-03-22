r"""
Solve Poisson equation in 2D with periodic bcs in one direction
and homogeneous Neumann in the other

    \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Neumann basis for the
non-periodic direction.

The equation to solve for the Legendre basis is

    -(\nabla u, \nabla v) = (f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace, Function,\
    inner_product
from shenfun.operators import div, grad
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenNeumannBasis
Solver = shen.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
u =  cos(4*y)*sin(2*np.pi*x)*(1-x**2)
f = u.diff(x, 2) + u.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), u, 'numpy')
fl = lambdify((x, y), f, 'numpy')

# Size of discretization
N = (31, 32)

SD = Basis(N[0])
K1 = R2CBasis(N[1])
T = TensorProductSpace(comm, (SD, K1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = T.scalar_product(fj, f_hat)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
v = T.test_function()
# Create Helmholtz linear algebra solver
if basis == 'chebyshev':
    matrices = inner_product(v, div(grad(v)))
else:
    matrices = inner_product(grad(v), grad(v))

# Create Helmholtz linear algebra solver
H = Solver(**matrices, local_shape=T.local_shape())

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

