r"""
Solve Poisson equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet in the third

    \nabla^2 u = f,

Use Fourier basis for the periodic directions and Shen's Dirichlet basis for the
remaining non-periodic direction. Discretization leads to a Holmholtz problem
to be solved.

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
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis
Solver = shen.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
u = (cos(4*x) + sin(2*y) + sin(4*z))*(1-x**2)
f = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), u, 'numpy')
fl = lambdify((x, y, z), f, 'numpy')

# Size of discretization
N = (36, 44, 24)

SD = Basis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
T = TensorProductSpace(comm, (SD, K1, K2))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1], X[2])

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = T.scalar_product(fj, f_hat)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
v = T.test_function()
if basis == 'chebyshev':
    matrices = inner_product(v, div(grad(v)))
else:
    matrices = inner_product(grad(v), grad(v))

# Create Helmholtz linear algebra solver
H = Solver(**matrices, local_shape=T.local_shape())

# Solve and transform to real space
u = Function(T, False)        # Solution real space
u_hat = Function(T)           # Solution spectral space
t0 = time.time()
u_hat = H(u_hat, f_hat)       # Solve
print("Timing solve = {}".format(time.time()-t0))

u = T.backward(u_hat, u, fast_transform=False)

# Compare with analytical solution
uj = ul(X[0], X[1], X[2])
print(abs(uj-u).max())
assert np.allclose(uj, u)

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
#plt.show()

