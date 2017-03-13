r"""
Solve Poisson equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet in the other

    \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Dirichlet basis for the
non-periodic direction.

"""
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.chebyshev.bases import ShenDirichletBasis
from shenfun.tensorproductspace import TensorProductSpace, Function, Laplace,\
    inner_product
from shenfun.la import Helmholtz
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
u = (cos(4*x) + sin(2*y))*(1-y**2)
f = u.diff(x, 2) + u.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), u, 'numpy')
fl = lambdify((x, y), f, 'numpy')

# Size of discretization
N = (31, 32)

K1 = R2CBasis(N[0])
SD = ShenDirichletBasis(N[1])
T = TensorProductSpace(comm, (K1, SD))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = T.scalar_product(fj, f_hat)

# Get left hand side of Poisson equation
v = T.test_function()
matrices = inner_product(v, Laplace(v))

# Create Helmholtz linear algebra solver
H = Helmholtz(matrices['ADDmat'], matrices['BDDmat'], T)

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

