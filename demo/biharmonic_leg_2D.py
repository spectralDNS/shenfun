r"""
Solve Biharmonic equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet and Neumann in the other

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.legendre.bases import ShenBiharmonicBasis
from shenfun.tensorproductspace import TensorProductSpace, Function,\
    Laplace, inner_product, BiharmonicOperator
from shenfun.legendre.la import Biharmonic
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
u = (sin(4*np.pi*x)*sin(2*y))*(1-x**2)
f = u.diff(x, 4) + u.diff(y, 4) + 2*u.diff(x, 2, y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), u, 'numpy')
fl = lambdify((x, y), f, 'numpy')

# Size of discretization
N = (64, 64)

SD = ShenBiharmonicBasis(N[0])
K1 = R2CBasis(N[1])
T = TensorProductSpace(comm, (SD, K1))
X = T.local_mesh(True)

# Discretization
use_integration_by_parts = False

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side of Biharmonic equation
f_hat = Function(T)
f_hat = T.scalar_product(fj, f_hat)

# Get left hand side of Biharmonic equation
v = T.test_function()
if use_integration_by_parts:
    matrices = inner_product(Laplace(v), Laplace(v))
else:
    matrices = inner_product(v, BiharmonicOperator(v))

# Create Helmholtz linear algebra solver
S, A, B = matrices['SBBmat'], matrices['PBBmat'], matrices['BBBmat']
H = Biharmonic(S, A, B, S.scale, A.scale, B.scale, T.local_shape())

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

