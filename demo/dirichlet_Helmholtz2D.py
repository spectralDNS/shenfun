r"""
Solve Helmholtz equation in 2D with periodic bcs in one direction
and Dirichlet in the other

   alpha u - \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Dirichlet basis for the
non-periodic direction.

The equation to solve for the Legendre basis is

     alpha (u, v) + (\nabla u, \nabla v) = (f, v)

whereas for Chebyshev we solve

     alpha (u, v) - (\nabla^2 u, v) = (f, v)

"""
import sys, os
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project, Array
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except:
    plt = None

comm = MPI.COMM_WORLD

assert len(sys.argv) == 3, "Call with two command-line arguments"
assert sys.argv[-1] in ('legendre', 'chebyshev')
assert isinstance(eval(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1]
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis
Solver = shen.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
alpha = 2.
x, y = symbols("x,y")
ue = (cos(4*np.pi*x) + sin(2*y))*(1-x**2)
fe = alpha*ue - ue.diff(x, 2) - ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (eval(sys.argv[-2]),)*2

SD = Basis(N[0], scaled=True)
K1 = R2CBasis(N[1])
T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = Array(T)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Helmholtz equation
if basis == 'chebyshev':
    matrices = inner(v, alpha*u - div(grad(u)))
else:
    matrices = inner(grad(v), grad(u))    # Both ADDmat and BDDmat
    B = inner(v, alpha*u)
    matrices['BDDmat'] += B

# Create Helmholtz linear algebra solver
H = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
uq = Function(T, False)
uq = T.backward(u_hat, uq)

# Compare with analytical solution
uj = ul(*X)
print("Error=%2.16e" %(np.linalg.norm(uj-uq)))
assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.contourf(X[0], X[1], uq)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uj)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uq-uj)
    plt.colorbar()
    plt.title('Error')

    plt.show()
