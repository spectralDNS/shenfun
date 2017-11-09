r"""
Solve Poisson equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet in the other

    \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Dirichlet basis for the
non-periodic direction.

The equation to solve for the Legendre basis is

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
a = -2
b = 2
x, y = symbols("x,y")
ue = (cos(4*np.pi*y) + sin(2*x))*(1 - y**2) + a*(1 + y)/2. + b*(1 - y)/2.
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (eval(sys.argv[-2]), eval(sys.argv[-2]))

SD = Basis(N[1], scaled=True, bc=(a, b))
K1 = R2CBasis(N[0])
T = TensorProductSpace(comm, (K1, SD), axes=(1, 0))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = Array(T)
f_hat = inner(v, fj, output_array=f_hat)
if basis == 'legendre':
    f_hat *= -1.

#from IPython import embed; embed()
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
uq = Function(T, False)
uq = T.backward(u_hat, uq)

# Compare with analytical solution
uj = ul(*X)
#assert np.allclose(uj, uq)

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

    plt.figure()
    X = T.local_mesh()
    for x in np.squeeze(X[0]):
        plt.plot((x, x), (np.squeeze(X[1])[0], np.squeeze(X[1])[-1]), 'k')
    for y in np.squeeze(X[1]):
        plt.plot((np.squeeze(X[0])[0], np.squeeze(X[0])[-1]), (y, y), 'k')

    plt.show()
