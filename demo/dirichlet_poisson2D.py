r"""
Solve Poisson equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet in the other

    \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Dirichlet basis for the
non-periodic direction.

The equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import importlib
from sympy import symbols, cos, sin
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, TensorProductSpace, comm

assert len(sys.argv) == 3, "Call with two command-line arguments"
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi')
assert isinstance(int(sys.argv[-2]), int)

# Collect solver
family = sys.argv[-1].lower()
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
a = 1
b = -1
if family == 'jacobi':
    a = 0
    b = 0
x, y = symbols("x,y")

ue = (cos(4*x) + sin(2*y))*(1 - x**2) + a*(1 - x)/2 + b*(1 + x)/2
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Size of discretization
N = (int(sys.argv[-2]), int(sys.argv[-2])+1)

SD = FunctionSpace(N[0], family=family, scaled=True, bc=(a, b))
K1 = FunctionSpace(N[1], family='F', dtype='d', domain=(-2*np.pi, 2*np.pi))
T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))

u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
matrices = inner(v, div(grad(u)))

# Create Helmholtz linear algebra solver
H = Solver(*matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(f_hat, u_hat)       # Solve
uq = u_hat.backward()
uh = uq.forward()

# Compare with analytical solution
uj = Array(T, buffer=ue)
assert np.allclose(uj, uq)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh(True)
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
    for xj in np.squeeze(X[0]):
        plt.plot((xj, xj), (np.squeeze(X[1])[0], np.squeeze(X[1])[-1]), 'k')
    for yj in np.squeeze(X[1]):
        plt.plot((np.squeeze(X[0])[0], np.squeeze(X[0])[-1]), (yj, yj), 'k')

    #plt.show()
