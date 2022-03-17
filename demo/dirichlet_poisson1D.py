r"""
Solve Poisson equation in 1D with homogeneous Dirichlet bcs

    \nabla^2 u = f,

The equation to solve is

    (\nabla^2 u, v) = (f, v)

"""
import sys
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, dx

assert len(sys.argv) == 3, 'Call with two command-line arguments, N and family'
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi', 'chebyshevu')
assert isinstance(int(sys.argv[-2]), int)

# Get family from args
family = sys.argv[-1].lower()

# Use sympy to compute a rhs, given an analytical solution
x = sp.symbols("x", real=True)
ue = sp.sin(4*sp.pi*x)
fe = ue.diff(x, 2)

# Size of discretization
N = int(sys.argv[-2])
SD = FunctionSpace(N, family=family, bc='u(-1)=0 && u(1)=0', alpha=1, beta=2) # alpha, beta are ignored by all other than jacobi
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(SD)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
A = inner(v, div(grad(u)))

u_hat = Function(SD)
u_hat = A.solve(f_hat, u_hat)
uj = u_hat.backward()

# Compare with analytical solution
ua = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.sqrt(dx((uj-ua)**2))))
assert np.linalg.norm(u_hat - u_hat.backward().forward()) < 1e-8
