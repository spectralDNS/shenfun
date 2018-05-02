r"""
Solve Poisson equation in 1D with possibly inhomogeneous Dirichlet bcs

    \nabla^2 u = f,

The equation to solve for a Legendre basis is

     (\nabla u, \nabla v) = -(f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys
import importlib
from sympy import symbols, sin, lambdify
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Basis

assert len(sys.argv) == 3, 'Call with two command-line arguments'
assert sys.argv[-1] in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)

# Get family from args
family = sys.argv[-1].lower()

# Use sympy to compute a rhs, given an analytical solution
a = 0.
b = 0.
x = symbols("x")
ue = sin(4*np.pi*x)*(1-x**2) + a*(1 + x)/2. + b*(1 - x)/2.
fe = ue.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy')
fl = lambdify(x, fe, 'numpy')

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, plan=True, bc=(a, b))
X = SD.mesh(N)
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = fl(X)

# Compute right hand side of Poisson equation
f_hat = Array(SD)
f_hat = inner(v, fj, output_array=f_hat)
if family == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if family == 'chebyshev':
    A = inner(v, div(grad(u)))
else:
    A = inner(grad(v), grad(u))

f_hat = A.solve(f_hat)
uj = SD.backward(f_hat)

# Compare with analytical solution
ua = ul(X)
print("Error=%2.16e" %(np.linalg.norm(uj-ua)))
#assert np.allclose(uj, ua)
