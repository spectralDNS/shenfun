r"""
Solve Poisson equation in 1D with possibly inhomogeneous Dirichlet bcs

    \nabla^2 u = f,

The equation to solve for a Legendre basis is

     (\nabla u, \nabla v) = -(f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys
from sympy import symbols, sin, lambdify
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, Basis

assert len(sys.argv) == 3, 'Call with two command-line arguments'
assert sys.argv[-1] in ('legendre', 'chebyshev', 'jacobi')
assert isinstance(int(sys.argv[-2]), int)

# Get family from args
family = sys.argv[-1].lower()

# Use sympy to compute a rhs, given an analytical solution
domain = (-1., 1.)
a = 1.
b = -1.
if family == 'jacobi':
    a = 0
    b = 0

x = symbols("x")
d = 2./(domain[1]-domain[0])
x_map = -1+(x-domain[0])*d
ue = sin(4*np.pi*x_map)*(x_map-1)*(x_map+1) + a*(1+x_map)/2. + b*(1-x_map)/2.
fe = ue.diff(x, 2)

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, bc=(a, b), domain=domain, scaled=False)
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(SD)
f_hat = inner(v, fj, output_array=f_hat)
if family in ('legendre', 'jacobi'):
    f_hat *= -1.

# Get left hand side of Poisson equation
if family == 'chebyshev':
    A = inner(v, div(grad(u)))
else:
    A = inner(grad(v), grad(u))

u_hat = Function(SD)
u_hat = A.solve(f_hat, u_hat)
uj = u_hat.backward()
uh = uj.forward()

# Compare with analytical solution
ua = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.linalg.norm(uj-ua)))
assert np.allclose(uj, ua)

point = np.array([0.1, 0.2])
p = SD.eval(point, u_hat)
assert np.allclose(p, lambdify(x, ue)(point))
