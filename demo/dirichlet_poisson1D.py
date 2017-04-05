r"""
Solve Poisson equation in 1D with homogeneous Dirichlet bcs

    \nabla^2 u = f,

The equation to solve for Legendre basis is

     (\nabla u, \nabla v) = -(f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.inner import inner
from shenfun.arguments import TestFunction, TrialFunction
from shenfun.operators import div, grad

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x")
ue = sin(np.pi*x)*(1-x**2)
fe = ue.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy')
fl = lambdify(x, fe, 'numpy')

# Size of discretization
N = 32

SD = Basis(N, plan=True)
X = SD.mesh(N)
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = fl(X)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if basis == 'chebyshev':
    A = inner(v, div(grad(u)))
else:
    A = inner(grad(v), grad(u))

f_hat = A.solve(f_hat)

uj = SD.backward(f_hat)

# Compare with analytical solution
ue = ul(X)
print(abs(uj-ue).max())
assert np.allclose(uj, ue)

plt.figure()
plt.plot(X, uj)

plt.figure()
plt.plot(X, ue)

plt.figure()
plt.plot(X, uj-ue)
plt.title('Error')
#plt.show()
