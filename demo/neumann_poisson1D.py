r"""
Solve Poisson equation in 1D with homogeneous Neumann bcs

    \nabla^2 u = f

Use Shen's Neumann basis

The equation to solve for Legendre basis is

    -(\nabla u, \nabla v) = (f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun import inner_product
from shenfun.operators import div, grad

# Collect basis from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenNeumannBasis

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x")
u = sin(np.pi*x)*(1-x**2)
f = u.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, u, 'numpy')
fl = lambdify(x, f, 'numpy')

# Size of discretization
N = 32

SD = Basis(N, plan=True)
X = SD.mesh(N)

# Get f on quad points
fj = fl(X)

# Compute right hand side of Poisson equation
f_hat = np.zeros(N)
f_hat = SD.scalar_product(fj, f_hat)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
v = SD.test_function()
if basis == 'chebyshev':
    A = inner_product(v, div(grad(v)))
else:
    A = inner_product(grad(v), grad(v))

f_hat = A.solve(f_hat)

# Solve and transform to real space
u = np.zeros(N)               # Solution real space
u = SD.backward(f_hat, u)

# Compare with analytical solution
uj = ul(X)
print(abs(uj-u).max())
assert np.allclose(uj, u)

plt.figure()
plt.plot(X, u)

plt.figure()
plt.plot(X, uj)

plt.figure()
plt.plot(X, u-uj)
plt.title('Error')
#plt.show()
