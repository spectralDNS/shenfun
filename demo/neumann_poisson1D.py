r"""
Solve Poisson equation in 1D with homogeneous Neumann bcs

    \nabla^2 u = f

Use Shen's Neumann basis

The equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import importlib
from sympy import symbols, sin
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Basis, \
    Array, Function

# Collect basis from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', family)))

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x", real=True)
ue = sin(np.pi*x)*(1-x**2)
fe = ue.diff(x, 2)

# Size of discretization
N = 32

SD = Basis(N, family=family, bc='Neumann')
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(SD, buffer=inner(v, fj))

# Get left hand side of Poisson equation
A = inner(v, div(grad(u)))

f_hat = A.solve(f_hat)

# Solve and transform to real space
u = np.zeros(N)               # Solution real space
u = SD.backward(f_hat, u)

# Compare with analytical solution
uj = Array(SD, buffer=ue)
print(abs(uj-u).max())
assert np.allclose(uj, u)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X, u)

    plt.figure()
    plt.plot(X, uj)

    plt.figure()
    plt.plot(X, u-uj)
    plt.title('Error')
    #plt.show()
