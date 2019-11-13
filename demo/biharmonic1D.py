r"""
Solve biharmonic equation in 1D

    u'''' + a*u'' + b*u = f,

Use Shen's Biharmonic basis.

"""
import sys
import os
import importlib
from sympy import symbols, sin, lambdify
import numpy as np
from shenfun import inner, Dx, TestFunction, TrialFunction, Basis, Array, \
    Function

assert len(sys.argv) == 3
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi')
assert isinstance(int(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1]
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Biharmonic

# Use sympy to compute a rhs, given an analytical solution
# Allow for a non-standard domain. Reference domain is (-1, 1)
domain = (-2., 1.)
d = 2./(domain[1]-domain[0])
x = symbols("x")
x_map = -1+(x-domain[0])*d
# Manufactured solution that satisfies (u(\pm 1) = u'(\pm 1) = 0)
ue = sin(4*np.pi*x_map)*(x_map-1)*(x_map+1)

# Use coefficients typical for Navier-Stokes solver for channel (https://github.com/spectralDNS/spectralDNS/blob/master/spectralDNS/solvers/KMM.py)
k = 8
nu = 1./590.
dt = 5e-5
c = -(k**2+nu*dt/2*k**4)
b = 1.0+nu*dt*k**2
a = -nu*dt/2.
fe = a*ue.diff(x, 4) + b*ue.diff(x, 2) + c*ue

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, bc='Biharmonic', domain=domain)
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation (no integration by parts)
S = inner(v, a*Dx(u, 0, 4))
A = inner(v, b*Dx(u, 0, 2))
B = inner(v, c*u)

# Create linear algebra solver
H = Solver(S, A, B)

# Solve and transform to real space
u_hat = Function(SD)          # Solution spectral space
u_hat = H(u_hat, f_hat)

uj = Array(SD)
uj = SD.backward(u_hat, uj)
uh = uj.forward()

# Compare with analytical solution
uq = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.linalg.norm(uj-uq)))
assert np.allclose(uj, uq)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X, uq)

    plt.figure()
    plt.plot(X, uj)

    plt.figure()
    plt.plot(X, uq-uj)
    plt.title('Error')
    #plt.show()
