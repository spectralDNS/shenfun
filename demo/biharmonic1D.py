r"""
Solve biharmonic equation in 1D

    u'''' + a*u'' + b*u = f,

Use Shen's Biharmonic basis.

"""
import sys
import os
from sympy import symbols, sin, chebyshevt
import numpy as np
from shenfun import inner, Dx, TestFunction, TrialFunction, FunctionSpace, Array, \
    Function, la, chebyshev

assert len(sys.argv) == 3
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi', 'chebyshev2')
assert isinstance(int(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower()
Solver = chebyshev.la.Biharmonic if family == 'chebyshev' else la.Solver

# Use sympy to compute a rhs, given an analytical solution
# Allow for a non-standard domain. Reference domain is (-1, 1)
domain = (-1., 1.)
d = 2./(domain[1]-domain[0])
x = symbols("x", real=True)
x_map = -1+(x-domain[0])*d
a = 1
b = -1
if family == 'jacobi':
    a = 0
    b = 0
# Manufactured solution that satisfies (u(\pm 1) = u'(\pm 1) = 0)
ue = sin(4*np.pi*x_map)*(x_map-1)*(x_map+1) + a*(0.5-9./16.*x_map+1./16.*chebyshevt(3, x_map)) + b*(0.5+9./16.*x_map-1./16.*chebyshevt(3, x_map))

# Use coefficients typical for Navier-Stokes solver for channel (https://github.com/spectralDNS/spectralDNS/blob/master/spectralDNS/solvers/KMM.py)
k = 8
nu = 1./590.
dt = 5e-5
cc = -(k**2+nu*dt/2*k**4)
bb = 1.0+nu*dt*k**2
aa = -nu*dt/2.
fe = aa*ue.diff(x, 4) + bb*ue.diff(x, 2) + cc*ue

# Size of discretization
N = int(sys.argv[-2])

SD = FunctionSpace(N, family=family, bc=(a, 0, b, 0), domain=domain)
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation (no integration by parts)
matrices = inner(v, aa*Dx(u, 0, 4) + bb*Dx(u, 0, 2) + cc*u)

# Function to hold the solution
u_hat = Function(SD)

# Create linear algebra solver
H = Solver(matrices)
u_hat = H(f_hat, u_hat)
#sol = la.Solve(matrices)
#u_hat = sol(f_hat, u_hat)

uj = u_hat.backward()
uh = uj.forward()

# Compare with analytical solution
uq = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.linalg.norm(uj-uq)))
assert np.linalg.norm(uj-uq) < 1e-8

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X, uq)

    plt.figure()
    plt.plot(X, uj)

    plt.figure()
    plt.plot(X, uq-uj)
    plt.title('Error')
    plt.show()
