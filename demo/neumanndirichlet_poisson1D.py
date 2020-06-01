r"""
Solve Poisson equation in 1D with mixed Dirichlet and Neumann bcs

    \nabla^2 u = f,

The equation to solve is

     (\nabla^2 u, v) = (f, v)

u'(-1) = 0 and u(1) = 0

"""
import os
import sys
from sympy import symbols, cos, pi
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, Basis, dx

assert len(sys.argv) == 3, 'Call with two command-line arguments'
assert sys.argv[-1] in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)

# Get family from args
family = sys.argv[-1].lower()

# Use sympy to compute a rhs, given an analytical solution
domain = (-1., 1.)

x = symbols("x", real=True)
d = 2./(domain[1]-domain[0])
x_map = -1+(x-domain[0])*d
ue = 1+cos(5*pi*(x_map+1)/2)
fe = ue.diff(x, 2)

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, bc='NeumannDirichlet', domain=domain)
X = SD.mesh()
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
uh = uj.forward()

# Compare with analytical solution
ua = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.sqrt(dx((uj-ua)**2))))
assert np.allclose(uj, ua)
if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.plot(SD.mesh(), uj, 'b', SD.mesh(), ua, 'r')
    plt.show()
