r"""
Solve Poisson equation in 1D with homogeneous Dirichlet bcs on the domain [0, inf)

    \nabla^2 u = f,

The equation to solve for a Laguerre basis is

     (\nabla u, \nabla v) = -(f, v)

"""
import os
import sys
from sympy import symbols, sin, exp, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, dx

assert len(sys.argv) == 2, 'Call with one command-line argument'
assert isinstance(int(sys.argv[-1]), int)

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x", real=True)
ue = sin(2*x)*exp(-x)
fe = ue.diff(x, 2)

# Size of discretization
N = int(sys.argv[-1])

SD = FunctionSpace(N, 'Laguerre', bc=(0, 0))
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(SD)
f_hat = inner(v, -fj, output_array=f_hat)

# Get left hand side of Poisson equation
#A = inner(v, -div(grad(u)))
A = inner(grad(v), grad(u))

f_hat = A.solve(f_hat)
uj = f_hat.backward()
uh = uj.forward()

# Compare with analytical solution
ua = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.linalg.norm(uj-ua)))
print("Error=%2.16e" %(np.sqrt(dx(uj-ua)**2)))
assert np.allclose(uj, ua, atol=1e-5)

point = np.array([0.1, 0.2])
p = SD.eval(point, f_hat)
assert np.allclose(p, lambdify(x, ue)(point), atol=1e-5)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    xx = np.linspace(0, 16, 100)
    plt.plot(xx, lambdify(x, ue)(xx), 'r', xx, uh.eval(xx), 'bo', markersize=2)
    plt.show()
