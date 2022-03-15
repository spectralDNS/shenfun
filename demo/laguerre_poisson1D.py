r"""
Solve Poisson equation in 1D with homogeneous Dirichlet bcs on the domain [0, inf)

    \nabla^2 u = f,

The equation to solve for a Laguerre basis is

     (\nabla u, \nabla v) = -(f, v)

"""
import os
import sys
from sympy import symbols, sin, cos, exp, lambdify, diff
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, dx, la

assert len(sys.argv) == 3, 'Call with two command-line arguments'
assert isinstance(int(sys.argv[-2]), int)
assert sys.argv[-1].lower() in ('dirichlet', 'neumann')

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x", real=True)
ue = (sin(2*x)+cos(2*x))*exp(-x)
fe = -ue.diff(x, 2)+ue

# Size of discretization
N = int(sys.argv[-2])

if sys.argv[-1].lower() == 'dirichlet':
    SD = FunctionSpace(N, 'Laguerre', bc={'left': {'D': ue.subs(x, 0)}})
elif sys.argv[-1].lower() == 'neumann':
    SD = FunctionSpace(N, 'Laguerre', bc={'left': {'N': diff(ue, x, 1).subs(x, 0)}})

u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(SD)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
A = inner(v, -div(grad(u))+u)
sol = la.Solver(A)
u_hat = Function(SD)
u_hat = sol(f_hat, u_hat)
uj = u_hat.backward()

# Compare with analytical solution
ua = Array(SD, buffer=ue)
print("Error=%2.16e" %(np.linalg.norm(uj-ua)))
print("Error=%2.16e" %(np.sqrt(dx(uj-ua)**2)))
assert np.sqrt(dx(uj-ua)**2) < 1e-5

point = np.array([0.1, 0.2])
p = SD.eval(point, u_hat)
assert np.allclose(p, lambdify(x, ue)(point), atol=1e-5)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    xx = np.linspace(0, 16, 100)
    plt.plot(xx, lambdify(x, ue)(xx), 'r', xx, u_hat.eval(xx), 'bo', markersize=2)
    plt.show()
