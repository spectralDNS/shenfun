r"""
Solve Poisson equation on (0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi) = u(0)

Use Fourier basis and find u in V such that

    (v, div(grad(u))) = (v, f)    for all v in V

V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1}

"""
from sympy import Symbol, cos, sin, exp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction
from shenfun.fourier.bases import FourierBasis
import shenfun
import os
try:
    import matplotlib.pyplot as plt
except:
    plt = None

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
ue = cos(4*x)
fe = ue.diff(x, 2)

# Size of discretization
N = 32

ST = FourierBasis(N, np.float, plan=True, domain=(-np.pi, np.pi))
u = TrialFunction(ST)
v = TestFunction(ST)

X = ST.mesh(N)

# Get f on quad points and exact solution
fj = np.array([fe.subs(x, j) for j in X], dtype=np.float)
uj = np.array([ue.subs(x, i) for i in X], dtype=np.float)

# Compute right hand side
f_hat = inner(v, fj)

# Solve Poisson equation
A = inner(grad(v), grad(u))
u_hat = A.solve(-f_hat)

uq = ST.backward(u_hat)

assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.plot(X, uj)
    plt.title("U")
    plt.figure()
    plt.plot(X, uq - uj)
    plt.title("Error")
    plt.show()
