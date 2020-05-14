r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi, y) = u(0, y), u(x, 2pi) = u(x, 0)

Use Fourier basis and find u in VxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxV

where V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxV is a tensorproductspace.

"""
import os
from sympy import symbols, cos, sin, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, Array, Basis, \
    TensorProductSpace, Function, dx, comm

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y", real=True)
ue = cos(4*x) + sin(8*y)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Size of discretization
N = (64, 64)

K0 = Basis(N[0], family='F', dtype='D', domain=(-2*np.pi, 2*np.pi))
K1 = Basis(N[1], family='F', dtype='d', domain=(-2*np.pi, 2*np.pi))
T = TensorProductSpace(comm, (K0, K1), axes=(0, 1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Solve Poisson equation
u_hat = Function(T)
#A = inner(v, div(grad(u)))
A = inner(grad(v), grad(u))
u_hat = A.solve(-f_hat, u_hat)

uq = Array(T)
uq = T.backward(u_hat, uq, fast_transform=True)

uj = Array(T, buffer=ue)
assert np.allclose(uj, uq)

# Test eval at point
point = np.array([[0.1, 0.5], [0.5, 0.6]])
p = T.eval(point, u_hat)
ul = lambdify((x, y), ue)
assert np.allclose(p, ul(*point))
p2 = u_hat.eval(point)
assert np.allclose(p2, ul(*point))
print(np.sqrt(dx((uj-uq)**2)))

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0], X[1], uq)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uj)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uq-uj)
    plt.colorbar()
    plt.title('Error')
    plt.show()
