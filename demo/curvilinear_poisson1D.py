"""
Solve Poisson's equation on a curved line in 2D or 3D space.

Define a position vector, `rv(t)`, as::

    rv(t) = x(t)i + y(t)j + z(t)k,

where i, j, k are the Cartesian unit vectors and t is found
in some suitable interval, e.g., [0, 1] or [0, 2\pi]. Note that
the position vector describes a, possibly curved, 1D domain.

Solve::

    -div(grad(u(t))) = f(t), for t in interval

using curvilinear coordinates.

"""
import os
from shenfun import *
import sympy as sp

t = sp.Symbol('x', real=True, positive=True)
#rv = (t, t+sp.sin(t))
rv = (sp.sin(2*t), sp.cos(2*t), 0.5*t)

N = 200
L = FunctionSpace(N, 'C', bc=(0, 0), domain=(0, 2*np.pi), coordinates=((t,), rv))

u = TrialFunction(L)
v = TestFunction(L)

# Compute rhs for manufactured solution
ue = sp.sin(8*t)
g = L.coors.get_sqrt_det_g()
f = -1/g*sp.diff(1/g*ue.diff(t, 1), t, 1)
#or
#f = (-div(grad(u))).tosympy(basis=ue, psi=(t,))

fj = Array(L, buffer=f)
f_hat = inner(v, fj)

A = inner(v, -div(grad(u)))

u_hat = Function(L)
if isinstance(A, list):
    A = np.sum(A)
    u_hat[:-2] = A.solve(f_hat[:-2], u_hat[:-2])
else:
    u_hat = A.solve(f_hat, u_hat)

uj = u_hat.backward()
uq = Array(L, buffer=ue)
print('Error = ', np.linalg.norm(uj-uq))
assert np.linalg.norm(uj-uq) < 1e-8
uj = u_hat.backward(uniform=True)
X = L.curvilinear_mesh(uniform=True)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    if len(rv) == 3:
        ax.plot(X[0], X[1], X[2], 'r')
        ax.plot(X[0], X[1], X[2]+uj, 'b')
        ax.set_xticks(np.linspace(-1, 1, 5))
        ax.set_yticks(np.linspace(-1, 1, 5))
        plt.title("Poisson's equation on a coil")
    elif len(rv) == 2:
        ax.plot(X[0], X[1], uj, 'b')
        ax.plot(X[0], X[1], 'r')

    plt.show()
