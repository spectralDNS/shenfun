r"""
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
#config['basisvectors'] = 'covariant'

t = sp.Symbol('x', real=True, positive=True)
#rv = (t, t+sp.sin(t))
rv = (sp.sin(2*sp.pi*t), sp.cos(2*sp.pi*t), 2*t)

N = 50
L = FunctionSpace(N, 'L', bc=(0, 0), domain=(-1, 1), coordinates=((t,), rv))

u = TrialFunction(L)
v = TestFunction(L)

# Compute rhs for manufactured solution
ue = sp.sin(4*np.pi*t)
sg = L.coors.sg
#f = -1/sg*sp.diff(1/sg*ue.diff(t, 1), t, 1)
#or
f = (-div(grad(u))).tosympy(basis=ue, psi=(t,))

fj = Array(L, buffer=f*sg)
f_hat = inner(v, fj)

A = inner(v*sg, -div(grad(u)))

u_hat = Function(L)
sol = la.Solver(A)
u_hat = sol(f_hat, u_hat)

uj = u_hat.backward()
uq = Array(L, buffer=ue)
error = np.sqrt(inner(1, (uj-uq)**2))
print(f'curvilinear_poisson1D L2 error = {error:2.6e}')

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    uj = u_hat.backward(mesh='uniform')
    X = L.cartesian_mesh(kind='uniform')
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
else:
    assert error < 1e-6
