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
import matplotlib.pyplot as plt
from shenfun import *
import sympy as sp

t = sp.Symbol('x', real=True, positive=True)
rv = (t, t+sp.sin(t))
#rv = (sp.sin(2*t), sp.cos(2*t), t)

N = 200
L = Basis(N, 'C', bc=(0, 0), domain=(0, 2*np.pi), coordinates=((t,), rv))

# Compute rhs for manufactured solution
ue = sp.sin(8*t)
g = L.coors.get_sqrt_g()
f = -1/g*sp.diff(1/g*ue.diff(t, 1), t, 1)

u = TrialFunction(L)
v = TestFunction(L)

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
uj = u_hat.backward(uniform=False)
X = L.curvilinear_mesh(uniform=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if len(rv) == 3:
    ax.plot(X[0], X[1], X[2], 'r')
    ax.plot(X[0], X[1], X[2]+uj, 'b')
elif len(rv) == 2:
   ax.plot(X[0], X[1], uj, 'b')
   ax.plot(X[0], X[1], 'r')

plt.show()

