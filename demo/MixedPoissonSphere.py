r"""Solve Poisson's equation on a sphere using a mixed formulation

The Poisson equation is in strong form

.. math::

    \nabla^2 u &= f \\
    u(x, y=\pm 1) &= 0 \\
    u(x=2\pi, y) &= u(x=0, y)

We solve using the mixed formulation

.. math::

    g - \nabla(u) &= 0 \\
    \nabla \cdot g &= f \\
    u(x, y=\pm 1) &= 0 \\
    u(x=2\pi, y) &= u(x=0, y) \\
    g(x=2\pi, y) &= g(x=0, y)

The problem is solved without boundary conditions and in spherical
coordinates. The mixed equations are solved coupled and implicit.

"""

import numpy as np
import sympy as sp
from shenfun import *

# Define spherical coordinates
r = 1
theta, phi = psi =sp.symbols('x,y', real=True, positive=True)
rv = (r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta))

# Define a manufactured solution
#ue = rv[0]*rv[1]*rv[2]
sph = sp.functions.special.spherical_harmonics.Ynm
ue = sph(6, 3, theta, phi)
#ue = sp.cos(4*(sp.sin(theta)*sp.cos(phi) + sp.sin(theta)*sp.sin(phi) + sp.cos(theta)))

N, M = 40, 40
L0 = FunctionSpace(N, 'L', domain=(0, np.pi))
F1 = FunctionSpace(M, 'F', dtype='D')
T = TensorProductSpace(comm, (L0, F1), coordinates=(psi, rv, sp.Q.positive(sp.sin(theta))))

VT = VectorSpace(T)
Q = CompositeSpace([VT, T])

gu = TrialFunction(Q)
pq = TestFunction(Q)

g, u = gu
p, q = pq

A00 = inner(p, g)
A01 = inner(div(p), u)
A10 = inner(q, div(g))

# Get f and g on quad points
gh = (div(grad(TrialFunction(T)))).tosympy(basis=ue, psi=psi)
vfj = Array(Q, buffer=(0, 0, gh))
vj, fj = vfj

vf_hat = Function(Q)
vf_hat[1] = inner(q, fj, output_array=vf_hat[1])

M = BlockMatrix(A00+A01+A10)

gu_hat = M.solve(vf_hat, constraints=((2, 0, 0),))
gu = gu_hat.backward()
g_, u_ = gu

# Exact Cartesian gradient
gradue = Array(VT, buffer=(ue.diff(theta, 1), ue.diff(phi, 1)/sp.sin(theta)**2))

uj = Array(T, buffer=ue)

error = [comm.reduce(np.linalg.norm(uj-u_)),
         comm.reduce(np.linalg.norm(gradue[0]-g_[0])),
         comm.reduce(np.linalg.norm(gradue[1]-g_[1]))]
if comm.Get_rank() == 0:
    print('Error    u         dudx        dudy')
    print('     %2.4e %2.4e %2.4e' %(error[0], error[1], error[2]))
    #assert np.all(abs(np.array(error)) < 1e-8), error

from mayavi import mlab
xx, yy, zz = T.local_cartesian_mesh(uniform=True)
gu = gu_hat.backward(kind='uniform')
g_, u_ = gu

# For plotting - get gradient on Cartesian
b = T.coors.get_covariant_basis()
ui, vi = T.local_mesh(True)
b1 = np.array(sp.lambdify(psi, b[0])(ui, vi))
b2 = sp.lambdify(psi, b[1])(ui, vi)
b2[2] = np.zeros(ui.shape) # b2[2] is 0, so need to broadcast for the next line to work
b2 = np.array(b2)
# Compute Cartesian gradient
df = g_[0]*b1 + g_[1]*b2

# Wrap periodic direction around
if T.bases[1].domain == (0, 2*np.pi):
    xx, yy, zz, u_ = wrap_periodic([xx, yy, zz, u_], axes=[1])
    df = wrap_periodic([df], axes=[2])

mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
mlab.mesh(xx, yy, zz, scalars=u_.real, colormap='jet')
#fig = surf3D(u_, backend='mayavi', wrapaxes=[1], kind='uniform')
mlab.quiver3d(xx, yy, zz, df[0].real, df[1].real, df[2].real,
              color=(0, 0, 0), scale_factor=0.2, mode='2darrow')

mlab.show()
