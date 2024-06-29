r"""Solve Stokes equations using a coupled formulation

The Stokes equations are in strong form

.. math::

    \nabla^2 u - \nabla p &= f \\
    \nabla \cdot u &= h \\
    u(x, y, z=\pm 1) &= 0

where :math:`f` and :math:`g` are given functions of space.
In addition we require :math:`\int p d\ = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0, 0} = 0`.

We use a tensorproductspace with Fourier expansions in the x- and
y-directions and a composite Chebyshev or Legendre basis in the z-direction
for ``u`` and a regular (no boundary conditions) Chebyshev or Legendre basis
for ``p``.

When both Fourier wavenumbers are zero the assembled coefficient matrix has
two nullspaces. One of these are removed by enforcing the global constraint
on the pressure. The second is removed by fixing :math:`\hat{p}_{0, 0, N-1} = 0`.

"""
import os
import sys
import numpy as np
from sympy import symbols, sin, cos
from shenfun import *

x, y, z = symbols("x,y,z", real=True)

# Some right hand side (manufactured solution)
uex = sin(2*y)*(1-z**2)
uey = sin(2*x)*(1-z**2)
uez = sin(2*z)*(1-z**2)
pe = -0.1*sin(2*x)*cos(4*y)
fx = uex.diff(x, 2) + uex.diff(y, 2) + uex.diff(z, 2) - pe.diff(x, 1)
fy = uey.diff(x, 2) + uey.diff(y, 2) + uey.diff(z, 2) - pe.diff(y, 1)
fz = uez.diff(x, 2) + uez.diff(y, 2) + uez.diff(z, 2) - pe.diff(z, 1)
h = uex.diff(x, 1) + uey.diff(y, 1) + uez.diff(z, 1)

N = (20, 20, 20)
family = sys.argv[-1] if len(sys.argv) == 2 else 'Legendre'
K0 = FunctionSpace(N[0], 'Fourier', dtype='D', domain=(0, 2*np.pi))
K1 = FunctionSpace(N[1], 'Fourier', dtype='d', domain=(0, 2*np.pi))
SD = FunctionSpace(N[2], family, bc=(0, 0))
ST = FunctionSpace(N[2], family)

TD = TensorProductSpace(comm, (K0, K1, SD), axes=(2, 0, 1))
Q = TensorProductSpace(comm, (K0, K1, ST), axes=(2, 0, 1))
V = VectorSpace(TD)
VQ = CompositeSpace([V, Q])

up = TrialFunction(VQ)
vq = TestFunction(VQ)

u, p = up
v, q = vq

# Assemble blocks of complete matrix
if family.lower() == 'chebyshev':
    A = inner(v, div(grad(u)))
    G = inner(v, -grad(p))
else:
    A = inner(grad(v), -grad(u))
    G = inner(div(v), p)
D = inner(q, div(u))

# Create block matrix
M = BlockMatrix(A+G+D)

# Get f and h on quad points
fh = Array(VQ, buffer=(fx, fy, fz, h))
f_, h_ = fh

fh_hat = Function(VQ)
f_hat, h_hat = fh_hat
f_hat = inner(v, f_, output_array=f_hat)
h_hat = inner(q, h_, output_array=h_hat)

# Solve problem using integral constraint on pressure
up_hat = M.solve(fh_hat, constraints=((3, 0, 0), (3, N[2]-1, 0)))
up = up_hat.backward()
u_, p_ = up

# Exact solution
ux, uy, uz = Array(V, buffer=(uex, uey, uez))
pe = Array(Q, buffer=pe)

error = [comm.reduce(np.linalg.norm(ux-u_[0])),
         comm.reduce(np.linalg.norm(uy-u_[1])),
         comm.reduce(np.linalg.norm(uz-u_[2])),
         comm.reduce(np.linalg.norm(pe-p_))]

if comm.Get_rank() == 0:
    print('Error    u          v          w        p')
    print('     %2.4e %2.4e %2.4e %2.4e' %(error[0], error[1], error[2], error[3]))
    assert np.all(abs(np.array(error)) < 1e-8), error

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = Q.local_mesh(True)
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], u_[2, :, :, 6], 100)
    plt.figure()
    plt.quiver(X[0][:, :, 0], X[1][:, :, 0], u_[0, :, :, 6], u_[1, :, :, 6])
    plt.figure()
    l, m = 2, 2
    plt.spy(M.diags((l, m), 'csr'), markersize=2, color='k') # The matrix for Fourier given wavenumber
    plt.title('Block matrix: l, m = ({}, {})'.format(l, m))
    plt.xticks([])
    plt.yticks([])
    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], u_[0, :, :, 6], 100)
    plt.show()
cleanup(vars())