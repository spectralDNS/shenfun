r"""Solve Stokes equations using a coupled formulation

The Stokes equations are in strong form

.. math::

    -\nabla^2 u - \nabla p &= f \\
    \nabla \cdot u &= h \\
    u(x, y=\pm 1) &= 0 \\
    u(x=2\pi, y) &= u(x=0, y) \\
    p(x=2\pi, y) &= p(x=0, y)

where :math:`f` and :math:`g` are given functions of space.
In addition we require :math:`\int p dx = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0} = 0`.

We use a tensorproductspace with Fourier expansions in the x-direction and
a composite Chebyshev or Legendre basis in the y-direction for ``u`` and
a regular (no boundary conditions) Chebyshev or Legendre basis for ``p``.

For the zeroth Fourier wavenumber the assembled coefficient matrix has
two nullspaces. One of these are removed by enforcing the global constraint
on the pressure. The second is removed by fixing :math:`\hat{p}_{0, N-1} = 0`.

"""
import os
import sys
import numpy as np
from sympy import symbols, sin
from shenfun import *

x, y = symbols("x,y", real=True)

# Some right hand side (manufactured solution)
uex = sin(2*y)*(1-y**2)
uey = sin(2*x)*(1-y**2)
pe = -0.1*sin(2*x)
fx = -uex.diff(x, 2) - uex.diff(y, 2) - pe.diff(x, 1)
fy = -uey.diff(x, 2) - uey.diff(y, 2) - pe.diff(y, 1)
h = uex.diff(x, 1) + uey.diff(y, 1)

N = (20, 20)
family = sys.argv[-1] if len(sys.argv) == 2 else 'Legendre'
K0 = FunctionSpace(N[0], 'Fourier', dtype='d', domain=(0, 2*np.pi))
SD = FunctionSpace(N[1], family, bc=(0, 0))
ST = FunctionSpace(N[1], family)

TD = TensorProductSpace(comm, (K0, SD), axes=(1, 0))
Q = TensorProductSpace(comm, (K0, ST), axes=(1, 0))
V = VectorSpace(TD)
VQ = CompositeSpace([V, Q])

up = TrialFunction(VQ)
vq = TestFunction(VQ)

u, p = up
v, q = vq

# Assemble blocks of complete matrix
if family.lower() == 'chebyshev':
    A00 = inner(v, -div(grad(u)))
    A01 = inner(v, -grad(p))

else:
    A00 = inner(grad(v), grad(u))
    A01 = inner(div(v), p)
A10 = inner(q, div(u))

# Create block matrix
sol = la.BlockMatrixSolver(A00+A01+A10)

# Get f and h on quad points
fh = Array(VQ, buffer=(fx, fy, h))
f_, h_ = fh

fh_hat = Function(VQ)
f_hat, h_hat = fh_hat
f_hat = inner(v, f_, output_array=f_hat)
h_hat = inner(q, h_, output_array=h_hat)
fh_hat.mask_nyquist()

# Solve problem using integral constraint on pressure
up_hat = sol(fh_hat, constraints=((2, 0, 0), (2, N[1]-1, 0)))
up_ = up_hat.backward()
u_, p_ = up_

# Exact solution
ux, uy = Array(V, buffer=(uex, uey))
pe = Array(Q, buffer=pe)

error = [comm.reduce(np.linalg.norm(ux-u_[0])),
         comm.reduce(np.linalg.norm(uy-u_[1])),
         comm.reduce(np.linalg.norm(pe-p_))]

if comm.Get_rank() == 0:
    print('Error    u          v          p')
    print('     %2.4e %2.4e %2.4e' %(error[0], error[1], error[2]))
    assert np.all(abs(np.array(error)) < 1e-8), error

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = TD.local_mesh(True)
    plt.contourf(X[0], X[1], p_, 100)
    plt.figure()
    plt.quiver(X[0], X[1], u_[0], u_[1])
    plt.figure()
    plt.spy(sol.mat.diags((0, 0))) # The matrix for Fourier given wavenumber
    plt.figure()
    plt.contourf(X[0], X[1], u_[0], 100)
    #plt.show()
