r"""Solve Poisson's equation using a mixed formulation

The Poisson equation is

.. math::

    \nabla^2 u &= f \\
    u(x, y=\pm 1) &= 0

We solve using the mixed formulation

.. math::

    g - \nabla(u) &= 0 \\
    \nabla \cdot g &= f \\
    u(x, y=\pm 1) &= 0

We use a composite Chebyshev or Legendre basis. The equations are solved
coupled and implicit.

"""

import os
import numpy as np
from mpi4py import MPI
from sympy import symbols, cos, lambdify
from shenfun import *
from shenfun.spectralbase import MixedBasis

comm = MPI.COMM_WORLD
x = symbols("x")

#ue = (sin(2*x)*cos(3*y))*(1-x**2)
ue = cos(5*x)*(1-x**2)
dx = ue.diff(x, 1)
fe = ue.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy')
fl = lambdify(x, fe, 'numpy')
dxl = lambdify(x, dx, 'numpy')

N = 24
SD = Basis(N, 'L', bc=(0, 0))
ST = Basis(N, 'L')
X = SD.mesh(True)

# Solve first regularly
u = TrialFunction(SD)
v = TestFunction(SD)
A = inner(v, div(grad(u)))
b = inner(v, Array(SD, buffer=fl(X)))
u_ = Function(SD)
u_ = A.solve(b, u_)
ua = Array(SD)
ua = u_.backward(ua)

# Now solved in mixed formulation
Q = MixedBasis([ST, SD])

gu = TrialFunction(Q)
pq = TestFunction(Q)

g, u = gu
p, q = pq

A00 = inner(p, g)
A01 = inner(div(p), u)
#A01 = inner(p, -grad(u))
A10 = inner(q, div(g))

# Get f and g on quad points
fvj = Array(Q)
fj = fvj[1]
fj[:] = fl(X)

fv_hat = Function(Q)
fv_hat[1] = inner(q, fj)

M = BlockMatrix([A00, A01, A10])
gu_hat = M.solve(fv_hat)
gu = gu_hat.backward()

uj = ul(X)
dxj = dxl(X)

error = [comm.reduce(np.linalg.norm(uj-gu[1])),
         comm.reduce(np.linalg.norm(dxj-gu[0]))]

if comm.Get_rank() == 0:
    print('Error    u         dudx')
    print('     %2.4e %2.4e' %(error[0], error[1]))
    assert np.all(abs(np.array(error)) < 1e-8), error
    assert np.allclose(gu[1], ua)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X, gu[1])
    plt.figure()
    plt.spy(M.diags().toarray()) # The matrix for given Fourier wavenumber
    plt.show()
