r"""Solve Poisson's equation using a mixed formulation

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

We use a Tensorproductspace with Fourier expansions in the x-direction and a
composite Chebyshev or Legendre basis in the y-direction for ``u``, whereas a
regular Chebyshev or Legendre basis is used for ``g``. The equations are solved
coupled and implicit.

"""

import os
import numpy as np
from sympy import symbols, sin, cos
from shenfun import *

x, y, z = symbols("x,y,z", real=True)

#ue = (sin(2*x)*cos(3*y))*(1-x**2)
ue = (sin(4*x)*cos(5*y)*sin(4*z))*(1-z**2)
dux = ue.diff(x, 1)
duy = ue.diff(y, 1)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

def main(N, family):
    K0 = FunctionSpace(N, 'Fourier', dtype='d')
    K1 = FunctionSpace(N, 'Fourier', dtype='D')
    SD = FunctionSpace(N, family, bc=(0, 0))
    ST = FunctionSpace(N, family)

    TD = TensorProductSpace(comm, (K0, K1, SD), axes=(2, 1, 0))
    TT = TensorProductSpace(comm, (K0, K1, ST), axes=(2, 1, 0))
    VT = VectorSpace(TT)
    Q = CompositeSpace([VT, TD])

    gu = TrialFunction(Q)
    pq = TestFunction(Q)

    g, u = gu
    p, q = pq

    A00 = inner(p, g)
    A01 = inner(p, -grad(u))
    A10 = inner(q, div(g))

    # Get f and g on quad points
    vfj = Array(Q, buffer=(0, 0, 0, fe))
    vj, fj = vfj

    vf_hat = Function(Q)
    vf_hat[1] = inner(q, fj, output_array=vf_hat[1])

    M = BlockMatrix(A00+A01+A10)

    gu_hat = M.solve(vf_hat)
    gu = gu_hat.backward()
    g_, u_ = gu

    uj = Array(TD, buffer=ue)
    duxj, duyj, duzj = Array(VT, buffer=(dux, duy, 0))

    error = [np.sqrt(inner(1, (uj-u_)**2)),
             np.sqrt(inner(1, (duxj-g_[0])**2)),
             np.sqrt(inner(1, (duyj-g_[1])**2))]

    if comm.Get_rank() == 0:
        print(SD.family())
        print('    L2 error      u       dudx       dudy')
        print('         %2.4e %2.4e %2.4e' %(error[0], error[1], error[2]))

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        X = TD.local_mesh(True)
        plt.contourf(X[0][:, :, 0], X[1][:, :, 0], u_[:, :, 0])
        plt.figure()
        plt.spy(M.diags((4, 4, 0)).toarray()) # The matrix for given Fourier wavenumber
        plt.show()

    else:
        assert np.all(abs(np.array(error)) < 1e-8), error

if __name__ == '__main__':
    for family in ('CLUQJ'):
        main(36, family)