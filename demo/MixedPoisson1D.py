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
from sympy import symbols, cos
from shenfun import *
from shenfun.spectralbase import MixedFunctionSpace

x = symbols("x", real=True)

#ue = (sin(2*x)*cos(3*y))*(1-x**2)
ue = cos(5*x)*(1-x**2)
dx = ue.diff(x, 1)
fe = ue.diff(x, 2)

def main(N, family):
    SD = FunctionSpace(N, family, bc=(0, 0))
    ST = FunctionSpace(N, family)

    # Solve first regularly
    u = TrialFunction(SD)
    v = TestFunction(SD)
    A = inner(v, div(grad(u)))
    b = inner(v, Array(SD, buffer=fe))
    u_ = Function(SD)
    u_ = A.solve(b, u_)
    ua = Array(SD)
    ua = u_.backward(ua)

    # Now solved in mixed formulation
    Q = MixedFunctionSpace([ST, SD])

    gu = TrialFunction(Q)
    pq = TestFunction(Q)

    g, u = gu
    p, q = pq

    A00 = inner(p, g)
    A01 = inner(p, -grad(u))
    A10 = inner(q, div(g))

    # Get f and g on quad points
    vfj = Array(Q, buffer=(0, fe))
    vj, fj = vfj

    vf_hat = Function(Q)
    v_hat, f_hat = vf_hat
    f_hat = inner(q, fj, output_array=f_hat)

    M = BlockMatrix([A00, A01, A10])
    gu_hat = M.solve(vf_hat)
    gu = gu_hat.backward()

    uj = Array(SD, buffer=ue)
    dxj = Array(ST, buffer=dx)

    error = [np.sqrt(inner(1, (uj-gu[1])**2)),
             np.sqrt(inner(1, (dxj-gu[0])**2))]

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        X = SD.mesh(True)
        plt.plot(X, gu[1], X, uj)
        plt.figure()
        plt.spy(M.diags(0, 'csr').toarray())
        plt.show()
    else:
        if comm.Get_rank() == 0:
            print(SD.family())
            print('    L2 error      u       dudx')
            print('         %2.4e %2.4e' %(error[0], error[1]))
        assert np.all(abs(np.array(error)) < 1e-8), error
        assert np.allclose(gu[1], ua)

if __name__ == '__main__':
    for family in ('CULQJ'):
        main(48, family)
