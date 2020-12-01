import sys
import os
from shenfun import *
import sympy as sp

domain = (-2, 1)
# Manufactured solution
x = sp.symbols("x", real=True)
ue = sp.cos(5*sp.pi*(x+0.1)/2)
fe = ue.diff(x, 4)

# Beam boundary conditions: u(-1), u'(-1), u''(1) and u'''(1)
bc = {'left': (('D', ue.subs(x, domain[0])), ('N', ue.diff(x, 1).subs(x, domain[0]))),
      'right': (('N2', ue.diff(x, 2).subs(x, domain[1])), ('N3', ue.diff(x, 3).subs(x, domain[1])))}

N = int(sys.argv[-1])
SB = FunctionSpace(N, 'L', bc=bc, domain=domain)
u = TrialFunction(SB)
v = TestFunction(SB)
A = inner(div(grad(div(grad(u)))), v)
f_hat = inner(v, Array(SB, buffer=fe))
u_hat = Function(SB).set_boundary_dofs()
u_hat = A.solve(f_hat, u_hat)
uj = u_hat.backward()
ua = Array(SB, buffer=ue)
assert np.sqrt(inner(1, (uj-ua)**2)) < 1e-8

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    print('Error =', np.sqrt(inner(1, (uj-ua)**2)))
    plt.plot(SB.mesh(), uj)
    plt.show()