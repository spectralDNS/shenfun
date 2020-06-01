"""
Solve Poisson's equation on parabolic domain

"""

import matplotlib.pyplot as plt
from shenfun import *
from shenfun.la import SolverGeneric2NP
import sympy as sp

# Define parabolic coordinates
tau, sigma = psi = sp.symbols('x,y', real=True)
rv = (tau*sigma, (tau**2-sigma**2)/2)

N = 40
by_parts = True
L0 = Basis(N, 'L', bc=(0, 0), domain=(0, 1))
L1 = Basis(N, 'L', bc=(0, 0), domain=(-1, 1))
T = TensorProductSpace(comm, (L0, L1), axes=(1, 0), coordinates=(psi, rv))

v = TestFunction(T)
u = TrialFunction(T)

# Manufactured solution
ue = (tau*(1-tau))**2*(1-sigma**2)**1*sp.sin(4*sp.pi*sigma)
#g = (ue.diff(tau, 2)+ue.diff(sigma, 2))/(tau**2+sigma**2)
g = (div(grad(u))).tosympy(basis=ue, psi=psi)

# Compute the right hand side on the quadrature mesh
gj = Array(T, buffer=g)

# Take scalar product
g_hat = Function(T)
g_hat = inner(v, gj, output_array=g_hat)

if by_parts:
    mats = inner(grad(v), -grad(u))

else:
    mats = inner(v, div(grad(u)))

# Solve
u_hat = Function(T)
Sol1 = SolverGeneric2NP(mats)
u_hat = Sol1(g_hat, u_hat)

# Transform back to real space.
uj = u_hat.backward()
uq = Array(T, buffer=ue)
print('Error =', np.linalg.norm(uj-uq))

# Postprocess
u_hat2 = u_hat.refine([N*2, N*2])
ur = u_hat2.backward()
xx, yy = u_hat2.function_space().local_curvilinear_mesh(True)

# plot
plt.figure()
plt.contourf(xx, yy, ur)
plt.colorbar()
plt.axis('equal')
plt.title("Poisson's equation - parabolic coordinates")
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()
