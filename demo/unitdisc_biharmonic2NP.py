"""
Solve Biharmonic equation on the unit disc

Using polar coordinates and numerical method from:

"Efficient spectral-Galerkin methods III: Polar and cylindrical geometries",
J Shen, SIAM J. Sci Comput. 18, 6, 1583-1604

"""

import matplotlib.pyplot as plt
from shenfun import *
from shenfun.la import SolverGeneric2NP
import sympy as sp

# Define polar coordinates using angle along first axis and radius second
theta, r = psi = sp.symbols('x,y', real=True, positive=True)
rv = (r*sp.cos(theta), r*sp.sin(theta))

N = 20
by_parts = False
L0 = Basis(N, 'L', bc='Biharmonic', domain=(0, np.pi/2))
L1 = Basis(N, 'L', bc='Biharmonic', domain=(0.5, 1))
T = TensorProductSpace(comm, (L0, L1), axes=(1, 0), coordinates=(psi, rv))

# Manufactured solution
ue = ((0.5-r)*(1-r))**2*((sp.pi/2-theta)*theta)**2

# Right hand side of numerical solution
g = (ue.diff(r, 4)
     + (2/r**2)*ue.diff(r, 2, theta, 2)
     + 1/r**4*ue.diff(theta, 4)
     + (2/r)*ue.diff(r, 3)
     - 2/r**3*ue.diff(r, 1, theta, 2)
     - 1/r**2*ue.diff(r, 2)
     + 4/r**4*ue.diff(theta, 2)
     + 1/r**3*ue.diff(r, 1))

v = TestFunction(T)
u = TrialFunction(T)

# Compute the right hand side on the quadrature mesh
gj = Array(T, buffer=g)

# Take scalar product
g_hat = Function(T)
g_hat = inner(v, gj, output_array=g_hat)

if by_parts:
    mats = inner(div(grad(v)), div(grad(u)))

else:
    mats = inner(v, div(grad(div(grad(u)))))

# Solve
u_hat = Function(T)
Sol1 = SolverGeneric2NP(mats)
u_hat = Sol1(g_hat, u_hat)

# Transform back to real space.
uj = u_hat.backward()
uq = Array(T, buffer=ue)
X = T.local_mesh(True)
print('Error =', np.linalg.norm(uj-uq))

theta0, r0 = X[0], X[1]
x0, y0 = r0*np.cos(theta0), r0*np.sin(theta0)
plt.contourf(x0, y0, uq)
plt.colorbar()

# Postprocess
# Refine for a nicer plot. Refine simply pads Functions with zeros, which
# gives more quadrature points. u_hat has NxN quadrature points, refine
# using any higher number.
u_hat2 = u_hat.refine([N*2, N*2])
ur = u_hat2.backward()
Y = u_hat2.function_space().local_mesh(True)
thetaj, rj = Y[0], Y[1]
xx, yy = rj*np.cos(thetaj), rj*np.sin(thetaj)

# plot
plt.figure()
plt.contourf(xx, yy, ur)
plt.colorbar()
plt.axis('equal')
plt.title('Biharmonic equation - polar coordinates')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()
