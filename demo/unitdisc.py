"""
Solve Helmholtz equation on the unit disc

Using polar coordinates and numerical method from:

"Efficient spectral-Galerkin methods III: Polar and cylindrical geometries",
J Shen, SIAM J. Sci Comput. 18, 6, 1583-1604

"""

import matplotlib.pyplot as plt
import functools
from shenfun import *
import sympy as sp

theta, r = sp.symbols('x,y')

alpha = 8
beta = alpha / 4

# Manufactured solution
#u = (1-r**2)**2*sp.cos(4*x)-0*(r-1)/2
u = (1-r**4)*sp.cos(4*theta)-0*(r-1)/2
g = -u.diff(r, 2) - (1/(r+1))*u.diff(r, 1) - (1/(r+1)**2)*u.diff(theta, 2) + beta*u

N = 12
F = Basis(N, 'F', dtype='d')
L = Basis(N, 'L', bc='Dirichlet')
L0 = Basis(N, 'L', bc='UpperDirichlet')
T = TensorProductSpace(comm, (F, L), axes=(1, 0))

# Compute the right hand side on the quadrature mesh
gj = Array(T, buffer=g*(1+r))

# Fourier transform rhs first, since we need to apply a different
# basis for one given Fourier wavenumber.
# g_tilde is intermediate, whereas g_hat full scalar product
g_tilde = Function(T)
g_hat = Function(T)
g_tilde = F.scalar_product(gj, g_tilde) # Fourier transform

# Take scalar product in Legendre direction
g_hat = L.scalar_product(g_tilde, g_hat)
g_hat[0] = 0

# Scalar product of rhs for m=0
M = g_hat.dims()
g0_hat = Function(L0)
g0_hat = L0.scalar_product(g_tilde[0].real, g0_hat)

# Assemble matrices. Note that sympy integrate is slow compared to quadrature,
# but integration is exact, so this is pure spectral Galerkin. For the entire
# method to be pure, we should also do the Legendre scalar product with sympy.

if L.family().lower() == 'legendre':
    A = inner_product((L, 1), (L, 1), (1+r))
    B = inner_product((L, 0), (L, 0), (1/(1+r)))
    C = inner_product((L, 0), (L, 0), (1+r))
    MC = A + beta*C

    # case m=0
    A0 = inner_product((L0, 1), (L0, 1), (1+r))
    C0 = inner_product((L0, 0), (L0, 0), (1+r))

else:
    A = inner_product((L, 0), (L, 2), (1+r))
    B = inner_product((L, 0), (L, 1))
    C = inner_product((L, 0), (L, 0), 1/(1+r))
    D = inner_product((L, 0), (L, 0), (1+r))
    MC = beta*D - A - B

    # case m=0
    A0 = inner_product((L0, 0), (L0, 2), (1+r))
    B0 = inner_product((L0, 0), (L0, 1))
    D0 = inner_product((L0, 0), (L0, 0), (1+r))

# Solve
# case m > 0
u_hat = Function(T)
for m in range(1, M[0]):
    if L.family().lower() == 'legendre':
        MM = MC + m**2*B
    else:
        MM = MC + m**2*C
    u_hat[m, :-2] = MM.solve(g_hat[m, :-2], u_hat[m, :-2])

# case m = 0
u0_hat = Function(L0)
if L.family().lower() == 'legendre':
    M0 = A0 + beta*C0
else:
    M0 = - A0 - B0 + beta*D0
u0_hat[:-1] = M0.solve(g0_hat[:-1], u0_hat[:-1])

# Transform back to real space.
uj = u_hat.backward() + u0_hat.backward()[None, :]
ue = Array(T, buffer=u)
X = T.local_mesh(True)
print('Error =', np.linalg.norm(uj-ue))

# Postprocess
# Refine for a nicer plot. Refine simply pads Functions with zeros, which
# gives more quadrature points. u_hat has NxN quadrature points, refine
# using any higher number.
u_hat2 = u_hat.refine([N*5, N*5])
u0_hat2 = u0_hat.refine(N*5)
ur = u_hat2.backward() + u0_hat2.backward()[None, :]
Y = u_hat2.function_space().local_mesh(True)
thetaj, tj = Y[0], Y[1]
rj = (tj+1)/2

# Wrap periodic plot around since it looks nicer
xx, yy = rj*np.cos(thetaj), rj*np.sin(thetaj)
xp = np.vstack([xx, xx[0]])
yp = np.vstack([yy, yy[0]])
up = np.vstack([ur, ur[0]])

# plot
plt.contourf(xp, yp, up)
plt.colorbar()
plt.show()
