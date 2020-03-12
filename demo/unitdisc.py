"""
Solve Helmholtz equation on the unit disc

Using polar coordinates and numerical method from:

"Efficient spectral-Galerkin methods III: Polar and cylindrical geometries", 
J Shen, SIAM J. Sci Comput. 18, 6, 1583-1604

"""

import matplotlib.pyplot as plt
from shenfun import *
import sympy as sp 

x, r = sp.symbols('x,r')

beta = 1

# Manufactured solution
u = (1-r**4)*sp.cos(4*x)-(r-1)/2
g = -u.diff(r, 2) - (1/(r+1))*u.diff(r, 1) - (1/(r+1)**2)*u.diff(x, 2) + beta*u

N = 16
F = Basis(N, 'F', dtype='d')
L = Basis(N, 'L', bc='Dirichlet')
L0 = Basis(N, 'L')
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

# case m=0 has different radial basis (see SIAM paper referenced above), 
# but the Fourier basis is unity (exp(0))
P = lambda n: sp.legendre(n, r) - sp.legendre(n+1, r)

# Scalar product of rhs for m=0, simply overwrite
M = g_hat.dims()
xj, wj = L.points_and_weights()
for j in range(M[1]+1):
    hj = sp.lambdify(r, P(j))(xj) # Testfunction on quadrature mesh
    g_hat[0, j] = np.sum(g_tilde[0]*hj*wj)

# Assemble matrices. Note that sympy integrate is slow compared to quadrature,
# but integration is exact, so this is pure spectral Galerkin.
A = np.zeros((M[1], M[1]))
B = np.zeros((M[1], M[1]))
C = np.zeros((M[1], M[1]))
for i in range(M[1]):
    psi_i = L.sympy_basis(i).subs({'x': r})
    for j in range(M[1]):
        psi_j =  L.sympy_basis(j).subs({'x': r})
        A[i, j] = sp.integrate((1+r)*psi_i.diff(r, 1)*psi_j.diff(r, 1), (r, -1, 1))
        B[i, j] = sp.integrate(1/(1+r)*psi_i*psi_j, (r, -1, 1)) 
        C[i, j] = sp.integrate((1+r)*psi_i*psi_j, (r, -1, 1))  

A = extract_diagonal_matrix(A)
B = extract_diagonal_matrix(B)
C = extract_diagonal_matrix(C)

#case m=0
A0 = np.zeros((N-1, N-1))
C0 = np.zeros((N-1, N-1))
for i in range(N-1):
    psi_i = P(i)
    for j in range(N-1):
        psi_j =  P(j)
        A0[i, j] = sp.integrate((1+r)*psi_i.diff(r, 1)*psi_j.diff(r, 1), (r, -1, 1))
        C0[i, j] = sp.integrate((1+r)*psi_i*psi_j, (r, -1, 1))

A0 = extract_diagonal_matrix(A0)
C0 = extract_diagonal_matrix(C0)

# Solve
# case m > 0
u_hat = Function(T)
for m in range(1, M[0]):
    MM = A + m**2*B + beta*C
    u_hat[m, :-2] = MM.solve(g_hat[m, :-2], u_hat[m, :-2])

# case m = 0
u0_hat = Function(L0) 
M0 = A0 + beta*C0
u0_hat[:-1] = M0.solve(g_hat[0, :-1].real, u0_hat[:-1])

# Transform back to real space. There's no shenfun basis for the 
# m=0 basis, so do backward manually here
uj = u_hat.backward()
w0_hat = u0_hat.copy()
w0_hat[1:] -= u0_hat[:-1]
w0 = w0_hat.backward()
uj += w0[None, :]

ue = Array(T, buffer=u)
X = T.local_mesh(True) 
print('Error =', np.linalg.norm(uj-ue))

# Postprocess
# Refine for a nicer plot. Refine simply pads Functions with zeros, which
# gives more quadrature points. u_hat has NxN quadrature points, refine 
# using any higher number.
u_hat2 = u_hat.refine([N*5, N*5])
w0_hat2 = w0_hat.refine(N*5)
ur = u_hat2.backward()
wr = w0_hat2.backward()
ur += wr[None, :]
Y = u_hat2.function_space().local_mesh(True)
theta, t = Y[0], Y[1]
r = (t+1)/2

# Wrap periodic plot around since it looks nicer
xx, yy = r*np.cos(theta), r*np.sin(theta)
xp = np.vstack([xx, xx[0]])
yp = np.vstack([yy, yy[0]])
up = np.vstack([ur, ur[0]])

# plot
plt.contourf(xp, yp, up)
plt.colorbar()
plt.show()
