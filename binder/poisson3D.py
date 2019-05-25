from sympy import symbols, sin, cos, lambdify
from shenfun import *

# Use sympy to compute manufactured solution
x, y, z = symbols("x,y,z")
ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-x**2)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

C0 = Basis(32, 'Chebyshev', bc=(0, 0))
F1 = Basis(32, 'Fourier', dtype='D')
F2 = Basis(32, 'Fourier', dtype='d')
T = TensorProductSpace(comm, (C0, F1, F2))
u = TrialFunction(T)
v = TestFunction(T)

# Assemble left and right hand
f_hat = inner(v, Array(T, buffer=fe))
A = inner(v, div(grad(u)))

# Solve
solver = chebyshev.la.Helmholtz(*A)
u_hat = Function(T)
u_hat = solver(u_hat, f_hat)
uj = u_hat.backward()
assert np.linalg.norm(u_hat.backward()-Array(T, buffer=ue)) < 1e-12
print(u_hat.shape)