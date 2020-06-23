r"""
Solve 6th order equation in 1D

    u(x)^(6) - a(x)u(x) = f(x), for x in [-1, 1]

where a(x) and f(x) are given. Homogeneous boundary conditions
u(\pm 1) = u'(\pm 1) = u''(\pm 1) = 0.

Use Shen's 6th order Jacobi basis.

"""
import sys
from sympy import symbols, sin, exp
import numpy as np
from scipy.sparse.linalg import spsolve
from shenfun import *

assert len(sys.argv) == 2
assert isinstance(int(sys.argv[-1]), int)

# Manufactured solution that satisfies boundary conditions
sol = 0
x = symbols("x")

if sol == 0:
    domain = (-1., 1.)
    d = 2./(domain[1]-domain[0])
    x_map = -1+(x-domain[0])*d
    ue = (1-x**2)**3*sin(np.pi*x)
    fe = ue.diff(x, 6) - ue

elif sol == 1:
    domain = (0, 1.)
    ue = x**3*(1-x)**3
    fe = ue.diff(x, 6) - exp(-x)*ue

# Size of discretization
N = int(sys.argv[-1])

SD = FunctionSpace(N, 'J', bc='6th order', domain=domain)
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

S = inner(Dx(v, 0, 3), -Dx(u, 0, 3))
B = inner(v, u)

if sol == 0:
    M = (S - B).diags('csr')
elif sol == 1:
    d = SparseMatrix({0: np.exp(-X)}, shape=(N-6, N-6))
    M = S.diags('csr') - d.diags('csr').dot(B.diags('csr'))

# Get f on quad points
fj = Array(SD, buffer=fe)
f_hat = inner(v, fj)

u_hat = Function(SD)
u_hat[:-6] = spsolve(M, f_hat[:-6])

uq = Array(SD, buffer=ue)
print(np.linalg.norm(u_hat.backward()-uq))
assert np.linalg.norm(u_hat.backward()-uq) < 1e-8

B0 = FunctionSpace(N, 'J', alpha=0, beta=0)
u_b = project(u_hat, B0)
