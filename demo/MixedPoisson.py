import os
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from sympy import symbols, sin, cos, lambdify
from shenfun import *

comm = MPI.COMM_WORLD
x, y = symbols("x,y")

#ue = (sin(2*x)*cos(3*y))*(1-x**2)
ue = (sin(4*x)*cos(5*y))*(1-y**2)
dux = ue.diff(x, 1)
duy = ue.diff(y, 1)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')
duxl = lambdify((x, y), dux, 'numpy')
duyl = lambdify((x, y), duy, 'numpy')

N = (36, 36)
K0 = Basis(N[0], 'F', dtype='d')
SD = Basis(N[1], 'C', bc=(0, 0))
ST = Basis(N[1], 'C')
TD = TensorProductSpace(comm, (K0, SD), axes=(1, 0))
TT = TensorProductSpace(comm, (K0, ST), axes=(1, 0))
VT = VectorTensorProductSpace(TT)
Q = MixedTensorProductSpace([VT, TD])
X = TD.local_mesh(True)

uv = TrialFunction(Q)
pq = TestFunction(Q)

g, u = uv[0], uv[1]
p, q = pq[0], pq[1]

A00 = inner(p, g)
A01 = inner(p, -grad(u))
A10 = inner(q, div(g))

# Get f and g on quad points
fvj = Array(Q)
fj = fvj[2]
fj[:] = fl(*X)

fv_hat = Function(Q)
fv_hat[2] = inner(q, fj)

M = BlockMatrix(A00+A01+A10)

gu_hat = M.solve(fv_hat)
gu = gu_hat.backward()

uj = ul(*X)
duxj = duxl(*X)
duyj = duyl(*X)

error = [comm.reduce(np.linalg.norm(uj-gu[2])),
         comm.reduce(np.linalg.norm(duxj-gu[0])),
         comm.reduce(np.linalg.norm(duyj-gu[1]))]

if comm.Get_rank() == 0:
    print('Error    u         dudx        dudy')
    print('     %2.4e %2.4e %2.4e' %(error[0], error[1], error[2]))
    assert np.all(abs(np.array(error)) < 1e-12)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0], X[1], gu[2])
    plt.figure()
    plt.quiver(X[1], X[0], gu[1], gu[0])
    #plt.show()
