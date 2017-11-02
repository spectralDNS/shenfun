r"""
Solve Ginzburg-Landau equation on (-50, 50)x(-50, 50) with periodic bcs

    u_t = div(grad(u)) + u - (1+1.5i)*u*|u|**2         (1)

Use Fourier basis V and find u in VxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxV

The Fourier basis is span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxV is a tensor product space.

"""
from sympy import symbols, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import time
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import *

comm = MPI.COMM_WORLD

# Use sympy to set up initial condition
x, y = symbols("x,y")
ue = exp(-0.01*(x**2+y**2))        # + exp(-0.02*((x-15*np.pi)**2+(y)**2))
ul = lambdify((x, y), ue, 'numpy')

# Size of discretization
N = (128, 128)

K0 = C2CBasis(N[0], domain=(-30*np.pi, 30*np.pi))
K1 = R2CBasis(N[1], domain=(-30*np.pi, 30*np.pi))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorTensorProductSpace([T, T])

u = TrialFunction(T)
v = TestFunction(T)

# Create solution and work arrays
U = Array(T, False)
U_hat = Array(T)
gradu = Array(TV, False)
K = np.array(T.local_wavenumbers(True, True, True))

def LinearRHS():
    # Assemble diagonal bilinear forms
    A = inner(u, v)
    L = -(inner(div(grad(u)), v) + inner(div(grad(div(grad(u)))), v)) / A
    return L

def NonlinearRHS(U, U_hat, dU):
    # Assemble nonlinear term
    global gradu
    gradu = TV.backward(1j*K*U_hat, gradu)
    dU = T.forward(0.5*(gradu[0]*gradu[0]+gradu[1]*gradu[1]), dU)
    return dU

#initialize
X = T.local_mesh(True)
U[:] = ul(*X)
U_hat = T.forward(U, U_hat)

# Integrate using an exponential time integrator
plt.figure()
cm = plt.get_cmap('hot')
image = plt.contourf(X[0], X[1], U, 256, cmap=cm)
plt.draw()
plt.pause(1e-6)
def update(U, U_hat, t):
    U = T.backward(U_hat, U)
    image.ax.clear()
    image.ax.contourf(X[0], X[1], U, 256, cmap=cm)
    plt.pause(1e-6)
    #plt.savefig('KS/Kuramato_Sivashinsky_{}_{}.png'.format(N[0], int(np.round(t))))

dt = 0.01
end_time = 100
integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, call_update=50)
#integrator = RK4(T, L=LinearRHS, N=NonlinearRHS, update=update, call_update=50)

integrator.setup(dt)
t0 = time()
U_hat = integrator.solve(U, U_hat, dt, (0, end_time))
