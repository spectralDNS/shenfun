r"""
Solve Kuramato-Kuramato_Sivashinsky equation on [-30pi, 30pi]^2
with periodic bcs

    u_t = -div(grad(u)) - div(grad(div(grad(u)))) - |grad(u)|^2    (1)

Initial condition is

    u(x, y) = exp(-0.01*(x^2 + y^2))

Use Fourier basis V and tensor product space VxV

"""
from sympy import symbols, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
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
count = 0
def update(u, u_hat, t, tstep, **params):
    global count
    if tstep % params['plot_step'] == 0 and params['plot_step'] > 0:
        u = T.backward(u_hat, u)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], U, 256, cmap=cm)
        plt.pause(1e-6)
        count += 1
        plt.savefig('Kuramato_Sivashinsky_N_{}_{}.png'.format(N[0], count))

if __name__ == '__main__':
    par = {'plot_step': 100}
    dt = 0.01
    end_time = 100
    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    #integrator = RK4(T, L=LinearRHS, N=NonlinearRHS, update=update, call_update=50)

    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time))
