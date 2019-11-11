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
from shenfun import *

comm = MPI.COMM_WORLD

# Use sympy to set up initial condition
x, y = symbols("x,y")
ue = exp(-0.01*(x**2+y**2))        # + exp(-0.02*((x-15*np.pi)**2+(y)**2))
ul = lambdify((x, y), ue, 'numpy')

# Size of discretization
N = (128, 128)

K0 = Basis(N[0], 'F', dtype='D', domain=(-30*np.pi, 30*np.pi))
K1 = Basis(N[1], 'F', dtype='d', domain=(-30*np.pi, 30*np.pi))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorTensorProductSpace(T)

Tp = T.get_padded_space((1.5, 1.5))
TVp = VectorTensorProductSpace(Tp)

u = TrialFunction(T)
v = TestFunction(T)

#Tp = T
#TVp = TV

# Create solution and work arrays
U = Array(T)
U_hat = Function(T)
gradu = Array(TVp)
K = np.array(T.local_wavenumbers(True, True, True))

def LinearRHS(self, **params):
    # Assemble diagonal bilinear forms
    L = inner(-div(grad(u))-div(grad(div(grad(u)))), v)
    return L

def NonlinearRHS(self, U, U_hat, dU, gradu, **params):
    # Assemble nonlinear term
    gradu = TVp.backward(1j*K*U_hat, gradu)
    dU = Tp.forward(0.5*(gradu[0]*gradu[0]+gradu[1]*gradu[1]), dU)
    return -dU

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
def update(self, u, u_hat, t, tstep, plot_step, **params):
    if tstep % plot_step == 0 and plot_step > 0:
        u = u_hat.backward(u)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], U, 256, cmap=cm)
        plt.pause(1e-6)
        self.params['count'] += 1
        #plt.savefig('Kuramato_Sivashinsky_N_{}_{}.png'.format(N[0], self.params['count']))

if __name__ == '__main__':
    par = {'plot_step': 100,
           'gradu': gradu,
           'count': 0}
    dt = 0.01
    end_time = 500
    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    #integrator = RK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time))
