r"""
Solve Kuramato-Kuramato_Sivashinsky equation on [-30pi, 30pi]^2
with periodic bcs

    u_t = -div(grad(u)) - div(grad(div(grad(u)))) - 0.5*|grad(u)|^2    (1)

Initial condition is

    u(x, y) = exp(-0.01*(x^2 + y^2))

Use Fourier basis V and tensor product space VxV

"""
import sys
from sympy import symbols, exp, pi, Rational
import numpy as np
import matplotlib.pyplot as plt
from shenfun import *

# Use sympy to set up initial condition
x, y = symbols("x,y", real=True)
ue = exp(-((x-30*pi)**2+(y-30*pi)**2)/100) - Rational(1, 36) / pi

# Size of discretization
N = (128, 128)

K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0, 60*np.pi))
K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0, 60*np.pi))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorSpace(T)
padding_factor = 1.5
Tp = T.get_dealiased()
TVp = VectorSpace(Tp)
gradu = Array(TVp)

u = TrialFunction(T)
v = TestFunction(T)

# Create solution and work arrays
U = Array(T, buffer=ue)
U_hat = Function(T)
gradu = Array(TVp)
K = np.array(T.local_wavenumbers(True, True, True))
mask = T.get_mask_nyquist()

def LinearRHS(self, u, **params):
    # Assemble diagonal bilinear forms
    return -div(grad(u))-div(grad(div(grad(u))))

def NonlinearRHS(self, U, U_hat, dU, gradu, **params):
    # Assemble nonlinear term
    gradu = TVp.backward(1j*K*U_hat, gradu)
    dU = Tp.forward(0.5*(gradu[0]*gradu[0]+gradu[1]*gradu[1]), dU)
    dU.mask_nyquist(mask)
    return -dU

#initialize
X = T.local_mesh(True)
U_hat = U.forward(U_hat)
U_hat.mask_nyquist(mask)

# Integrate using an exponential time integrator
def update(self, u, u_hat, t, tstep, plot_step, wash, **params):
    if not hasattr(self, 'fig'):
        self.fig = plt.figure()
        self.cm = plt.get_cmap('hot')
        self.image = plt.contourf(X[0], X[1], U, 256, cmap=self.cm)
    u_hat.mask_nyquist(mask)
    if tstep % wash == 0 and wash > 0:
        u = u_hat.backward(u)
        u_hat = u.forward(u_hat)
    
    if tstep % plot_step == 0 and plot_step > 0:
        u = u_hat.backward(u)    
        self.image.axes.contourf(X[0], X[1], u, 256, cmap=self.cm)
        plt.autoscale()
        plt.pause(1e-6)
        self.params['count'] += 1
        #plt.savefig('Kuramato_Sivashinsky_N_{}_{}.png'.format(N[0], self.params['count']))
        print(tstep, 'Energy =', dx(u**2), np.linalg.norm(u_hat - u_hat.backward().forward()))


if __name__ == '__main__':
    par = {'plot_step': 100,
           'wash': 1, 
           'gradu': gradu,
           'count': 0}
    dt = 0.01
    end_time = 100
    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    #integrator = RK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time))
    cleanup((T, Tp))