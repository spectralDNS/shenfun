"""
Simple spectral Navier-Stokes solver

Not implemented for efficiency. For efficiency use the Navier-Stokes
solver in the https://github.com/spectralDNS/spectralDNS repository
"""
import numpy as np
from shenfun import *

nu = 0.000625
end_time = 0.1
dt = 0.01
N = (2**5, 2**5, 2**5)

V0 = FunctionSpace(N[0], 'F', dtype='D')
V1 = FunctionSpace(N[1], 'F', dtype='D')
V2 = FunctionSpace(N[2], 'F', dtype='d')
T = TensorProductSpace(comm, (V0, V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorSpace(T)
u = TrialFunction(T)
v = TestFunction(T)

U = Array(TV)
U_hat = Function(TV)
P_hat = Function(T)
curl_hat = Function(TV)
W = Array(TV)
curl_ = Array(TV)
A = inner(grad(u), grad(v))

def LinearRHS(self, u, **params):
    return nu*div(grad(u))

def NonlinearRHS(self, U, U_hat, dU, **params):
    global TV, curl_hat, curl_, P_hat, W
    curl_hat = project(curl(U_hat), TV, output_array=curl_hat)
    curl_ = TV.backward(curl_hat, curl_)
    U = U_hat.backward(U)
    W[:] = np.cross(U, curl_, axis=0)     # Nonlinear term in physical space
    dU = project(W, TV, output_array=dU)             # dU = W.forward(dU)
    P_hat = A.solve(inner(div(dU), v), P_hat)
    dU += inner(grad(P_hat), TestFunction(TV))
    return dU

if __name__ == '__main__':
    X = T.local_mesh(True)
    for i, integrator in enumerate((ETD, RK4, ETDRK4)):
        # Initialization
        U[0] = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])
        U[1] = -np.cos(X[0])*np.sin(X[1])*np.cos(X[2])
        U[2] = 0
        U_hat = TV.forward(U, U_hat)
        # Solve
        integ = integrator(TV, L=LinearRHS, N=NonlinearRHS)
        U_hat = integ.solve(U, U_hat, dt, (0, end_time))
        # Check accuracy
        U = U_hat.backward(U)
        k = comm.reduce(0.5*np.sum(U*U)/np.prod(np.array(N)))
        if comm.Get_rank() == 0:
            assert np.round(k - 0.124953117517, (4, 7, 7)[i]) == 0
