import pytest
import numpy as np
from shenfun.chebyshev.bases import ShenDirichletBasis
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace, VectorTensorProductSpace
from shenfun import inner, div, grad, curl, TestFunction, TrialFunction, Function
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
# Set global size of the computational box
M = 4
N = [2**M, 2**M, 2**M]
L = np.array([2*np.pi, 4*np.pi, 4*np.pi], dtype=float) # Needs to be (2*int)*pi in all directions (periodic) because of initialization

@pytest.mark.parametrize('typecode', 'fd')
def test_curl(typecode):
    K0 = C2CBasis(N[0])
    K1 = C2CBasis(N[1])
    K2 = R2CBasis(N[2])
    T = TensorProductSpace(comm, (K0, K1, K2), dtype=typecode)
    X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
    K = T.local_wavenumbers(True)
    Tk = VectorTensorProductSpace([T]*3)
    u = TrialFunction(Tk)
    v = TestFunction(Tk)

    # Declare variables needed to solve Navier-Stokes
    U = Function(Tk, False)       # Velocity
    U_hat = Function(Tk)          # Velocity transformed
    curl_hat = Function(Tk)
    curl_ = Function(Tk, False)

    # Initialize a Taylor Green vortex
    U[0] = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])
    U[1] = -np.cos(X[0])*np.sin(X[1])*np.cos(X[2])
    U[2] = 0
    U_hat = Tk.forward(U, U_hat)
    Uc = U_hat.copy()
    U = Tk.backward(U_hat, U)
    U_hat = Tk.forward(U, U_hat)
    assert np.allclose(U_hat, Uc)

    curl_hat[0] = 1j*(K[1]*U_hat[2] - K[2]*U_hat[1])
    curl_hat[1] = 1j*(K[2]*U_hat[0] - K[0]*U_hat[2])
    curl_hat[2] = 1j*(K[0]*U_hat[1] - K[1]*U_hat[0])

    curl_ = Tk.backward(curl_hat, curl_)

    w = Function(Tk, False)
    w_hat = Function(Tk)
    t0 = time()
    w_hat = inner(v, curl(U), output_array=w_hat, uh_hat=U_hat)
    A = inner(v, u)
    for i in range(3):
        w_hat[i] = A[i].solve(w_hat[i])

    w = Tk.backward(w_hat, w)

    u_hat = Function(Tk)
    u_hat = inner(v, U, output_array=u_hat, uh_hat=U_hat)
    for i in range(3):
        u_hat[i] = A[i].solve(u_hat[i])

    uu = Function(Tk, False)
    uu = Tk.backward(u_hat, uu)

    assert np.allclose(u_hat, U_hat)
    assert np.allclose(w, curl_)


#def test_curl2():
    #K0 = ShenDirichletBasis(N[0])
    #K1 = C2CBasis(N[1])
    #K2 = R2CBasis(N[2])
    #T = TensorProductSpace(comm, (K0, K1, K2))
    #X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
    #K = T.local_wavenumbers(True)
    #Tk = VectorTensorProductSpace([T]*3)
    #u = TrialFunction(Tk)
    #v = TestFunction(Tk)

    ## Declare variables needed to solve Navier-Stokes
    #U = Function(Tk, False)       # Velocity
    #U_hat = Function(Tk)          # Velocity transformed
    #curl_hat = Function(Tk)
    #curl_ = Function(Tk, False)

    ## Initialize a Taylor Green vortex
    #U[0] = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])*(1-X[0]**2)
    #U[1] = -np.cos(X[0])*np.sin(X[1])*np.cos(X[2])*(1-X[0]**2)
    #U[2] = 0
    #U_hat = Tk.forward(U, U_hat)
    #Uc = U_hat.copy()
    #U = Tk.backward(U_hat, U)
    #U_hat = Tk.forward(U, U_hat)
    #assert np.allclose(U_hat, Uc)

    ##curl_hat[0] = 1j*(K[1]*U_hat[2] - K[2]*U_hat[1])
    ##curl_hat[1] = 1j*(K[2]*U_hat[0] - K[0]*U_hat[2])
    ##curl_hat[2] = 1j*(K[0]*U_hat[1] - K[1]*U_hat[0])

    ##curl_ = Tk.backward(curl_hat, curl_)

    #w = Function(Tk, False)
    #f_hat = Function(Tk)
    #w_hat = Function(Tk)
    #f_hat = inner(v, curl(U), output_array=f_hat, uh_hat=U_hat)
    #A = inner(v, u)

    #for i in range(3):
        #w_hat[i] = A[i].solve(f_hat[i], w_hat[i], axis=0)

    #w = Tk.backward(w_hat, w)

    #u_hat = Function(Tk)
    #b_hat = Function(Tk)
    #b_hat = inner(v, U, output_array=b_hat, uh_hat=U_hat)
    #for i in range(3):
        #u_hat[i] = A[i].solve(b_hat[i], u_hat[i])

    #uu = Function(Tk, False)
    #uu = Tk.backward(u_hat, uu)

    #assert np.allclose(u_hat, U_hat)
    #assert np.allclose(w, curl_)


if __name__ == '__main__':
    test_curl2()
