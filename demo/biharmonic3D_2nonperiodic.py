r"""
Solve Biharmonic equation in 3D with homogeneous Dirichlet and
Neumann boundary conditions in two directions and Fourier in the
last

    \nabla^4 u = f,

Use Shen's Biharmonic basis for both non-periodic directions.


"""
import sys
import os
from sympy import symbols, cos, sin
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, FunctionSpace, comm
from shenfun.la import SolverGeneric2ND

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = (sin(2*np.pi*z)*sin(4*np.pi*y)*cos(4*x))*(1-y**2)*(1-z**2)
fe = ue.diff(x, 4) + ue.diff(y, 4) + ue.diff(z, 4) + 2*ue.diff(x, 2, y, 2) + 2*ue.diff(x, 2, z, 2) + 2*ue.diff(y, 2, z, 2)

def main(N, family):

    K0 = FunctionSpace(N, 'Fourier', dtype='d')
    S0 = FunctionSpace(N, family=family, bc=(0, 0, 0, 0))
    S1 = FunctionSpace(N, family=family, bc=(0, 0, 0, 0))
    T = TensorProductSpace(comm, (K0, S0, S1), axes=(1, 0, 2), slab=True)

    u = TrialFunction(T)
    v = TestFunction(T)

    # Get f on quad points
    fj = Array(T, buffer=fe)

    # Compute right hand side of biharmonic equation
    f_hat = inner(v, fj)

    # Get left hand side of biharmonic equation
    matrices = inner(v, div(grad(div(grad(u)))))

    # Create linear algebra solver
    H = SolverGeneric2ND(matrices)

    # Solve and transform to real space
    u_hat = Function(T)           # Solution spectral space
    u_hat = H(f_hat, u_hat)       # Solve
    uq = u_hat.backward()

    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    if comm.Get_rank() == 0:
        print(f"biharmonic3D_2nonperiodic {S0.family():14s} L2 error = {error:2.6e}")

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        X = T.local_mesh(True)
        plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uq[:, :, 8])
        plt.colorbar()

        plt.figure()
        plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uj[:, :, 8])
        plt.colorbar()

        plt.figure()
        plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uq[:, :, 8]-uj[:, :, 8])
        plt.colorbar()
        plt.title('Error')
        plt.show()

    else:
        assert error < 1e-6
    
    T.destroy()

if __name__ == '__main__':
    for family in ('legendre', 'chebyshev', 'chebyshevu', 'ultraspherical', 'jacobi'):
        main(30, family)
