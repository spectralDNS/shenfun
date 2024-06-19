r"""
Solve Biharmonic equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet and Neumann in the remaining third

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import os
from sympy import symbols, cos, sin, diff
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    FunctionSpace, TensorProductSpace, Function, comm, la, chebyshev

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z", real=True)
#ue = (sin(4*np.pi*x)*sin(6*z)*cos(4*y))*(1-x**2)
ue = x*cos(4*np.pi*x)*sin(6*z)*cos(4*y)
fe = ue.diff(x, 4) + ue.diff(y, 4) + ue.diff(z, 4) + 2*ue.diff(x, 2, y, 2) + 2*ue.diff(x, 2, z, 2) + 2*ue.diff(y, 2, z, 2)

def main(N, family):

    SD = FunctionSpace(N, family=family, bc={'left': {'D': ue.subs(x, -1), 'N': diff(ue, x, 1).subs(x, -1)},
                                             'right': {'D': ue.subs(x, 1), 'N': diff(ue, x, 1).subs(x, 1)}})
    K1 = FunctionSpace(N, family='F', dtype='D')
    K2 = FunctionSpace(N, family='F', dtype='d')
    T = TensorProductSpace(comm, (SD, K1, K2), axes=(0, 1, 2))

    u = TrialFunction(T)
    v = TestFunction(T)

    # Get f on quad points
    fj = Array(T, buffer=fe)

    # Compute right hand side of biharmonic equation
    f_hat = inner(v, fj)

    # Get left hand side of biharmonic equation
    matrices = inner(v, div(grad(div(grad(u)))))

    # Create linear algebra solver
    BiharmonicSolver = chebyshev.la.Biharmonic if SD.family() == 'chebyshev' and N % 2 == 0 else la.SolverGeneric1ND
    H = BiharmonicSolver(matrices)

    # Solve and transform to real space
    u_hat = Function(T)             # Solution spectral space
    u_hat = H(f_hat, u_hat)         # Solve
    uq = u_hat.backward()
    #uh = uq.forward()

    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    if comm.Get_rank() == 0:
        print(f"biharmonic3D {SD.family():14s} L2 error = {error:2.6e}")

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
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
