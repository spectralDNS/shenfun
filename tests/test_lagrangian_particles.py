import sympy as sp
from mpi4py import MPI
import pytest
from shenfun import *

comm = MPI.COMM_WORLD

def test_lagrangian_particles():
    N = (20, 20)
    F0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0., 1.))
    F1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0., 1.))
    T = TensorProductSpace(comm, (F0, F1))
    TV = VectorSpace(T)

    x, y = sp.symbols("x,y")
    psi = 1./np.pi*sp.sin(np.pi*x)**2*sp.sin(np.pi*y)**2 # Streamfunction
    ux = -psi.diff(y, 1)
    uy = psi.diff(x, 1)

    uxl = sp.lambdify((x, y), ux, 'numpy')
    uyl = sp.lambdify((x, y), uy, 'numpy')
    X = T.local_mesh(True)
    u = Array(T, buffer=uxl(X[0], X[1]))
    v = Array(T, buffer=uyl(X[0], X[1]))
    uv = Function(TV)
    uv[0] = T.forward(u, uv[0])
    uv[1] = T.forward(v, uv[1])

    points = np.array([[0.5], [0.75]])

    # Create LagrangianParticles instance with given points
    dt = 0.01
    lp = LagrangianParticles(points, dt, uv)
    for i in range(4):
        lp.step()

    assert np.allclose(lp.x, np.array([[0.53986228], [0.74811753]]), 1e-6)
    assert np.allclose(lp.up, np.array([[0.99115526], [-0.09409196]]), 1e-6)
    T.destroy()

if __name__ == '__main__':
    test_lagrangian_particles()
