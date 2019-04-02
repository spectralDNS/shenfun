import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

__all__ = ['LagrangianParticles']

class LagrangianParticles(object):
    """Class for tracking Lagrangian particles

    Parameters
    ----------
    points : array
        Initial location of particles. (D, N) array, with N particles in D dimensions
    dt : float
        Time step
    u_hat : :class:`.Function`
        Spectral Galerkin :class:`.Function` for the Eulerian velocity

    """

    def __init__(self, points, dt, u_hat):
        self.x = points
        self.u_hat = u_hat
        self.dt = dt
        self.up = np.zeros(self.x.shape)

    def step(self):
        up = self.rhs()
        self.x[:] = self.x + self.dt*up

    def rhs(self):
        return self.u_hat.eval(self.x, output_array=self.up)

if __name__ == '__main__':
    from shenfun import *
    import sympy as sp
    import matplotlib.pyplot as plt
    import h5py

    N = (40, 40)
    # Should work for any of these bases
    #F0 = Basis(N[0], 'F', dtype='D', domain=(0., 1.))
    #F1 = Basis(N[1], 'F', dtype='d', domain=(0., 1.))
    F0 = Basis(N[0], 'C', bc=(0, 0), domain=(0., 1.))
    F1 = Basis(N[1], 'C', bc=(0, 0), domain=(0., 1.))
    #F0 = Basis(N[0], 'C', domain=(0., 1.))
    #F1 = Basis(N[1], 'C', domain=(0., 1.))

    T = TensorProductSpace(comm, (F0, F1))
    TV = VectorTensorProductSpace(T)

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

    # Arrange particles in a circle around (0.5, 0.75) with radius 0.15
    t0 = np.linspace(0, 2*np.pi, 100)[:-1]
    points = np.array([0.5+0.15*np.cos(t0), 0.75+0.15*np.sin(t0)])

    # Create LagrangianParticles instance with given points
    dt = 0.001
    lp = LagrangianParticles(points, dt, uv)

    # Store velocity vectors for later plotting on rank 0
    u.write('velocity.h5', name='u', domain=T.mesh())
    v.write('velocity.h5', name='v', domain=T.mesh())
    if comm.Get_rank() == 0:
        f = h5py.File('velocity.h5', 'r+')
        f.create_group('points')

    # Run simulation from time = 0 to 1 forwards, and then integrate back to 0
    end_time = 2.0
    t = 0
    lg = ['Velocity field']
    b = 'Fwd'
    nsteps = int(end_time/dt)+1
    count = 0
    for i in range(nsteps):
        if np.any(np.round(t, 4) in (0, 0.5, 1.0)):
            if comm.Get_rank() == 0:
                f['points'].create_dataset(str(count), shape=points.shape, dtype=np.float)
                f['points/'+str(count)][:] = lp.x
            print('Storing points at time %2.1f'%t)
            lg.append('%s at %2.1f' %(b, t))
            count += 1
        if i == (nsteps-1)//2:
            lp.dt *= -1
            b = 'Bwd'
            print('Integrate backwards')
        t += lp.dt
        lp.step()

    if comm.Get_rank() == 0:
        plt.quiver(f['u/mesh/x0'], f['u/mesh/x1'], f['u/2D/0'].__array__().T, f['v/2D/0'].__array__().T)
        steps = list(f['points'].keys())
        for step in steps:
            plt.scatter(f['points/'+str(step)][0], f['points/'+str(step)][1])
        plt.title('Particles integrated forwards and backwards')
        plt.legend(lg)
        plt.show()
