import matplotlib.pyplot as plt
from shenfun import *
from ChannelFlow import KMM


class OrrSommerfeld(KMM):

    def __init__(self, N=(32, 32, 32), domain=((-1, 1), (0, 2*np.pi), (0, np.pi)), Re=8000.,
                 dt=0.1, conv=0, modplot=100, modsave=1e8, moderror=100, filename='KMM',
                 family='C', padding_factor=(1, 1.5, 1.5), checkpoint=1000, timestepper='IMEXRK3'):
        KMM.__init__(self, N=N, domain=domain, nu=1/Re, dt=dt, conv=conv, modplot=modplot,
                     modsave=modsave, moderror=moderror, filename=filename, family=family,
                     padding_factor=padding_factor, checkpoint=checkpoint, timestepper=timestepper,
                     dpdy=-2/Re)
        self.Re = Re

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        from OrrSommerfeld_eigs import OrrSommerfeld
        self.OS = OS = OrrSommerfeld(Re=self.Re, N=128)
        eigvals, eigvectors = OS.solve(False)
        OS.eigvals, OS.eigvectors = eigvals, eigvectors
        ub = Array(self.BD)
        self.initOS(OS, eigvals, eigvectors, ub)
        self.u_ = ub.forward(self.u_)
        # Compute convection from data in context (i.e., context.U_hat and context.g)
        # This is the convection at t=0
        self.e0 = 0.5*dx(ub[0]**2+(ub[1]-(1-self.X[0]**2))**2)
        self.acc = np.zeros(1)
        return 0, 0

    def initOS(self, OS, eigvals, eigvectors, U, t=0.):
        X = self.X
        x = X[0][:, 0, 0].copy()
        eigval, phi, dphidy = OS.interp(x, eigvals, eigvectors, eigval=1, verbose=False)
        OS.eigval = eigval
        for j in range(U.shape[2]):
            y = X[1][0, j, 0]
            v = (1-x**2) + 1e-7*np.real(dphidy*np.exp(1j*(y-eigval*t)))
            u = -1e-7*np.real(1j*phi*np.exp(1j*(y-eigval*t)))
            U[0, :, j, :] = u.repeat(U.shape[3]).reshape((len(x), U.shape[3]))
            U[1, :, j, :] = v.repeat(U.shape[3]).reshape((len(x), U.shape[3]))
        U[2] = 0

    def compute_error(self, t):
        U = self.u_.backward()
        pert = (U[1] - (1-self.X[0]**2))**2 + U[0]**2
        e1 = 0.5*dx(pert)
        exact = np.exp(2*np.imag(self.OS.eigval)*t)
        U0 = self.work[(U, 0, True)]
        self.initOS(self.OS, self.OS.eigvals, self.OS.eigvectors, U0, t=t)
        pert = (U[0] - U0[0])**2 + (U[1] - U0[1])**2
        e2 = 0.5*dx(pert)
        return e1, e2, exact

    def init_plots(self):
        self.ub = ub = self.u_.backward()
        self.im1 = 1
        if comm.Get_rank() == 0 and comm.Get_size() == 1:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[0, :, :, 0], 100)
            plt.colorbar(self.im1)
            plt.draw()

            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[1, :, :, 0] - (1-self.X[0][:, :, 0]**2), 100)
            plt.colorbar(self.im2)
            plt.draw()

            plt.figure(3, figsize=(6, 3))
            self.im3 = plt.quiver(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[1, :, :, 0]-(1-self.X[0][:, :, 0]**2), ub[0, :, :, 0])
            plt.colorbar(self.im3)
            plt.draw()

    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            if comm.Get_rank() == 0 and comm.Get_size() == 1:
                ub = self.u_.backward(self.ub)
                X = self.X
                self.im1.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], ub[0, :, :, 0], 100)
                self.im1.autoscale()
                self.im2.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], ub[1, :, :, 0], 100)
                self.im2.autoscale()
                self.im3.set_UVC(ub[1, :, :, 0]-(1-self.X[0][:, :, 0]**2), ub[0, :, :, 0])
                plt.pause(1e-6)

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            divu = self.divu().backward()
            e3 = dx(divu*divu)
            e0 = self.e0
            e1, e2, exact = self.compute_error(t)
            self.acc[0] += abs(e1/e0-exact)*self.dt
            if comm.Get_rank() == 0:
                print("Time %2.5f Norms %2.12e %2.12e %2.12e %2.12e %2.12e" %(t, e1/e0, exact, e1/e0-exact, np.sqrt(e2), np.sqrt(e3)))

if __name__ == '__main__':
    from time import time
    from mpi4py_fft import generate_xdmf
    t0 = time()
    N = (128, 32, 4)
    config['optimization']['mode'] = 'cython'
    d = {
        'N': N,
        'Re': 8000.,
        'dt': 0.001,
        'filename': f'KMM_OS_{N[0]}_{N[1]}_{N[2]}',
        'conv': 0,
        'modplot': 100,
        'modsave': 1000,
        'moderror': 10,
        'family': 'C',
        'checkpoint': 10000000,
        'padding_factor': 1,
        'timestepper': 'IMEXRK222'
        }
    OS = True
    c = OrrSommerfeld(**d)
    t, tstep = c.initialize(from_checkpoint=False)
    c.solve(t=t, tstep=tstep, end_time=0.1)
    print('Computing time %2.4f'%(time()-t0))
    p = c.compute_pressure()
    p0 = p.backward().get((slice(None), slice(None), 0))
    X = c.TD.mesh()
    if comm.Get_rank() == 0:
        plt.figure()
        plt.contourf(X[1][0, :, 0], X[0][:, 0, 0], p0)
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
    plt.show()
    cleanup(vars(c))
