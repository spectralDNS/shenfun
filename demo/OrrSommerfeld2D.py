import matplotlib.pyplot as plt
from shenfun import *
from ChannelFlow2D import KMM


class OrrSommerfeld(KMM):

    def __init__(self, N=(32, 32), domain=((-1, 1), (0, 2*np.pi)), Re=8000., alfa=1.,
                 dt=0.1, conv=0, modplot=100, modsave=1e8, moderror=100, filename='KMM',
                 family='C', padding_factor=(1, 1.5), checkpoint=1000, timestepper='IMEXRK3'):
        KMM.__init__(self, N=N, domain=(domain[0], (domain[1][0], domain[1][1]/alfa)), nu=1/Re, dt=dt, conv=conv, modplot=modplot,
                     modsave=modsave, moderror=moderror, filename=filename, family=family,
                     padding_factor=padding_factor, checkpoint=checkpoint, timestepper=timestepper,
                     dpdy=-2/Re)
        self.Re = Re
        self.alfa = alfa

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        from OrrSommerfeld_eigs import OrrSommerfeld
        self.OS = OS = OrrSommerfeld(Re=self.Re, N=128, alfa=self.alfa)
        eigvals, eigvectors = OS.solve(False)
        OS.eigvals, OS.eigvectors = eigvals, eigvectors
        self.initOS(OS, eigvals, eigvectors, self.ub)
        self.u_ = self.ub.forward(self.u_)
        self.e0 = 0.5*dx(self.ub[0]**2+(self.ub[1]-(1-self.X[0]**2))**2)
        self.acc = np.zeros(1)
        return 0, 0

    def initOS(self, OS, eigvals, eigvectors, U, t=0.):
        X = self.X
        x = X[0][:, 0].copy()
        eigval, phi, dphidy = OS.interp(x, eigvals, eigvectors, eigval=1, verbose=False)
        OS.eigval = eigval
        for j in range(U.shape[2]):
            y = X[1][0, j]
            v = (1-x**2) + 1e-7*np.real(dphidy*np.exp(1j*self.alfa*(y-eigval*t)))
            u = -1e-7*np.real(1j*self.alfa*phi*np.exp(1j*self.alfa*(y-eigval*t)))
            U[0, :, j] = u
            U[1, :, j] = v

    def compute_error(self, t):
        ub = self.u_.backward(self.ub)
        pert = (ub[1] - (1-self.X[0]**2))**2 + ub[0]**2
        e1 = 0.5*dx(pert)
        exact = np.exp(2*np.imag(self.alfa*self.OS.eigval)*t)
        U0 = self.work[(ub, 0, True)]
        self.initOS(self.OS, self.OS.eigvals, self.OS.eigvectors, U0, t=t)
        pert = (ub[0] - U0[0])**2 + (ub[1] - U0[1])**2
        e2 = 0.5*dx(pert)
        return e1, e2, exact

    def init_plots(self):
        ub = self.u_.backward(self.ub)
        self.im1 = 1
        if comm.Get_rank() == 0 and comm.Get_size() == 1:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.contourf(self.X[1], self.X[0], ub[0], 100)
            plt.colorbar(self.im1)
            plt.draw()

            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1], self.X[0], ub[1] - (1-self.X[0]**2), 100)
            plt.colorbar(self.im2)
            plt.draw()

            plt.figure(3, figsize=(6, 3))
            self.im3 = plt.quiver(self.X[1], self.X[0], ub[1]-(1-self.X[0]**2), ub[0])
            plt.colorbar(self.im3)
            plt.draw()

    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            if comm.Get_rank() == 0 and comm.Get_size() == 1:
                ub = self.u_.backward(self.ub)
                X = self.X
                self.im1.axes.contourf(X[1], X[0], ub[0], 100)
                self.im1.autoscale()
                self.im2.axes.contourf(X[1], X[0], ub[1], 100)
                self.im2.autoscale()
                self.im3.set_UVC(ub[1]-(1-self.X[0]**2), ub[0])
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
    N = (128, 32)
    config['optimization']['mode'] = 'cython'
    d = {
        'N': N,
        'Re': 8000.,
        'dt': 0.002,
        'filename': f'KMM_OSX_{N[0]}_{N[1]}',
        'conv': 0,
        'alfa': 1,
        'modplot': -1,
        'modsave': 10,
        'moderror': 10,
        'family': 'C',
        'checkpoint': 10000000,
        'padding_factor': 1,
        'timestepper': 'IMEXRK443'
        }
    OS = True
    c = OrrSommerfeld(**d)
    t, tstep = c.initialize(from_checkpoint=False)
    c.solve(t=t, tstep=tstep, end_time=0.1)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
    cleanup(vars(c))
