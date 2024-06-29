import matplotlib.pyplot as plt
from shenfun import *
from ChannelFlow import KMM


class MKM(KMM):

    def __init__(self, N=(32, 32, 32), domain=((-1, 1), (0, 2*np.pi), (0, np.pi)), Re=180.,
                 dt=0.1, conv=0, modplot=100, modsave=1e8, moderror=100, filename='KMM',
                 family='C', padding_factor=(1, 1.5, 1.5), checkpoint=1000, rand=1e-7,
                 timestepper='IMEXRK3'):
        utau = self.utau = 1
        KMM.__init__(self, N=N, domain=domain, nu=1./Re, dt=dt, conv=conv, modplot=modplot,
                     modsave=modsave, moderror=moderror, filename=filename, family=family,
                     padding_factor=padding_factor, checkpoint=checkpoint,
                     timestepper=timestepper, dpdy=-utau**2)
        self.Re = Re
        self.rand = rand
        self.flux = np.array([618.97]) # Re_tau=180

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        X = self.X
        Y = np.where(X[0] < 0, 1+X[0], 1-X[0])
        utau = self.nu*self.Re
        Um = 46.9091*utau # For Re=180
        #Um = 56.*utau
        Xplus = Y*self.Re
        Yplus = X[1]*self.Re
        Zplus = X[2]*self.Re
        duplus = Um*0.2/utau  #Um*0.25/utau
        alfaplus = self.F1.domain[1]/200.  # May have to adjust these two for different Re
        betaplus = self.F2.domain[1]/100.  #
        sigma = 0.00055 # 0.00055
        epsilon = Um/200.   #Um/200.
        U = Array(self.BD)
        U[1] = Um*(Y-0.5*Y**2)
        dev = 1+0*self.rand*np.random.randn(Y.shape[0], Y.shape[1], Y.shape[2])
        #dev = np.fromfile('dev.dat').reshape((64, 64, 64))
        dd = utau*duplus/2.0*Xplus/40.*np.exp(-sigma*Xplus**2+0.5)*np.cos(betaplus*Zplus)*dev[:, slice(0, 1), :]
        U[1] += dd
        U[2] += epsilon*np.sin(alfaplus*Yplus)*Xplus*np.exp(-sigma*Xplus**2)*dev[:, :, slice(0, 1)]
        u_ = U.forward(self.u_)
        u_.mask_nyquist(self.mask)
        U = u_.backward(U)
        u_ = U.forward(self.u_)
        self.g_[:] = 1j*self.K[1]*u_[2] - 1j*self.K[2]*u_[1]
        return 0, 0

    def init_plots(self):
        ub = self.u_.backward(self.ub)
        self.im1 = 1
        if comm.Get_rank() == 0:

            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[0, :, :, 0], 100)
            plt.colorbar(self.im1)
            plt.draw()

            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[1, :, :, 0], 100)
            plt.colorbar(self.im2)
            plt.draw()

            plt.figure(3, figsize=(6, 3))
            self.im3 = plt.contourf(self.X[2][:, 0, :], self.X[0][:, 0, :], ub[0, :, 0, :], 100)
            plt.colorbar(self.im3)
            plt.draw()

    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            ub = self.u_.backward(self.ub)
            if comm.Get_rank() == 0:
                X = self.X
                self.im1.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], ub[0, :, :, 0], 100)
                self.im1.autoscale()
                plt.figure(1)
                plt.pause(1e-6)
                self.im2.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], ub[1, :, :, 0], 100)
                self.im2.autoscale()
                plt.figure(2)
                plt.pause(1e-6)
                self.im3.axes.contourf(X[2][:, 0, :], X[0][:, 0, :], ub[0, :, 0, :], 100)
                self.im3.autoscale()
                plt.figure(3)
                plt.pause(1e-6)

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            e0 = inner(1, ub[0]*ub[0])
            e1 = inner(1, ub[1]*ub[1])
            e2 = inner(1, ub[2]*ub[2])
            q = inner(1, ub[1])
            dp = (self.flux[0]-inner(1, self.u_[1]))/self.TD.volume
            #dp = self.pdes1d['v0'].N[1][0]
            divu = self.divu().backward()
            e3 = np.sqrt(inner(1, divu*divu))
            if comm.Get_rank() == 0:
                print("Time %2.5f Energy %2.6e %2.6e %2.6e Flux %2.12e %2.6e div %2.6e" %(t, e0, e1, e2, q, dp, e3))

    def update(self, t, tstep):
        self.plot(t, tstep)
        # Dynamically adjust flux
        if tstep % 1 == 0:
            ub1 = self.u_[1].backward(self.ub[1])
            beta = inner(1, ub1)
            q = (self.flux[0] - beta)
            V = self.TD.volume
            if comm.Get_rank() == 0:
                self.u_[1, :, 0, 0] *= (1+q/V/self.u_[1, 0, 0, 0])
                #self.pdes1d['v0'].N[1][:] += q/V/beta/10
        self.print_energy_and_divergence(t, tstep)

if __name__ == '__main__':
    from time import time
    from mpi4py_fft import generate_xdmf
    N = (64, 64, 32)
    d = {
        'N': N,
        'Re': 180.,
        'dt': 0.0005,
        'filename': f'MKM_{N[0]}_{N[1]}_{N[1]}',
        'conv': 0,
        'modplot': 100,
        'modsave': 1000,
        'moderror': 1,
        'family': 'C',
        'checkpoint': 100,
        'padding_factor': (1.5, 1.5, 1.5),
        'timestepper': 'IMEXRK222'
        }
    c = MKM(**d)
    t, tstep = c.initialize(from_checkpoint=False)
    t0 = time()
    c.solve(t=t, tstep=tstep, end_time=0.01)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
    cleanup(vars(c))
