import numpy as np
from shenfun import Function

__all__ = ('RK4', 'ETDRK4', 'ETD')

class IntegratorBase(object):

    def __init__(self, T,
                 L=lambda *args, **kwargs: 0,
                 N=lambda *args, **kwargs: 0,
                 update=lambda *args, **kwargs: None,
                 **params):
        _p = {
             'call_update': -1,
             'dt': 0
             }
        _p.update(params)
        self.params = _p
        self.T = T
        self.LinearRHS = L
        self.NonlinearRHS = N
        self.update = update

    def setup(self, dt):
        pass

class ETD(IntegratorBase):

    def __init__(self, T,
                 L=lambda *args, **kwargs: 0,
                 N=lambda *args, **kwargs: 0,
                 update=lambda *args, **kwargs: None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.dU = Function(T)
        self.psi = None

    def setup(self, dt):
        # Set up ETDRK4 ODE solver
        self.params['dt'] = dt
        L = self.LinearRHS()
        L = np.atleast_1d(L)
        hL = L*dt
        self.ehL = np.exp(hL)
        M = 50
        psi = self.psi = np.zeros(hL.shape, dtype=np.float)
        for k in range(1, M+1):
            ll = hL+np.exp(np.pi*1j*(k-0.5)/M)
            psi += ((np.exp(ll)-1.)/ll).real

        psi /= M

    def solve(self, u, u_hat, dt, trange):
        if self.psi is None or abs(self.params['dt']-dt)>1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.dU = self.NonlinearRHS(u, u_hat, self.dU)
            u_hat[:] = self.ehL*u_hat + dt*self.psi*self.dU
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat

class ETDRK4(IntegratorBase):

    def __init__(self, T,
                 L=lambda *args, **kwargs: 0,
                 N=lambda *args, **kwargs: 0,
                 update=lambda *args, **kwargs: None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.U_hat0 = Function(T)
        self.U_hat1 = Function(T)
        self.dU = Function(T)
        self.dU0 = Function(T)
        self.V2 = Function(T)
        self.psi = np.zeros((4,)+self.U_hat0.shape, dtype=np.float)
        self.a = None
        self.b = [0.5, 0.5, 0.5]

    def setup(self, dt):
        # Set up ETDRK4 ODE solver
        self.params['dt'] = dt
        L = self.LinearRHS()
        L = np.atleast_1d(L)
        hL = L*dt
        ehL = self.ehL = np.exp(hL)
        ehL_h = self.ehL_h = np.exp(hL/2.)

        M = 50
        psi = self.psi = np.zeros((4,) + hL.shape, dtype=np.float)
        for k in range(1, M+1):
            ll = hL+np.exp(np.pi*1j*(k-0.5)/M)
            psi[0] += ((np.exp(ll)-1.)/ll).real
            psi[1] += ((np.exp(ll)-ll-1.)/ll**2).real
            psi[2] += ((np.exp(ll)-0.5*ll**2-ll-1.)/ll**3).real
            ll2 = hL/2.+np.exp(np.pi*1j*(k-0.5)/M)
            psi[3] += ((np.exp(ll2)-1.)/(ll2)).real

        psi /= M
        a = [psi[0]-3*psi[1]+4*psi[2]]
        a.append(2*psi[1]-4*psi[2])
        a.append(2*psi[1]-4*psi[2])
        a.append(-psi[1]+4*psi[2])
        self.a = a

    def solve(self, u, u_hat, dt, trange):
        if self.a is None or abs(self.params['dt']-dt)>1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.U_hat0[:] = u_hat*self.ehL_h
            self.U_hat1[:] = u_hat*self.ehL
            for rk in range(4):
                self.dU = self.NonlinearRHS(u, u_hat, self.dU)
                if rk < 2:
                    u_hat[:] = self.U_hat0 + self.b[rk]*dt*self.psi[3]*self.dU
                elif rk == 2:
                    u_hat[:] = self.ehL_h*self.V2 + self.b[rk]*dt*self.psi[3]*(2*self.dU-self.dU0)

                if rk == 0:
                    self.dU0[:] = self.dU
                    self.V2[:] = u_hat

                self.U_hat1 += self.a[rk]*dt*self.dU
            u_hat[:] = self.U_hat1
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat

class RK4(IntegratorBase):

    def __init__(self, T,
                 L=lambda *args, **kwargs: 0,
                 N=lambda *args, **kwargs: 0,
                 update=lambda *args, **kwargs: None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.U_hat0 = Function(T)
        self.U_hat1 = Function(T)
        self.dU = Function(T)
        self.a = np.array([1./6., 1./3., 1./3., 1./6.])
        self.b = np.array([0.5, 0.5, 1.])

    def setup(self, dt):
        # Set up RK4 ODE solver
        self.params['dt'] = dt

    def solve(self, u, u_hat, dt, trange):
        if self.a is None or abs(self.params['dt']-dt)>1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        L = self.LinearRHS()
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.U_hat0[:] = self.U_hat1[:] = u_hat
            for rk in range(4):
                dU = self.NonlinearRHS(u, u_hat, self.dU)
                dU += L*u_hat
                if rk < 3: u_hat[:] = self.U_hat0 + self.b[rk]*dt*dU
                self.U_hat1 += self.a[rk]*dt*dU
            u_hat[:] =self. U_hat1
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat
