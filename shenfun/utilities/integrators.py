r"""
Module for some integrators.

    - RK4:      Runge-Kutta fourth order
    - ETD:      Exponential time differencing Euler method
    - ETDRK4:   Exponential time differencing Runge-Kutta fourth order

See, e.g.,
H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

Integrators are set up to solve equations like

.. math::

    \frac{\partial u}{\partial t} = L u + N(u)

where :math:`u` is the solution, :math:`L` is a linear operator and
:math:`N(u)` is the nonlinear part of the right hand side.

"""
import types
import numpy as np
from shenfun import Function, TPMatrix
from shenfun import inheritdocstrings

__all__ = ('RK4', 'ETDRK4', 'ETD')

#pylint: disable=unused-variable

class IntegratorBase(object):
    """Abstract base class for integrators

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
                 To be called at the end of a timestep
        params : dictionary
                 Any relevant keyword arguments

    """

    def __init__(self, T,
                 L=lambda *args, **kwargs: 0,
                 N=lambda *args, **kwargs: 0,
                 update=lambda *args, **kwargs: None,
                 **params):
        _p = {'call_update': -1,
              'dt': 0}
        _p.update(params)
        self.params = _p
        self.T = T
        self.LinearRHS = types.MethodType(L, self)
        self.NonlinearRHS = types.MethodType(N, self)
        self.update = types.MethodType(update, self)

    def setup(self, dt):
        """Set up solver"""
        pass

    def solve(self, u, u_hat, dt, trange):
        """Integrate forward in end_time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                    The solution array in spectral space
            dt : float
                 Timestep
            trange : two-tuple
                     Time and end time
        """
        pass

@inheritdocstrings
class ETD(IntegratorBase):
    """Exponential time differencing Euler method

    H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
    3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments
    """

    def __init__(self, T,
                 L=lambda *args, **kwargs: 0,
                 N=lambda *args, **kwargs: 0,
                 update=lambda *args, **kwargs: None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.dU = Function(T)
        self.psi = None
        self.ehL = None

    def setup(self, dt):
        """Set up ETD ODE solver"""
        self.params['dt'] = dt
        L = self.LinearRHS(**self.params)
        if isinstance(L, TPMatrix):
            assert L.isidentity()
            L = L.scale
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
        """Integrate forward in time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        if self.psi is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.dU = self.NonlinearRHS(u, u_hat, self.dU, **self.params)
            u_hat[:] = self.ehL*u_hat + dt*self.psi*self.dU
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat

@inheritdocstrings
class ETDRK4(IntegratorBase):
    """Exponential time differencing Runge-Kutta 4'th order method

    H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
    3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments
    """
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
        self.ehL = None
        self.ehL_h = None

    def setup(self, dt):
        """Set up ETDRK4 ODE solver"""
        self.params['dt'] = dt
        L = self.LinearRHS(**self.params)
        if isinstance(L, TPMatrix):
            assert L.isidentity()
            L = L.scale
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
        """Integrate forward in time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        if self.a is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.U_hat0[:] = u_hat*self.ehL_h
            self.U_hat1[:] = u_hat*self.ehL
            for rk in range(4):
                self.dU = self.NonlinearRHS(u, u_hat, self.dU, **self.params)
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

@inheritdocstrings
class RK4(IntegratorBase):
    """Regular 4'th order Runge-Kutta integrator

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments
    """
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
        """Set up RK4 ODE solver"""
        self.params['dt'] = dt

    def solve(self, u, u_hat, dt, trange):
        """Integrate forward in end_time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        if self.a is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        L = self.LinearRHS(**self.params)
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.U_hat0[:] = self.U_hat1[:] = u_hat
            for rk in range(4):
                dU = self.NonlinearRHS(u, u_hat, self.dU, **self.params)
                dU += L*u_hat
                if rk < 3:
                    u_hat[:] = self.U_hat0 + self.b[rk]*dt*dU
                self.U_hat1 += self.a[rk]*dt*dU
            u_hat[:] = self. U_hat1
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat
