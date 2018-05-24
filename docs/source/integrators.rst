Integrators
-----------

The :mod:`.integrators` module contains some interator classes that can be
used to integrate a solution forward in time. However, for now these integrators
are only implemented for purely Fourier tensor product spaces. 
There are currently 3 different integrator classes

    * :class:`.RK4`: Runge-Kutta fourth order
    * :class:`.ETD`: Exponential time differencing Euler method
    * :class:`.ETDRK4`: Exponential time differencing Runge-Kutta fourth order

See, e.g.,
H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

Integrators are set up to solve equations like

.. math::
   :label: eq:nlsolver

    \frac{\partial u}{\partial t} = L u + N(u)

where :math:`u` is the solution, :math:`L` is a linear operator and
:math:`N(u)` is the nonlinear part of the right hand side.

To illustrate, we consider the time-dependent 1-dimensional Kortveeg-de Vries
equation

.. math::

    \frac{\partial u}{\partial t} + \frac{\partial ^3 u}{\partial x^3} + u \frac{\partial u}{\partial x} = 0

which can also be written as

.. math::

    \frac{\partial u}{\partial t} + \frac{\partial ^3 u}{\partial x^3} + \frac{1}{2}\frac{\partial u^2}{\partial x} = 0

We neglect boundary issues and choose a periodic domain :math:`[0, 2\pi]` with
Fourier exponentials as test functions. The initial condition is chosen as

.. math::
   :label: eq:init_kdv

    u(x, t=0) = 3 A^2/\cosh(0.5 A (x-\pi+2))^2 + 3B^2/\cosh(0.5B(x-\pi+1))^2
 
where :math:`A` and :math:`B` are constants. For discretization in space we use
the basis :math:`V_N = span\{exp(\imath k x)\}_{k=0}^N` and formulate the 
variational problem: find :math:`u \in V_N` such that

.. math::

    \frac{\partial }{\partial t} \Big(u, v \Big) = -\Big(\frac{\partial^3 u }{\partial x^3}, v \Big) - \Big(\frac{1}{2}\frac{\partial u^2}{\partial x}, v\Big), \quad \forall v \in V_N

We see that the first term on the right hand side is linear in :math:`u`, 
whereas the second term is nonlinear. To implement this problem in shenfun
we start by creating the necessary basis and test and trial functions

.. code-block:: python

    import numpy as np
    from shenfun import *

    N = 256
    T = Basis(N, 'F', dtype='d', plan=True)
    u = TrialFunction(T)
    v = TestFunction(T)
    u_ = Array(T)
    u_hat = Function(T)

We then create two functions representing the linear and nonlinear part of 
:eq:`eq:nlsolver`:

.. code-block:: python


    def LinearRHS(**params):
        return -inner(Dx(u, 0, 3), v)

    k = T.wavenumbers(N, scaled=True, eliminate_highest_freq=True)
    def NonlinearRHS(u, u_hat, rhs, **params):
        rhs.fill(0)
        u_[:] = T.backward(u_hat, u_)
        rhs = T.forward(-0.5*u_**2, rhs)
        return rhs*1j*k   # return inner(grad(-0.5*Up**2), v)


Note that we differentiate in ``NonlinearRHS`` by using the wavenumbers ``k``
directly. Alternative notation, that is given in commented out text, is slightly 
slower, but the results are the same.

The solution vector ``u_`` needs also to be initialized according to :eq:`eq:init_kdv`

.. code-block:: python

    A = 25.
    B = 16.
    x = T.points_and_weights(N)[0]
    u_[:] = 3*A**2/np.cosh(0.5*A*(x-np.pi+2))**2 + 3*B**2/np.cosh(0.5*B*(x-np.pi+1))**2
    u_hat = T.forward(u_, u_hat)

Finally we create an instance of the :class:`.ETDRK4` solver, and integrate
forward with a given timestep

.. code-block:: python

    dt = 0.01/N**2
    end_time = 0.006
    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS)
    integrator.setup(dt)
    u_hat = integrator.solve(u_, u_hat, dt, (0, end_time))

The solution is two waves travelling through eachother, seemingly undisturbed.

.. image:: KdV.png
    :width: 600px
    :height: 400px

