Integrators
===========

The :mod:`.integrators` module contains some interator classes that can be
used to integrate a solution forward in time. There are currently 3 different
integrator classes

    * :class:`.RK4`: Runge-Kutta fourth order
    * :class:`.ETD`: Exponential time differencing Euler method
    * :class:`.ETDRK4`: Exponential time differencing Runge-Kutta fourth order

See, e.g.,
H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

Integrators are set up to solve equations like

.. math::

    \frac{\partial u}{\partial t} = L u + N(u)

where :math:`u` is the solution, :math:`L` is a linear operator and
:math:`N(u)` is the nonlinear part of the right hand side.



