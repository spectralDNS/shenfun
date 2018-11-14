.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - Kuramato-Sivashinsky equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:Authors: Mikael Mortensen (mikaem at math.uio.no)
:Date: Nov 14, 2018

*Summary.* This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve the time-dependent,
nonlinear Kuramato-Sivashinsky equation, in a doubly periodic domain. The demo is implemented in
a single Python file `KuramatoSivashinsky.py <https://github.com/spectralDNS/shenfun/blob/master/demo/Kuramato_Sivashinsky.py>`__, and it may be run
in parallel using MPI.

The Kuramato-Sivashinsky equation
=================================

.. raw:: html
        
        <embed src="https://rawgit.com/spectralDNS/spectralutilities/master/movies/Kuramato_movie_128.gif"  autoplay="false" loop="true"></embed>
        <p><em></em></p>

Movie showing the evolution of the solution :math:`u` from Eq. :eq:`eq:ks`.

Model equation
--------------

The Kuramato-Sivashinsky (KS) equation is known for its chaotic bahaviour, and it is
often used in study of turbulence or turbulent combustion. We will here solve
the KS equation in a doubly periodic domain :math:`[-30\pi, 30\pi]^2`, starting from a
single Gaussian pulse

.. math::
   :label: eq:ks

        
        \frac{\partial u(\boldsymbol{x},t)}{\partial t} + \nabla^2 u(\boldsymbol{x},t) + \nabla^4
        u(\boldsymbol{x},t) + |\nabla u(\boldsymbol{x},t)|^2 = 0 \quad \text{for }\, \boldsymbol{x} \in \Omega=[-30 \pi, 30\pi]^2
         
        

.. math::
   :label: _auto1

          
        u(\boldsymbol{x}, 0) = \exp(-0.01 \boldsymbol{x} \cdot \boldsymbol{x}) \notag
        
        

.. _sec:spectralgalerkin:

Spectral Galerkin method
------------------------
The PDE in :eq:`eq:ks` can be solved with many different
numerical methods. We will here use the `shenfun <https://github.com/spectralDNS/shenfun>`__ software and this software makes use of
the spectral Galerkin method. Being a Galerkin method, we need to reshape the
governing equations into proper variational forms, and this is done by
multiplying  :eq:`eq:ks` with the complex conjugate of a proper
test function and then integrating
over the domain. To this end we use testfunction :math:`v\in H^s(\Omega)`, where :math:`H^s(\Omega)` is the Sobolev space of order :math:`s`, and we obtain

.. math::
   :label: eq:du_var

        
        \frac{\partial}{\partial t} \int_{\Omega} u\, \overline{v}\, w \,dx = -\int_{\Omega}
        \left(\nabla^2 u + \nabla^4 u \ + |\nabla u|^2 \right) \overline{v} \, w \,dx.
        
        

Note that the overline is used to indicate a complex conjugate, whereas :math:`w`
is a weight function. The function :math:`u`
is now to be considered a trial function, and the integrals over the
domain are often referred to as inner products. With inner product notation

.. math::
        
        \left(u, v\right) = \int_{\Omega} u \, \overline{v} \, w \, dx.
        

the variational problem can be formulated as

.. math::
   :label: eq:du_var2

        
        \frac{\partial}{\partial t} (u, v) = -\left(\nabla^2 u + \nabla^4 u + |\nabla u|^2,
        v \right). 
        

The space and time discretizations are
still left open. There are numerous different approaches that one could take for
discretizing in time. Here we will use a fourth order exponential Runge-Kutta
method.

Discretization
--------------

We discretize the model equation in space using continuously differentiable
Fourier basis functions

.. math::
   :label: _auto2

        
        \phi_l(x) = e^{\imath \underline{l} x}, \quad -\infty < l < \infty,
        
        

where :math:`l` is the wavenumber, and :math:`\underline{l}=\frac{2\pi}{L}l` is the scaled wavenumber, scaled with domain
length :math:`L` (here :math:`60\pi`). Since we want to solve these equations on a computer, we need to choose
a finite number of test functions. A basis :math:`V^N` can be defined as

.. math::
   :label: eq:Vn

        
        V^N(x) = \text{span} \{\phi_l(x)\}_{l\in \boldsymbol{l}}, 
        

where :math:`N` is chosen as an even positive integer and :math:`\boldsymbol{l} = (-N/2,
-N/2+1, \ldots, N/2-1)`. And now, since :math:`\Omega` is a
two-dimensional domain, we can create a Cartesian product of two such bases:

.. math::
   :label: eq:Wn

        
        W^{\boldsymbol{N}}(x, y) = V^N(x) \times V^N(y), 
        

where :math:`\boldsymbol{N} = (N, N)`. Obviously, it is not necessary to use the
same number (:math:`N`) of basis functions for each direction, but it is done here
for simplicity. A 2D tensor product basis function is now defined as

.. math::
   :label: _auto3

        
        \Phi_{lm}(x,y) = e^{\imath \underline{l} x} e^{\imath \underline{m} y}
        = e^{\imath (\underline{l}x + \underline{m}y )},
        
        

where the indices for :math:`y`-direction are :math:`\underline{m}=\frac{2\pi}{L}m`, and
:math:`\boldsymbol{m}` is the same set as :math:`\boldsymbol{l}` due to using the same number of basis functions for each direction. One
distinction, though, is that for the :math:`y`-direction expansion coefficients are only stored for
:math:`m=(0, 1, \ldots, N/2)` due to Hermitian symmetry (real input data).

We now look for solutions of the form

.. math::
   :label: _auto4

        
        u(x, y) = \sum_{l=-N/2}^{N/2-1}\sum_{m=-N/2}^{N/2-1}
        \hat{u}_{lm} \Phi_{lm}(x,y).
        
        

The expansion coefficients :math:`\hat{u}_{lm}` can be related directly to the solution :math:`u(x,
y)` using Fast Fourier Transforms (FFTs) if we are satisfied with obtaining
the solution in quadrature points corresponding to

.. math::
   :label: _auto5

        
         x_i = \frac{60 \pi i}{N}-30\pi \quad \forall \, i \in \boldsymbol{i},
        \text{where}\, \boldsymbol{i}=(0,1,\ldots,N-1), 
        
        

.. math::
   :label: _auto6

          
         y_j = \frac{60 \pi j}{N}-30\pi \quad \forall \, j \in \boldsymbol{j},
        \text{where}\, \boldsymbol{j}=(0,1,\ldots,N-1).
        
        

Note that these points are different from the standard (like :math:`2\pi j/N`) since
the domain
is set to :math:`[-30\pi, 30\pi]^2` and not the more common :math:`[0, 2\pi]^2`. We now have

.. math::
   :label: _auto7

        
        u(x_i, y_j) =
        \mathcal{F}_y^{-1}\left(\mathcal{F}_x^{-1}\left(\hat{u}\right)\right)
        \, \forall\, (i,j)\in\boldsymbol{i} \times \boldsymbol{j},
        
        

where :math:`\mathcal{F}_x^{-1}` is the inverse Fourier transform along direction
:math:`x`, for all :math:`j \in \boldsymbol{j}`. Note that the two
inverse FFTs are performed sequentially, one direction at the time, and that
there is no scaling factor due
the definition used for the inverse
`Fourier transform <https://mpi4py-fft.readthedocs.io/en/latest/dft.html>`__:

.. math::
   :label: _auto8

        
        u(x_j) = \sum_{l=-N/2}^{N/2-1} \hat{u}_l e^{\imath \underline{l}
        x_j}, \quad \,\, \forall \, j \in \, \boldsymbol{j}.
        
        

Note that this differs from the definition used by, e.g.,
`Numpy <https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html>`__.

The inner products used in Eq. :eq:`eq:du_var2` may be
computed using forward FFTs (using weight functions :math:`w=1/L`):

.. math::
   :label: _auto9

        
        \left(u, \Phi_{lm}\right) = \hat{u}_{lm} =
        \frac{1}{N^2}
        \mathcal{F}_l\left(\mathcal{F}_m\left({u}\right)\right)
        \quad \forall (l,m) \in \boldsymbol{l} \times \boldsymbol{m},
        
        

From this we see that the variational forms
may be written in terms of the Fourier transformed :math:`\hat{u}`. Expanding the
exact derivatives of the nabla operator, we have

.. math::
   :label: _auto10

        
        (\nabla^2 u, v) =
        -(\underline{l}^2+\underline{m}^2)\hat{u}_{l,m}, 
        
        

.. math::
   :label: _auto11

          
        (\nabla^4 u, v) = (\underline{l}^2+\underline{m}^2)^2\hat{u}_{l,m}, 
        
        

.. math::
   :label: _auto12

          
        (|\nabla u|^2, v) = \widehat{|\nabla u|^2}
        
        

and as such the equation to be solved can be found directly as

.. math::
   :label: eq:du_var3

        
        \frac{\partial \hat{u}}{\partial t}  =
        \left(\underline{l}^2+\underline{m}^2 -
        (\underline{l}^2+\underline{m}^2)^2\right)\hat{u} - \widehat{|\nabla u|^2},
        
        

Implementation
==============

The model equation :eq:`eq:ks` is implemented in shenfun using Fourier basis functions for
both :math:`x` and :math:`y` directions. We start the solver by implementing necessary
functionality from required modules like `Numpy <https://numpy.org>`__, `Sympy <https://sympy.org>`__
`matplotlib <https://matplotlib.org>`__ and `mpi4py <https://bitbucket.org/mpi4py>`__, in
addition to `shenfun <https://github.com/spectralDNS/shenfun>`__:

.. code-block:: python

    from sympy import symbols, exp, lambdify
    import numpy as np
    import matplotlib.pyplot as plt
    from mpi4py import MPI
    from shenfun import *

The size of the problem (in real space) is then specified, before creating
the :class:`.TensorProductSpace`, which is a Cartesian product of two Fourier bases. We also
create a :class:`.VectorTensorProductSpace`, since this is required for computing the
gradient of the scalar field ``u``. The gradient is required for the nonlinear
term.

.. code-block:: python

    # Size of discretization
    N = (128, 128)
    
    comm = MPI.COMM_WORLD
    K0 = Basis(N[0], 'F', domain=(-30*np.pi, 30*np.pi), dtype='D')
    K1 = Basis(N[1], 'F', domain=(-30*np.pi, 30*np.pi), dtype='d')
    T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
    TV = VectorTensorProductSpace([T, T])

Test and trialfunctions are required for assembling the variational forms:

.. code-block:: python

    u = TrialFunction(T)
    v = TestFunction(T)

and some arrays are required to hold the solution. We also create an array
``gradu``, that will be used to compute the gradient in the nonlinear term.
Finally, the wavenumbers are collected in list ``K``. Here one feature is worth
mentioning. The gradient in spectral space can be computed as ``1j*K*U_hat``.
However, since this is an odd derivative, and we are using an even number ``N``
for the size of the domain, the highest wavenumber must be set to zero. This is
the purpose of the last keyword argument to ``local_wavenumbers`` below.

.. code-block:: python

    U = Array(T)
    U_hat = Function(T)
    gradu = Array(TV)
    K = np.array(T.local_wavenumbers(True, True, eliminate_highest_freq=True))

Note that using this ``K`` in computing derivatives has the same effect as
achieved by symmetrizing the Fourier series by replacing the first sum below
with the second when computing odd derivatives.

.. math::
   :label: _auto13

        
        u  = \sum_{k=-N/2}^{N/2-1} \hat{u} e^{\imath k x}
        
        

.. math::
   :label: _auto14

          
        u  = \sideset{}{'}\sum_{k=-N/2}^{N/2} \hat{u} e^{\imath k x}
        
        

Here :math:`\sideset{}{'}\sum` means that the first and last items in the sum are
divided by two. Note that the two sums are equal as they stand (due to aliasing), but only the
latter (known as the Fourier interpolant) gives the correct (zero) derivative of
the basis with the highest wavenumber.

Sympy is used to generate an initial condition, as stated in Eq :eq:`eq:ks`

.. code-block:: python

    # Use sympy to set up initial condition
    x, y = symbols("x,y")
    ue = exp(-0.01*(x**2+y**2))
    ul = lambdify((x, y), ue, 'numpy')

Shenfun has a few integrators implemented in the :mod:`.integrators`
submodule. Two such integrators are the 4th order explicit Runge-Kutta method
``RK4``, and the exponential 4th order Runge-Kutta method ``ETDRK4``. Both these
integrators need two methods provided by the problem being solved, representing
the linear and nonlinear terms in the problem equation. We define two methods
below, called ``LinearRHS`` and ``NonlinearRHS``

.. code-block:: python

    def LinearRHS():
        # Assemble diagonal bilinear forms
        L = -(inner(div(grad(u)), v) + inner(div(grad(div(grad(u)))), v))
        return L
    
    def NonlinearRHS(U, U_hat, dU):
        # Assemble nonlinear term
        global gradu
        gradu = TV.backward(1j*K*U_hat, gradu)
        dU = T.forward(0.5*(gradu[0]*gradu[0]+gradu[1]*gradu[1]), dU)
        return dU

The code should, hopefully, be self-explanatory.

All that remains now is to initialize the solution arrays and to setup the
integrator plus some plotting functionality for visualizing the results. Note
that visualization is only nice when running the code in serial. For parallel,
it is recommended to use :class:`.HDF5Writer`, to store intermediate results to the HDF5
format, for later viewing in, e.g., Paraview.

The solution is initialized as

.. code-block:: python

    #initialize
    X = T.local_mesh(True)
    U[:] = ul(*X)
    U_hat = T.forward(U, U_hat)

And we also create an update function for plotting intermediate results with a
cool colormap:

.. code-block:: python

    # Integrate using an exponential time integrator
    plt.figure()
    cm = plt.get_cmap('hot')
    image = plt.contourf(X[0], X[1], U, 256, cmap=cm)
    plt.draw()
    plt.pause(1e-6)
    count = 0
    def update(u, u_hat, t, tstep, **params):
        global count
        if tstep % params['plot_step'] == 0 and params['plot_step'] > 0:
            u = T.backward(u_hat, u)
            image.ax.clear()
            image.ax.contourf(X[0], X[1], U, 256, cmap=cm)
            plt.pause(1e-6)
            count += 1
            plt.savefig('Kuramato_Sivashinsky_N_{}_{}.png'.format(N[0], count))
    

Now all that remains is to create the integrator and call it

.. code-block:: python

    if __name__ == '__main__':
        par = {'plot_step': 100}
        dt = 0.01
        end_time = 100
        integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
        #integrator = RK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
        integrator.setup(dt)
        U_hat = integrator.solve(U, U_hat, dt, (0, end_time))


