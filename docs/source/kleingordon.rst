.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - Cubic nonlinear Klein-Gordon equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:Authors: Mikael Mortensen (mikaem at math.uio.no)
:Date: Jan 24, 2019

*Summary.* This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve the time-dependent,
nonlinear Klein-Gordon equation, in a triply periodic domain. The demo is implemented in
a single Python file `KleinGordon.py <https://github.com/spectralDNS/shenfun/blob/master/demo/KleinGordon.py>`__, and it may be run
in parallel using MPI. The Klein-Gordon equation is solved using a mixed
formulation. The discretization, and some background on the spectral Galerkin
method is given first, before we turn to the actual details of the ``shenfun``
implementation.

.. _mov:kleingordon:

The nonlinear Klein-Gordon equation
===================================

.. raw:: html
        
        <embed src="https://rawgit.com/spectralDNS/spectralutilities/master/movies/KleinGordon.gif"  autoplay="false" loop="true"></embed>
        <p><em></em></p>

Movie showing the evolution of the solution :math:`u` from Eq. :eq:`eq:kg`, in a slice through the center of the domain, computed with the code described in this demo.

Model equation
--------------

The cubic nonlinear Klein-Gordon equation is a wave equation important for many
scientific applications such as solid state physics, nonlinear optics and
quantum field theory :cite:`abdul08`. The equation is given as

.. math::
   :label: eq:kg

        
        \frac{\partial^2 u}{\partial t^2} = \nabla^2 u - \gamma(u - u|u|^2) \quad
        \text{for} \, u \in
        \Omega, 
        

with initial conditions

.. math::
   :label: eq:init

        
        u(\boldsymbol{x}, t=0) = u^0 \quad \text{and} \quad \frac{\partial u(\boldsymbol{x},
        t=0)}{\partial t} = u_t^0. 
        

The spatial coordinates are here denoted as :math:`\boldsymbol{x} = (x, y, z)`, and
:math:`t` is time. The parameter :math:`\gamma=\pm 1` determines whether the equations are focusing
(:math:`+1`) or defocusing (:math:`-1`) (in the movie we have used :math:`\gamma=1`). The domain :math:`\Omega=[-2\pi, 2\pi]^3` is triply
periodic and initial conditions will here be set as

.. math::
   :label: _auto1

        
        u^0 = 0.1 \exp \left( -\boldsymbol{x} \cdot \boldsymbol{x} \right), 
        
        

.. math::
   :label: _auto2

          
        u_t^0 = 0.
        
        

We will solve these equations using a mixed formulation and a spectral Galerkin
method. The mixed formulation reads

.. math::
   :label: eq:df

        
        \frac{\partial f}{\partial t} = \nabla^2 u - \gamma (u - u|u|^2), 
        

.. math::
   :label: eq:du

          
        \frac{\partial u}{\partial t} = f. 
        

The energy of the solution can be computed as

.. math::
   :label: _auto3

        
        E(u) = \int_{\Omega} \left( \frac{1}{2} f^2 + \frac{1}{2}|\nabla u|^2 + \gamma(\frac{1}{2}u^2 - \frac{1}{4}u^4) \right) dx
        
        

and it is crucial that this energy remains constant in time.

The movie (:ref:`mov:kleingordon`) is showing the solution :math:`u`, computed with the
code shown in the bottom of Sec. :ref:`sec:solver`.

.. _sec:specgal:

Spectral Galerkin formulation
-----------------------------
The PDEs in :eq:`eq:df` and :eq:`eq:du` can be solved with many different
numerical methods. We will here use the `shenfun <https://github.com/spectralDNS/shenfun>`__ software and this software makes use of
the spectral Galerkin method. Being a Galerkin method, we need to reshape the
governing equations into proper variational forms, and this is done by
multiplying  :eq:`eq:df` and :eq:`eq:du` with the complex conjugate of proper
test functions and then integrating
over the domain. To this end we use continuously differentiable
testfunctions :math:`g\in C(\Omega)` with Eq. :eq:`eq:df`  and  :math:`v \in
C(\Omega)` with Eq. :eq:`eq:du`, and we obtain

.. math::
   :label: eq:df_var

        
        \frac{\partial}{\partial t} \int_{\Omega} f\, \overline{g}\, w \,dx = \int_{\Omega}
        \left(\nabla^2 u - \gamma( u\, - u|u|^2) \right) \overline{g} \, w \,dx,  
        

.. math::
   :label: eq:kg:du_var

          
        \frac{\partial }{\partial t} \int_{\Omega} u\, \overline{v}\, w \, dx =
        \int_{\Omega} f\, \overline{v} \, w \, dx. 
        

Note that the overline is used to indicate a complex conjugate, and
:math:`w` is a weight function associated with the test functions. The functions
:math:`f` and :math:`u` are now
to be considered as trial functions, and the integrals over the
domain are often referred to as inner products. With inner product notation

.. math::
        
        \left(u, v\right) = \int_{\Omega} u \, \overline{v} \, w\, dx.
        

and an integration by parts on the Laplacian, the variational problem can be
formulated as:

.. math::
   :label: eq:df_var2

        
        \frac{\partial}{\partial t} (f, g) = -(\nabla u, \nabla g)
        -\gamma \left( u - u|u|^2, g \right),  
        

.. math::
   :label: eq:kg:du_var2

          
        \frac{\partial }{\partial t} (u, v) = (f, v). 
        

The time and space discretizations are
still left open. There are numerous different approaches that one could take for
discretizing in time, and the first two terms on the right hand side of
:eq:`eq:df_var2` can easily be treated implicitly as well as explicitly. However,
the approach we will follow in Sec. (:ref:`sec:rk`) is a fully explicit 4th order `Runge-Kutta <https://en.wikipedia.org/wiki/Runge-Kutta_methods>`__ method.

Discretization
--------------
To find a numerical solution we need to discretize the continuous problem
:eq:`eq:df_var2` and :eq:`eq:kg:du_var2` in space as well as time. Since the
problem is triply periodic, Fourier exponentials are normally the best choice
for trial and test functions, and as such we use basis functions

.. math::
   :label: _auto4

        
        \phi_l(x) = e^{\imath \underline{l} x}, \quad -\infty < l < \infty,
        
        

where :math:`l` is the wavenumber, and
:math:`\underline{l}=\frac{2\pi}{L}l` is the scaled wavenumber, scaled with domain
length :math:`L` (here :math:`4\pi`). Since we want to solve these equations on a computer, we need to choose
a finite number of test functions. A basis :math:`V^N` can be defined as

.. math::
   :label: eq:kg:Vn

        
        V^N(x) = \text{span} \{\phi_l(x)\}_{l\in \boldsymbol{l}}, 
        

where :math:`N` is chosen as an even positive integer and :math:`\boldsymbol{l} = (-N/2,
-N/2+1, \ldots, N/2-1)`. And now, since :math:`\Omega` is a
three-dimensional domain, we can create Cartesian products of such bases to get,
e.g., for three dimensions

.. math::
   :label: eq:kg:Wn

        
        W^{\boldsymbol{N}}(x, y, z) = V^N(x) \times V^N(y) \times V^N(z), 
        

where :math:`\boldsymbol{N} = (N, N, N)`. Obviously, it is not necessary to use the
same number (:math:`N`) of basis functions for each direction, but it is done here
for simplicity. A 3D tensor product basis function is now defined as

.. math::
   :label: _auto5

        
        \Phi_{l,m,n}(x,y,z) = e^{\imath \underline{l} x} e^{\imath \underline{m} y}
        e^{\imath \underline{n} z} = e^{\imath
        (\underline{l}x + \underline{m}y + \underline{n}z)}
        
        

where the indices for :math:`y`- and :math:`z`-direction are :math:`\underline{m}=\frac{2\pi}{L}m,
\underline{n}=\frac{2\pi}{L}n`, and :math:`\boldsymbol{m}` and :math:`\boldsymbol{n}` are the same as
:math:`\boldsymbol{l}` due to using the same number of basis functions for each direction. One
distinction, though, is that for the :math:`z`-direction expansion coefficients are only stored for
:math:`n=(0, 1, \ldots, N/2)` due to Hermitian symmetry (real input data).

We now look for solutions of the form

.. math::
   :label: eq:usg

        
        u(x, y, z, t) = \sum_{n=-N/2}^{N/2-1}\sum_{m=-N/2}^{N/2-1}\sum_{l=-N/2}^{N/2-1}
        \hat{u}_{l,m,n} (t)\Phi_{l,m,n}(x,y,z). 
        

The expansion coefficients :math:`\hat{u}_{l,m,n}(t)` can be related directly to the solution :math:`u(x,
y, z, t)` using Fast Fourier Transforms (FFTs) if we are satisfied with obtaining
the solution in quadrature points corresponding to

.. math::
   :label: _auto6

        
         x_i = \frac{4 \pi i}{N}-2\pi \quad \forall \, i \in \boldsymbol{i},
        \text{where}\, \boldsymbol{i}=(0,1,\ldots,N-1), 
        
        

.. math::
   :label: _auto7

          
         y_j = \frac{4 \pi j}{N}-2\pi \quad \forall \, j \in \boldsymbol{j},
        \text{where}\, \boldsymbol{j}=(0,1,\ldots,N-1), 
        
        

.. math::
   :label: _auto8

          
         z_k = \frac{4 \pi k}{N}-2\pi \quad \forall \, k \in \boldsymbol{k},
        \text{where}\, \boldsymbol{k}=(0,1,\ldots,N-1). 
        
        

.. math::
   :label: _auto9

          
        
        

Note that these points are different from the standard (like :math:`2\pi j/N`) since
the domain
is set to :math:`[-2\pi, 2\pi]^3` and not the more common :math:`[0, 2\pi]^3`. We have

.. math::
   :label: _auto10

        
        u(x_i, y_j, z_k) =
        \mathcal{F}_k^{-1}\left(\mathcal{F}_j^{-1}\left(\mathcal{F}_i^{-1}\left(\hat{u}\right)\right)\right)
        \, \forall\, (i,j,k)\in\boldsymbol{i} \times \boldsymbol{j} \times
        \boldsymbol{k},
        
        

where :math:`\mathcal{F}_i^{-1}` is the inverse Fourier transform along the direction
of index :math:`i`, for
all :math:`(j, k) \in \boldsymbol{j} \times \boldsymbol{k}`. Note that the three
inverse FFTs are performed sequentially, one direction at the time, and that there is no
scaling factor due to
the definition used for the inverse `Fourier transform <https://mpi4py-fft.readthedocs.io/en/latest/dft.html>`__

.. math::
   :label: _auto11

        
        u(x_j) = \sum_{l=-N/2}^{N/2-1} \hat{u}_l e^{\imath \underline{l}
        x_j}, \quad \,\, \forall \, j \in \, \boldsymbol{j}.
        
        

Note that this differs from the definition used by, e.g.,
`Numpy <https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html>`__.

The inner products used in Eqs. :eq:`eq:df_var2`, :eq:`eq:kg:du_var2` may be
computed using forward FFTs. However, there is a tiny detail that deserves
a comment. The regular Fourier inner product is given as

.. math::
        \int_{0}^{L} e^{\imath \underline{k}x} e^{- \imath \underline{l}x} dx = L\, \delta_{kl}

where a weight function is chosen as :math:`w(x) = 1` and :math:`\delta_{kl}` equals unity
for :math:`k=l` and zero otherwise. In Shenfun we choose instead to use a weight
function :math:`w(x)=1/L`, such that the weighted inner product integrates to
unity:

.. math::
        \int_{0}^{L} e^{\imath \underline{k}x} e^{- \imath \underline{l}x} \frac{1}{L} dx = \delta_{kl}.

With this weight function the scalar product and the forward transform
are the same and we obtain:

.. math::
   :label: _auto12

        
        \left(u, \Phi_{l,m,n}\right) = \hat{u}_{l,m,n} =
        \left(\frac{1}{N}\right)^3
        \mathcal{F}_l\left(\mathcal{F}_m\left(\mathcal{F}_n\left({u}\right)\right)\right)
        \quad \forall (l,m,n) \in \boldsymbol{l} \times \boldsymbol{m} \times
        \boldsymbol{n},
        
        

From this we see that the variational forms :eq:`eq:df_var2` and :eq:`eq:kg:du_var2`
may be written in terms of the Fourier transformed quantities :math:`\hat{u}` and
:math:`\hat{f}`. Expanding the exact derivatives of the nabla operator, we have

.. math::
   :label: _auto13

        
        (\nabla u, \nabla v) =
        (\underline{l}^2+\underline{m}^2+\underline{n}^2)\hat{u}_{l,m,n}, 
        
        

.. math::
   :label: _auto14

          
        (u, v) = \hat{u}_{l,m,n}, 
        
        

.. math::
   :label: _auto15

          
        (u|u|^2, v) = \widehat{u|u|^2}
        
        

and as such the equations to be solved can be found directly as

.. math::
   :label: eq:df_var3

        
        \frac{\partial \hat{f}}{\partial t}  =
        \left(-(\underline{l}^2+\underline{m}^2+\underline{n}^2+\gamma)\hat{u} + \gamma \widehat{u|u|^2}\right),  
        

.. math::
   :label: eq:kg:du_var3

          
        \frac{\partial \hat{u}}{\partial t} = \hat{f}. 
        

There is more than one way to arrive at these equations. Taking the 3D Fourier
transform of both equations  :eq:`eq:df` and :eq:`eq:du` is one obvious way.
With the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__, one can work with the
inner products as seen in :eq:`eq:df_var2` and :eq:`eq:kg:du_var2`, or the Fourier
transforms directly. See for example Sec. :ref:`sec:rk` for how :math:`(\nabla u, \nabla
v)` can be
implemented.  In short, :mod:`.shenfun` contains all the tools required to work with
the spectral Galerkin method, and we will now see how :mod:`.shenfun` can be used to solve
the Klein-Gordon equation.

For completion, we note that the discretized problem to solve can be formulated
with the Galerkin method as:
for all :math:`t>0`, find :math:`(f, u) \in W^N \times W^N`  such that

.. math::
   :label: eq:dff

        
        \frac{\partial}{\partial t} (f, g) = -(\nabla u, \nabla g)
        -\gamma \left( u - u|u|^2, g \right),  
        

.. math::
   :label: eq:kg:duu

          
        \frac{\partial }{\partial t} (u, v) = (f, v) \quad \forall \, (g, v) \in W^N \times W^N. 
        

where :math:`u(x, y, z, 0)` and :math:`f(x, y, z, 0)` are given as the initial conditions
according to Eq. :eq:`eq:init`.

Implementation
==============

To solve the Klein-Gordon equations we need to make use of the Fourier bases in
:mod:`.shenfun`, and these base are found in submodule
:mod:`shenfun.fourier.bases`.
The triply periodic domain allows for Fourier in all three directions, and we
can as such create one instance of this base class using :func:`.Basis` with
family ``Fourier``
for each direction. However, since the initial data are real, we
can take advantage of Hermitian symmetries and thus make use of a
real to complex class for one (but only one) of the directions, by specifying
``dtype='d'``. We can only make use of the
real-to-complex class for the direction that we choose to transform first with the forward
FFT, and the reason is obviously that the output from a forward transform of
real data is now complex. We may start implementing the solver as follows

.. code-block:: python

    from shenfun import *
    from mpi4py import MPI
    import numpy as np
    
    # Set size of discretization
    N = (32, 32, 32)
    
    # Create bases
    K0 = Basis(N[0], 'F', domain=(-2*np.pi, 2*np.pi), dtype='D')
    K1 = Basis(N[1], 'F', domain=(-2*np.pi, 2*np.pi), dtype='D')
    K2 = Basis(N[2], 'F', domain=(-2*np.pi, 2*np.pi), dtype='d')

We now have three instances ``K0``, ``K1`` and ``K2``, corresponding to the basis
:eq:`eq:kg:Vn`, that each can be used to solve
one-dimensional problems. However, we want to solve a 3D problem, and for this
we need a tensor product basis, like :eq:`eq:kg:Wn`, created as a Cartesian
product of these three bases

.. code-block:: python

    # Create communicator
    comm = MPI.COMM_WORLD
    T = TensorProductSpace(comm, (K0, K1, K2), **{'planner_effort':
                                                  'FFTW_MEASURE'})

Here the ``planner_effort``, which is a flag used by `FFTW <http://www.fftw.org>`__, is optional. Possibel choices are from the list
(``FFTW_ESTIMATE``, ``FFTW_MEASURE``, ``FFTW_PATIENT``, ``FFTW_EXHAUSTIVE``), and the
flag determines how much effort FFTW puts in looking for an optimal algorithm
for the current platform. Note that it is also possible to use FFTW `wisdom <http://www.fftw.org/fftw3_doc/Wisdom.html#Wisdom>`__ with
``shenfun``, and as such, for production, one may perform exhaustive planning once
and then simply import the result of that planning later, as wisdom.

The :class:`.TensorProductSpace` instance ``T`` contains pretty much all we need for
computing inner products or fast transforms between real and wavenumber space.
However, since we are going to solve for a mixed system, it is convenient to also use the
:class:`.MixedTensorProductSpace` class

.. code-block:: python

    TT = MixedTensorProductSpace([T, T])

We need containers for the solution as well as intermediate work arrays for,
e.g., the Runge-Kutta method. Arrays are created as

.. code-block:: python

    uf = Array(TT)           # Solution array in physical space
    u, f = uf[:]             # Split solution array by creating two views u and f
    duf = Function(TT)       # Array for right hand sides
    du, df = duf[:]          # Split into views
    uf_hat = Function(TT)    # Solution in spectral space
    uf_hat0 = Function(TT)   # Work array 1
    uf_hat1 = Function(TT)   # Work array 2
    u_hat, f_hat = uf_hat[:] # Split into views

The :class:`.Array` class is a subclass of Numpy's `ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__,
without much more functionality than constructors that return arrays of the
correct shape according to the basis used in the construction. The
:class:`.Array` represents the left hand side of :eq:`eq:usg`,
evaluated on the quadrature mesh. A different type
of array is returned by the :class:`.Function`
class, that subclasses both Nympy's ndarray as well as an internal
:class:`.BasisFunction`
class. An instance of the :class:`.Function` represents the entire
spectral Galerkin function :eq:`eq:usg`. As such, it can
be used in complex variational linear forms. For example, if you want
to compute the partial derivative :math:`\partial u/\partial x`, then this
may be achieved by projection, i.e., find :math:`u_x \in V^N` such that
:math:`(u_x-\partial u/\partial x, v) = 0`, for all :math:`v \in V^N`. This
projection may be easily computed in :mod:`.shenfun` using

.. code-block:: python

    ux = project(Dx(u_hat, 0, 1), T)

The following code, on the other hand, will raise an error since you cannot
take the derivative of an interpolated ``Array u``, only a ``Function``

.. code-block:: python

    try:
        project(Dx(u, 0, 1), T)
    except AssertionError:
        print("AssertionError: Dx not for Arrays")

Initialization
--------------

The solution array ``uf`` and its transform ``uf_hat`` need to be initialized according to Eq.
:eq:`eq:init`. To this end it is convenient (but not required, we could just as
easily use Numpy for this as well) to use `Sympy <http://www.sympy.org/en/index.html>`__, which is a Python library for symbolic
mathamatics.

.. code-block:: python

    from sympy import symbols, exp, lambdify
    
    x, y, z = symbols("x,y,z")
    ue = 0.1*exp(-(x**2 + y**2 + z**2))
    ul = lambdify((x, y, z), ue, 'numpy')
    X = T.local_mesh(True)
    u[:] = Array(T, buffer=ul(*X))
    u_hat = T.forward(u, u_hat)

Here ``X`` is a list of the three mesh coordinates ``(x, y, z)`` local to the
current processor. Each processor has its own part of the computational mesh,
and the distribution is handled during the creation of the
:class:`.TensorProductSpace`
class instance ``T``. There is no need
to do anything about the ``f/f_hat`` arrays since they are already initialized by default to
zero. Note that calling the ``ul`` function with the argument ``*X`` is the same as
calling with ``X[0], X[1], X[2]``.

.. _sec:rk:

Runge-Kutta integrator
----------------------
A fourth order explicit Runge-Kutta integrator requires only a function that
returns the right hand sides of :eq:`eq:df_var3` and :eq:`eq:kg:du_var3`. Such a
function can be implemented as

.. code-block:: python

    # focusing (+1) or defocusing (-1)
    gamma = 1
    uh = TrialFunction(T)
    vh = TestFunction(T)
    k2 = -(inner(grad(vh), grad(uh))  + gamma)
    
    def compute_rhs(duf_hat, uf_hat, up, Tp, w0):
        duf_hat.fill(0)
        u_hat, f_hat = uf_hat[:]
        du_hat, df_hat = duf_hat[:]
        df_hat[:] = k2*u_hat
        up = Tp.backward(u_hat, up)
        df_hat += Tp.forward(gamma*up**3, w0)
        du_hat[:] = f_hat
        return duf_hat

The code is fairly self-explanatory. ``k2`` represents the coefficients in front of
the linear :math:`\hat{u}` in :eq:`eq:df_var3`. The output array is ``duf_hat``, and
the input array is ``uf_hat``, whereas ``up`` and ``w0`` are work arrays. The array
``duf_hat`` contains the right hand sides of both :eq:`eq:df_var3` and
:eq:`eq:kg:du_var3`, where the linear and nonlinear terms are recognized in the
code as comments ``(1)`` and ``(2)``.
The array ``uf_hat`` contains the solution at initial and intermediate Runge-Kutta steps.

With a function that returns the right hand side in place, the actual integrator
can be implemented as

.. code-block:: python

    w0 = Function(T)
    a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
    b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
    t = 0
    dt = 0.01
    end_time = 1.0
    while t < end_time-1e-8:
        t += dt
        uf_hat1[:] = uf_hat0[:] = uf_hat
        for rk in range(4):
            duf = compute_rhs(duf, uf_hat, u, T, w0)
            if rk < 3:
                uf_hat[:] = uf_hat0 + b[rk]*dt*duf
            uf_hat1 += a[rk]*dt*duf
        uf_hat[:] = uf_hat1

.. _sec:solver:

Complete solver
---------------

A complete solver is given below, with intermediate plotting of the solution and
intermediate computation of the total energy. Note that the total energy is unchanged to 8
decimal points at :math:`t=100`.

.. code-block:: python

    from sympy import symbols, exp, lambdify
    import numpy as np
    import matplotlib.pyplot as plt
    from mpi4py import MPI
    from time import time
    from shenfun import *
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Use sympy to set up initial condition
    x, y, z = symbols("x,y,z")
    ue = 0.1*exp(-(x**2 + y**2 + z**2))
    ul = lambdify((x, y, z), ue, 'numpy')
    
    # Size of discretization
    N = (64, 64, 64)
    
    # Defocusing or focusing
    gamma = 1
    
    K0 = Basis(N[0], 'F', domain=(-2*np.pi, 2*np.pi), dtype='D')
    K1 = Basis(N[1], 'F', domain=(-2*np.pi, 2*np.pi), dtype='D')
    K2 = Basis(N[2], 'F', domain=(-2*np.pi, 2*np.pi), dtype='d')
    T = TensorProductSpace(comm, (K0, K1, K2), slab=False,
                           **{'planner_effort': 'FFTW_MEASURE'})
    
    TT = MixedTensorProductSpace([T, T])
    
    X = T.local_mesh(True)
    uf = Array(TT)
    u, f = uf[:]
    up = Array(T)
    duf = Function(TT)
    du, df = duf[:]
    
    uf_hat = Function(TT)
    uf_hat0 = Function(TT)
    uf_hat1 = Function(TT)
    w0 = Function(T)
    u_hat, f_hat = uf_hat[:]
    
    # initialize (f initialized to zero, so all set)
    u[:] = ul(*X)
    u_hat = T.forward(u, u_hat)
    
    uh = TrialFunction(T)
    vh = TestFunction(T)
    k2 = -inner(grad(vh), grad(uh)) - gamma
    
    count = 0
    def compute_rhs(duf_hat, uf_hat, up, T, w0):
        global count
        count += 1
        duf_hat.fill(0)
        u_hat, f_hat = uf_hat[:]
        du_hat, df_hat = duf_hat[:]
        df_hat[:] = k2*u_hat
        up = T.backward(u_hat, up)
        df_hat += T.forward(gamma*up**3, w0)
        du_hat[:] = f_hat
        return duf_hat
    
    def energy_fourier(comm, a):
        result = 2*np.sum(abs(a[..., 1:-1])**2) + np.sum(abs(a[..., 0])**2) + np.sum(abs(a[..., -1])**2)
        result =  comm.allreduce(result)
        return result
    
    # Integrate using a 4th order Rung-Kutta method
    a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
    b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
    t = 0.0
    dt = 0.005
    end_time = 1.
    tstep = 0
    if rank == 0:
        plt.figure()
        image = plt.contourf(X[1][..., 0], X[0][..., 0], u[..., 16], 100)
        plt.draw()
        plt.pause(1e-4)
    t0 = time()
    K = np.array(T.local_wavenumbers(True, True, True))
    TV = VectorTensorProductSpace([T, T, T])
    gradu = Array(TV)
    while t < end_time-1e-8:
        t += dt
        tstep += 1
        uf_hat1[:] = uf_hat0[:] = uf_hat
        for rk in range(4):
            duf = compute_rhs(duf, uf_hat, up, T, w0)
            if rk < 3:
                uf_hat[:] = uf_hat0 + b[rk]*dt*duf
            uf_hat1 += a[rk]*dt*duf
        uf_hat[:] = uf_hat1
    
        if tstep % 100 == 0:
            uf = TT.backward(uf_hat, uf)
            ekin = 0.5*energy_fourier(T.comm, f_hat)
            es = 0.5*energy_fourier(T.comm, 1j*K*u_hat)
            eg = gamma*np.sum(0.5*u**2 - 0.25*u**4)/np.prod(np.array(N))
            eg =  comm.allreduce(eg)
            gradu = TV.backward(1j*K*u_hat, gradu)
            ep = comm.allreduce(np.sum(f*gradu)/np.prod(np.array(N)))
            ea = comm.allreduce(np.sum(np.array(X)*(0.5*f**2 + 0.5*gradu**2
                                - (0.5*u**2 - 0.25*u**4)*f))/np.prod(np.array(N)))
            if rank == 0:
                image.ax.clear()
                image.ax.contourf(X[1][..., 0], X[0][..., 0], u[..., 16], 100)
                plt.pause(1e-6)
                plt.savefig('Klein_Gordon_{}_real_{}.png'.format(N[0], tstep))
                print("Time = %2.2f Total energy = %2.8e Linear momentum %2.8e Angular momentum %2.8e" %(t, ekin+es+eg, ep, ea))
            comm.barrier()
    
    print("Time ", time()-t0)

.. ======= Bibliography =======

.. bibliography:: papers.bib
   :notcited:
