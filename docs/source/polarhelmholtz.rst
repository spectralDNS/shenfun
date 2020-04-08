.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - Helmholtz equation in polar coordinates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:Authors: Mikael Mortensen (mikaem at math.uio.no)
:Date: Apr 8, 2020

*Summary.* This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve the
Helmholtz equation on a circular disc, using polar coordinates. This demo is implemented in
a single Python file `unitdisc_poisson.py <https://github.com/spectralDNS/shenfun/blob/master/demo/unitdisc_poisson.py>`__,
and the numerical method is is described in more detail by J. Shen :cite:`shen3`.

.. _fig:helmholtz:

.. figure:: https://rawgit.com/spectralDNS/spectralutilities/master/figures/Helmholtzdisc.png
   :width: 700

   *Poisson on the unit disc*

.. _demo:polar_helmholtz:

Helmholtz equation
==================

The Helmholtz equation is given as

.. math::
   :label: eq:helmholtz

        
        -\nabla^2 u(\boldsymbol{x}) + \alpha u(\boldsymbol{x}) = f(\boldsymbol{x}) \quad \text{for }\, \boldsymbol{x}=(x, y) \in \Omega, 
        

.. math::
   :label: _auto1

          
        u =0 \text{ on } \partial \Omega,
        
        

where :math:`u(\boldsymbol{x})` is the solution, :math:`f(\boldsymbol{x})` is a function and :math:`\alpha` a constant.
The domain is a circular disc :math:`\Omega = \{(x, y): x^2+y^2 < a^2\}` with radius :math:`a`.
We use polar coordinates :math:`(\theta, r)`, defined as

.. math::
   :label: _auto2

        
         x = r \cos \theta, 
        
        

.. math::
   :label: _auto3

          
         y = r \sin \theta,
        
        

which gives us a Cartesian product mesh :math:`(\theta, r) \in [0, 2\pi) \times [0, a]`
suitable for a numerical implementation. Note that we order the
two directions with :math:`\theta` first and then :math:`r`, which is less common
than :math:`(r, theta)`. This has to do with the fact that we will need to
solve linear equation systems along the radial direction, but not
the :math:`\theta`-direction, since Fourier matrices are diagonal. When
the radial direction is placed last, the data in the radial direction
will be contigeous in a row-major C memory, leading to faster memory
access where it is needed the most. Note that it takes very few
changes in ``shenfun`` to switch the directions to :math:`(r, \theta)` if this
is still desired.

We will use Chebyshev
or Legendre basis functions :math:`\psi_j(r)` for the radial direction and a periodic Fourier expansion
in :math:`\exp(\imath k \theta)` for the azimuthal direction. The polar basis functions are as such

.. math::
        v_{kj}(\theta, r) = \exp(\imath k \theta) \psi_j(r),

and we look for solutions

.. math::
        u(\theta, r) = \sum_{k} \sum_{j} \hat{u}_{kj} v_{kj}(\theta, r).

A discrete Fourier approximation space with :math:`N` basis functions is then

.. math::
        V_F^N = \text{span} \{\exp(\imath k \theta)\}, \text{ for } k \in K,

where :math:`K = \{-N/2, -N/2+1, \ldots, N/2-1\}`. Since the solution :math:`u(\theta, r)`
is real, we have Hermitian symmetry and :math:`\hat{u}_{k,j} = \hat{u}_{k,-j}^*`
(with :math:`*` denoting a complex conjugate).
For this reason we use only :math:`k \in K=\{0, 1, \ldots, N/2\}` in solving for
:math:`\hat{u}_{kj}`, and then use Hermitian symmetry to get the remaining
unknowns.

The radial basis is more tricky, because there is a nontrivial 'boundary'
condition (pole condition) that needs to be applied at the center of the disc :math:`(r=0)`

.. math::
        \frac{\partial u(\theta, 0)}{\partial \theta} = 0.

To apply this condition we split the solution into Fourier
coefficients with wavenumber 0 and :math:`K\backslash \{0\}`,
remembering that the Fourier basis function with :math:`k=0` is
simply 1

.. math::
        u(\theta, r) = \sum_{j} \left( \hat{u}_{0j} \psi_{j}(r) + \sum_{k=1}^{N/2} \hat{u}_{kj} \exp(\imath k \theta) \psi_j(r) \right).

We then apply a different radial basis for the two :math:`\psi`'s in the above equation
(renaming the first :math:`\overline{\psi}`)

.. math::
        u(\theta, r) = \sum_{j} \left( \hat{u}_{0j} \overline{\psi}_{j}(r) + \sum_{k=1}^{N/2} \hat{u}_{kj} \exp(\imath k \theta) \psi_j(r) \right).

Note that the first term :math:`\sum_{j} \hat{u}_{0j} \overline{\psi}_{j}(r)` is independent
of :math:`\theta`. Now, to enforce conditions

.. math::
   :label: _auto4

        
        u(\theta, a) = 0, 
        
        

.. math::
   :label: _auto5

          
        \frac{\partial u(\theta, 0)}{\partial \theta} = 0,
        
        

we see that it is sufficient for the two bases
(:math:`\overline{\psi}` and :math:`\psi`) to satisfy

.. math::
   :label: _auto6

        
        \overline{\psi}_j(a) = 0, 
        
        

.. math::
   :label: _auto7

          
        \psi_j(a) = 0,
        
        

.. math::
   :label: _auto8

          
        \psi_j(0) = 0.
        
        

We can find such bases both with Legendre and Chebyshev polynomials.
If we let :math:`\phi_j(x)` mean either the Legendre polynomial :math:`L_j(x)` or the
Chebyshev polynomial of the first kind :math:`T_j(x)`, we can have

.. math::
   :label: _auto9

        
        \overline{\psi}_j(r) = \phi_j(2r-1) - \phi_{j+1}(2r-1), \text{ for } j \in 0, 1, \ldots N-1, 
        
        

.. math::
   :label: eq:psi

          
        \psi_j(r) = \phi_j(2r-1) - \phi_{j+2}(2r-1), \text{ for } j \in 0, 1, \ldots N-2.
        
        

Define the following approximation spaces for the radial direction

.. math::
   :label: _auto10

        
        V_D^N = \text{span} \{\psi_j\}_{j=0}^{N-2} 
        
        

.. math::
   :label: _auto11

          
        V_U^N = \text{span} \{\overline{\psi}_j\}_{j=0}^{N-1} 
        
        

.. math::
   :label: _auto12

          
        
        

and split the function space for the azimuthal direction into

.. math::
   :label: _auto13

        
        V_F^0 =  \text{span}\{1\}, 
        
        

.. math::
   :label: _auto14

          
        V_F^{1} = \text{span} \{\exp(\imath k \theta)\}, \text{ for } k \in K \backslash \{0\}.
        
        

We then look for solutions

.. math::
   :label: _auto15

        
        u(\theta, r) = u^0(r) + u^1(\theta, r),
        
        

where

.. math::
   :label: _auto16

        
        u^0(r) = \sum_{j=0}^{N-1} \hat{u}^0_j \overline{\psi}_j(r), 
        
        

.. math::
   :label: _auto17

          
        u^1(\theta, r) = \sum_{j=0}^{N-2}\sum_{k=1}^{N/2} \hat{u}^1_{kj} \exp(\imath k \theta) \psi_j(r) .
        
        

As such the Helmholtz problem is split in two smaller problems.
The two problems read with the spectral Galerkin method:

Find :math:`u^0 \in V_F^0 \otimes V_U^N` such that

.. math::
   :label: eq:u0

           
           \int_{\Omega} (-\nabla^2 u^0 + \alpha u^0) v^0 w r dr d\theta = \int_{\Omega} f v^0 w r dr d\theta, \quad \forall \, v^0 \in V_F^0 \otimes V_U^N.
        
           

Find :math:`u^1 \in V_F^1 \otimes V_D^N` such that

.. math::
   :label: eq:u1

           
           \int_{\Omega} (-\nabla^2 u^1 + \alpha u^1) v^1 w r dr d\theta = \int_{\Omega} f v^1 w r dr d\theta, \quad \forall \, v^1 \in V_F^1 \otimes V_D^N.
        
           

Note that integration over the domain is done using
polar coordinates with an integral measure of :math:`rdrd\theta`.
However, the integral in the radial direction needs to be mapped
to :math:`t=2r-1`, where :math:`t \in [-1, 1]` suits the basis functions used,
see :eq:`eq:psi`. This leads to a measure of :math:`0.5(t+1)dtd\theta`.
Furthermore, the weight :math:`w(t)` will be unity for the Legendre basis and
:math:`(1-t^2)^{-0.5}` for the Chebyshev bases.

.. _demo:polarimplementation:

Implementation in shenfun
=========================

A complete implementation is found in the file `unitdisc_poisson.py <https://github.com/spectralDNS/shenfun/blob/master/demo/unitdisc_poisson.py>`__.
Here we give a brief explanation for the implementation. Start by
importing all functionality from `shenfun <https://github.com/spectralDNS/shenfun>`__
and `sympy <https://sympy.org>`__, where Sympy is required for handeling the
polar coordinates.

.. code-block:: python

    from shenfun import *
    import sympy as sp
    
    # Define polar coordinates using angle along first axis and radius second
    theta, r = psi = sp.symbols('x,y', real=True, positive=True)
    rv = (r*sp.cos(theta), r*sp.sin(theta)) # Map to Cartesian (x, y)

Note that Sympy symbols are both positive and real, :math:`\theta` is
chosen to be along the first axis and :math:`r` second. This has to agree with
the next step, which is the creation of tensorproductspaces
:math:`V_F^0 \otimes V_U^N` and :math:`V_F^1 \otimes V_D^N`

.. code-block:: python

    N = 16
    F = Basis(N, 'F', dtype='d')
    F0 = Basis(1, 'F', dtype='d')
    L = Basis(N, 'L', bc='Dirichlet', domain=(0, 1))
    L0 = Basis(N, 'L', bc='UpperDirichlet', domain=(0, 1))
    T = TensorProductSpace(comm, (F, L), axes=(1, 0), measures=(psi, rv))
    T0 = TensorProductSpace(MPI.COMM_SELF, (F0, L0), axes=(1, 0), measures=(psi, rv))

Note that since ``F0`` only has one component we could actually use
``L0`` without creating ``T0``. But the code turns out to be simpler
if we use ``T0``, much because the additional :math:`\theta`-direction is
required for the polar coordinates to apply.
Also note that ``F`` is created using the entire range of wavenumber
and as such we need to make sure that the coefficient created for
:math:`k=0` ) (i.e., :math:`\hat{u}^1_{0,j}`) will be exactly zero.
Finally, note that
``T0`` is not distributed with MPI, which is accomplished using
``MPI.COMM_SELF`` instead of ``comm`` (which equals ``MPI.COMM_WORLD``).
The small problem :eq:`eq:u0` is only solved on the one processor
with rank = 0.

Note that polar coordinates are ensured feeding ``measures=(psi, rv)``
to :class:`.TensorProductSpace`. Operators like :func:`.div`
:func:`.grad` and  :func:`.curl` will now work on
items of :class:`.Function`, :class:`.TestFunction` and
:class:`.TrialFunction` using a polar coordinate system.

To define the equations :eq:`eq:u0` and :eq:`eq:u1` we first declare
these test- and trialfunctions, and then use code that
is remarkably similar to the mathematics.

.. code-block:: python

    v = TestFunction(T)
    u = TrialFunction(T)
    v0 = TestFunction(T0)
    u0 = TrialFunction(T0)
    
    mats = inner(v, -div(grad(u))+alpha*u)
    if comm.Get_rank() == 0:
        mats0 = inner(v0, -div(grad(u0))+alpha*u0)

Here ``mats`` and ``mats0`` will contain several matrices in the form of
:class:`.TPMatrix`. Since there is only one non-periodic direction
the matrices can be easily solved using :class:`.SolverGeneric1NP`.
But first we need to define the function :math:`f(\theta, r)`, such
that we have something to solve. We use sympy and the method of
manufactured solution to define a possible solution ``ue``,
and then compute ``f`` exactly using exact differentiation

.. code-block:: python

    # Manufactured solution
    alpha = 2
    ue = (r*(1-r))**2*sp.cos(8*theta)-0.1*(r-1)
    f = -ue.diff(r, 2) - (1/r)*ue.diff(r, 1) - (1/r**2)*ue.diff(theta, 2) + alpha*ue
    
    # Compute the right hand side on the quadrature mesh
    fj = Array(T, buffer=f)
    
    # Take scalar product
    f_hat = Function(T)
    f_hat = inner(v, fj, output_array=f_hat)
    if T.local_slice(True)[0].start == 0: # The processor that owns k=0
        f_hat[0] = 0
    
    # For k=0 we solve only a 1D equation. Do the scalar product for Fourier
    # coefficient 0 by hand (or sympy).
    if comm.Get_rank() == 0:
        f0_hat = Function(T0)
        gt = sp.lambdify(r, sp.integrate(f, (theta, 0, 2*sp.pi))/2/sp.pi)(L0.mesh())
        f0_hat = L0.scalar_product(gt, f0_hat)

Note that for :math:`u^0` we perform the interal in the :math:`\theta` direction
exactly using sympy. This is necessary since one Fourier coefficient
is not sufficient to do this integral numerically. For the :math:`u^1`
case we do the integral numerically as part of the ``inner`` product.
With the correct right hand side assembled we can solve the
linear system of equations

.. code-block:: python

    u_hat = Function(T)
    Sol1 = SolverGeneric1NP(mats)
    u_hat = Sol1(f_hat, u_hat)
    
    # case k = 0
    u0_hat = Function(T0)
    if comm.Get_rank() == 0:
        Sol0 = SolverGeneric1NP(mats0)
        u0_hat = Sol0(f0_hat, u0_hat)
    comm.Bcast(u0_hat, root=0)

Having found the solution in spectral space all that is
left is to transform it back to real space.

.. code-block:: python

    # Transform back to real space. Broadcast 1D solution
    sl = T.local_slice(False)
    uj = u_hat.backward() + u0_hat.backward()[:, sl[1]]

Postprocessing
==============
The solution can now be compared with the exact solution
through

.. code-block:: python

    ue = Array(T, buffer=ue)
    X = T.local_mesh(True)
    print('Error =', np.linalg.norm(uj-ue))

And we can refine the solution to make it look better,
and plot on the unit disc, leading to Figure :ref:`fig:helmholtz`.

.. code-block:: python

    u_hat2 = u_hat.refine([N*3, N*3])
    u0_hat2 = u0_hat.refine([1, N*3])
    sl = u_hat2.function_space().local_slice(False)
    ur = u_hat2.backward() + u0_hat2.backward()[:, sl[1]]
    Y = u_hat2.function_space().local_mesh(True)
    thetaj, rj = Y[0], Y[1]
    
    # Wrap periodic plot around since it looks nicer
    xx, yy = rj*np.cos(thetaj), rj*np.sin(thetaj)
    xp = np.vstack([xx, xx[0]])
    yp = np.vstack([yy, yy[0]])
    up = np.vstack([ur, ur[0]])
    
    # plot
    plt.figure()
    plt.contourf(xp, yp, up)
    plt.colorbar()
    plt.title('Helmholtz - unitdisc')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

.. ======= Bibliography =======
