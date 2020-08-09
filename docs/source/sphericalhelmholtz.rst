.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - Helmholtz equation on the unit sphere
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:Authors: Mikael Mortensen (mikaem at math.uio.no)
<<<<<<< HEAD
:Date: Aug 9, 2020
=======
:Date: Aug 7, 2020
>>>>>>> 03ba55962b465612916ae3935e510b020f767c75

*Summary.* This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve the
Helmholtz equation on a unit sphere, using spherical
coordinates. This demo is implemented in
a single Python file `sphere_helmholtz.py <https://github.com/spectralDNS/shenfun/blob/master/demo/sphere_helmholtz.py>`__.
If requested the solver will run in parallel using MPI.

.. _fig:helmholtz:

.. figure:: https://rawgit.com/spectralDNS/spectralutilities/master/figures/spherewhite4.png
   :width: 700

   *Helmholtz on the unit sphere*

.. _demo:spherical_helmholtz:

Helmholtz equation
==================

The Helmholtz equation is given as

.. math::
   :label: eq:helmholtz

        
        -\nabla^2 u(\boldsymbol{x}) + \alpha u(\boldsymbol{x}) = f(\boldsymbol{x}) \quad \text{for }\, \boldsymbol{x} \in \Omega = \{(x, y, z): x^2+y^2+z^2 = 1\}, 
        

.. math::
   :label: _auto1

          
        
        

where :math:`u(\boldsymbol{x})` is the solution, :math:`f(\boldsymbol{x})` is a function and :math:`\alpha` a constant.
We use spherical coordinates :math:`(\theta, \phi)`, defined as

.. math::
   :label: _auto2

        
         x = r \sin \theta \cos \phi , 
        
        

.. math::
   :label: _auto3

          
         y = r \sin \theta \sin \phi, 
        
        

.. math::
   :label: _auto4

          
         z = r \cos \theta
        
        

which (with :math:`r=1`) leads to a 2D Cartesian product mesh :math:`(\theta, \phi) \in (0, \pi) \times [0, 2\pi)`
suitable for numerical implementations. There are no boundary
conditions on the problem under consideration.
However, with the chosen Cartesian mesh, periodic
boundary conditions are required for the :math:`\phi`-direction. As such,
the :math:`\phi`-direction will use a Fourier basis :math:`\exp(\imath k \phi)`.

A regular Chebyshev or Legendre basis
:math:`\psi_j(\theta) = \gamma_j(2\theta/\pi-1)` will be
used for the :math:`\theta`-direction, where :math:`\gamma_j` could be either
the Chebyshev polynomial of first kind :math:`T_j` or the Legendre
polynomial :math:`L_j`. Note the mapping from real coordinates :math:`\theta`
to computational coordinates in domain :math:`[-1, 1]`.

The spherical basis functions are as such

.. math::
        v_{jk}(\theta, \phi) = \psi_j(\theta) \exp(\imath k \phi),

and we look for solutions

.. math::
        u(\theta, \phi) = \sum_{j} \sum_{k} \hat{u}_{jk} v_{jk}(\theta, \phi).

A discrete Fourier approximation space with :math:`N` basis functions is then

.. math::
        V_F^N = \text{span} \{\exp(\imath k \theta)\}, \text{ for } k \in K,

where :math:`K = \{-N/2, -N/2+1, \ldots, N/2-1\}`. For this demo we assume
that the solution is complex, and as such there is no simplification
possible for Hermitian symmetry.

The following approximation space is used for the :math:`\theta`-direction

.. math::
   :label: _auto5

        
        V^N = \text{span} \{\psi_j\}_{j=0}^{N-1} 
        
        

.. math::
   :label: _auto6

          
        
        

and the variational formulation of the problem reads:
find :math:`u \in V^N \otimes V_F^N` such that

.. math::
   :label: eq:u0

           
           \int_{\Omega} (-\nabla^2 u + \alpha u) v w d\sigma = \int_{\Omega} f v w d\sigma, \quad \forall \, v \in V^N \otimes V_F^N.
        
           

Note that integration over the domain is done using
spherical coordinates with an integral measure of :math:`d\sigma=\sin \theta d\theta d\phi`.

.. _demo:sphericalimplementation:

Implementation in shenfun
=========================

A complete implementation is found in the file `sphere_helmholtz.py <https://github.com/spectralDNS/shenfun/blob/master/demo/sphere_helmholtz.py>`__.
Here we give a brief explanation for the implementation. Start by
importing all functionality from `shenfun <https://github.com/spectralDNS/shenfun>`__
and `sympy <https://sympy.org>`__, where Sympy is required for handeling the
spherical coordinates.

.. code-block:: python

    from shenfun import *
    import sympy as sp
    
    # Define spherical coordinates with unit radius
    r = 1
    theta, phi = sp.symbols('x,y', real=True, positive=True)
    psi = (theta, phi)
    rv = (r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta))

Note that the position vector ``rv`` has three components (for :math:`(x, y, z)`)
even though the computational domain is only 2D.
Also note that Sympy symbols are both positive and real, and :math:`\theta` is
chosen to be along the first axis and :math:`\phi` second. This has to agree with
the next step, which is the creation of tensorproductspaces
:math:`V^N \otimes V_F^N`.

.. code-block:: python

    N, M = 40, 30
    L0 = FunctionSpace(N, 'C', domain=(0, np.pi))
    F1 = FunctionSpace(M, 'F', dtype='D')
    T = TensorProductSpace(comm, (L0, F1), coordinates=(psi, rv))
    

Spherical coordinates are ensured by feeding ``coordinates=(psi, rv)``
to :class:`.TensorProductSpace`. Operators like :func:`.div`
:func:`.grad` and  :func:`.curl` will now work on
items of :class:`.Function`, :class:`.TestFunction` and
:class:`.TrialFunction` using a spherical coordinate system.

To define the equation :eq:`eq:u0` we first declare
these test- and trialfunctions, and then use code that
is very similar to the mathematics.

.. code-block:: python

    alpha = 2
    v = TestFunction(T)
    u = TrialFunction(T)
    
    mats = inner(v, -div(grad(u))+alpha*u)

Here ``mats`` will be a list containing several tensor product
matrices in the form of
:class:`.TPMatrix`. Since there is only one directions with
non-diagonal matrices (:math:`\theta`-direction) we
can use the generic :class:`.SolverGeneric1ND` solver.
Note that some of the non-diagonal matrices will be dense,
which is a weakness of the current method. Also note
that with Legendre one can use integration by parts
instead

.. code-block:: python

    mats = inner(grad(v), grad(u))
    mats += [inner(v, alpha*u)]

To solve the problem we also need to define the function :math:`f(\theta, r)`.
To this end we use sympy and the method of
manufactured solution to define a possible solution ``ue``,
and then compute ``f`` exactly using exact differentiation. We use
the `spherical harmonics function <https://docs.sympy.org/latest/modules/functions/special.html#spherical-harmonics>`__
to define an analytical solution

.. code-block:: python

    # Manufactured solution
    alpha = 2
    sph = sp.functions.special.spherical_harmonics.Ynm
    ue = sph(6, 3, theta, phi)
    
    # Compute the right hand side on the quadrature mesh
    # That is, compute f = -div(grad(ue)) + alpha*ue
    f = (-div(grad(u))+alpha*u).tosympy(basis=ue, psi=psi)
    fj = Array(T, buffer=f)
    
    # Take scalar product
    f_hat = Function(T)
    f_hat = inner(v, fj, output_array=f_hat)
    
    u_hat = Function(T)
    Sol = SolverGeneric1ND(mats)
    u_hat = Sol(f_hat, u_hat)

Having found the solution in spectral space all that is
left is to transform it back to real space.

.. code-block:: python

    uj = u_hat.backward()
    uq = Array(T, buffer=ue)
    print('Error =', np.linalg.norm(uj-uq))

Leading to

.. code-block:: python

    Error = 8.383877617440085e-10

Postprocessing
==============
We can refine the solution to make it look better,
and plot on the unit sphere using `mayavi <https://docs.enthought.com/mayavi/mayavi/>`__,
leading to Figure :ref:`fig:helmholtz`.

.. code-block:: text

    u_hat2 = u_hat.refine([N*2, M*2])
    ur = u_hat2.backward()
    from mayavi import mlab
    xx, yy, zz = u_hat2.function_space().local_curvilinear_mesh()
    # Wrap periodic direction around
    if T.bases[1].domain == (0, 2*np.pi):
        xx = np.hstack([xx, xx[:, 0][:, None]])
        yy = np.hstack([yy, yy[:, 0][:, None]])
        zz = np.hstack([zz, zz[:, 0][:, None]])
        ur = np.hstack([ur, ur[:, 0][:, None]])
    mlab.mesh(xx, yy, zz, scalars=ur.imag, colormap='jet')
    mlab.show()

.. ======= Bibliography =======
