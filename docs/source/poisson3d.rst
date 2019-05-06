.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - 3D Poisson equation
%%%%%%%%%%%%%%%%%%%%%%%%%%

:Authors: Mikael Mortensen (mikaem at math.uio.no)
:Date: Apr 26, 2019

*Summary.* This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve a 3D Poisson
equation in a 3D tensor product domain that has homogeneous Dirichlet boundary
conditions in one direction and periodicity in the
remaining two. The solver described runs with MPI without any further
considerations required from the user. Spectral convergence, as shown in Figure :ref:`fig:3d:ct0`, is demonstrated.
The demo is implemented in
a single Python file `dirichlet_poisson3D.py <https://github.com/spectralDNS/shenfun/blob/master/demo/dirichlet_poisson3D.py>`__, and the numerical method is is described in more detail by J. Shen :cite:`shen1` and :cite:`shen95`.

.. _fig:3d:ct0:

.. figure:: https://rawgit.com/spectralDNS/spectralutilities/master/figures/poisson3D_errornorm.png

   *Convergence of 3D Poisson solvers for both Legendre and Chebyshev modified basis function*

Model problem
=============

.. _demo:poisson3d:

Poisson equation
----------------

The Poisson equation is given as

.. math::
   :label: eq:3d:poisson

        
        \nabla^2 u(\boldsymbol{x}) = f(\boldsymbol{x}) \quad \text{for }\, \boldsymbol{x}=(x, y, z) \in \Omega, 
        

.. math::
   :label: _auto1

          
        u(\pm 1 ,y, z) =0, 
        
        

.. math::
   :label: _auto2

          
        u(x, 2\pi, z) = u(x, 0, z), 
        
        

.. math::
   :label: _auto3

          
        u(x, y, 2\pi) = u(x, y, 0),
        
        

where :math:`u(\boldsymbol{x})` is the solution and :math:`f(\boldsymbol{x})` is a function. The domain
:math:`\Omega = [-1, 1]\times [0, 2\pi]^2`.

To solve Eq. :eq:`eq:3d:poisson` with the Galerkin method we need smooth basis
functions, :math:`v(\boldsymbol{x})`, that live
in the Hilbert space :math:`H^1(\Omega)` and that satisfy the given boundary
conditions. To this end we will use one basis function for the :math:`x`-direction,
:math:`\mathcal{X}(x)`,
one for the :math:`y`-direction, :math:`\mathcal{Y}(y)`, and one for the :math:`z`-direction,
:math:`\mathcal{Z}(z)`. And
then we create three-dimensional basis functions like

.. math::
        v(x, y, z) = \mathcal{X}(x) \mathcal{Y}(y) \mathcal{Z} (z).

The basis functions :math:`\mathcal{Y}(y)` and :math:`\mathcal{Z}(z)` are chosen as Fourier exponentials, since these
functions are periodic. Likewise, the basis functions :math:`\mathcal{X}(x)` are chosen as
modified Legendre or Chebyshev polynomials, using :math:`\phi_l(x)` to refer to either
one

.. math::
   :label: _auto4

        
        \mathcal{X}_l(x) = \phi_l(x) - \phi_{l+2}(x), \forall \, l \in \boldsymbol{l}^{N_0},
        
        

.. math::
   :label: _auto5

          
        \mathcal{Y}_m(y) =  e^{\imath m y}, \forall \, m \in \boldsymbol{m}^{N_1}, 
        
        

.. math::
   :label: _auto6

          
        \mathcal{Z}_n(z) = e^{\imath n z}, \forall \, n \in \boldsymbol{n}^{N_2},
        
        

where the size of the discretized problem is :math:`\boldsymbol{N} = (N_0, N_1, N_2)`,
:math:`\boldsymbol{l}^{N_0} = (0, 1, \ldots, N_0-3)`, :math:`\boldsymbol{m}^{N_1} =
(-N_1/2, -N_1/2+1, \ldots, N_1/2-1)` and :math:`\boldsymbol{n}^{N_2} = (-N_2/2, -N_2/2+1,
\ldots, N_2/2-1)`. However, due to `Hermitian symmetry <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft>`__, we only store :math:`N_2/2+1`
wavenumbers in the :math:`z`-direction, such that :math:`\boldsymbol{n}^{N_2} = (0, 1, \ldots,
N_2/2)`. We refer to the Cartesian wavenumber mesh on vector form as :math:`\boldsymbol{k}`:

.. math::
        \boldsymbol{k} = \{(l, m, n)  \in \boldsymbol{l}^{N_0} \times \boldsymbol{m}^{N_1} \times \boldsymbol{n}^{N_2}\}.

We have the bases

.. math::
   :label: _auto7

        
        V^{N_0} = \text{span}\{ \mathcal{X}_l \}_{l\in\boldsymbol{l}^{N_0}}, 
        
        

.. math::
   :label: _auto8

          
        V^{N_1} = \text{span}\{ \mathcal{Y}_m \}_{m\in\boldsymbol{m}^{N_1}}, 
        
        

.. math::
   :label: _auto9

          
        V^{N_2} = \text{span}\{ \mathcal{Z}_n \}_{n\in\boldsymbol{n}^{N_2}},
        
        

and from these we create a tensor product space :math:`W^{\boldsymbol{N}}(\boldsymbol{x})` from
Cartesian products of the three bases

.. math::
   :label: _auto10

        
        W^{\boldsymbol{N}}(\boldsymbol{x}) = V^{N_0}(x) \times V^{N_1}(y) \times V^{N_2}(z).
        
        

And then we look for discrete solutions :math:`u \in W^{\boldsymbol{N}}` like

.. math::
   :label: eq:3d:u

        
        u(\boldsymbol{x}) = \sum_{l\in \boldsymbol{l}^{N_0}} \sum_{m\in \boldsymbol{m}^{N_1}}\sum_{n\in
        \boldsymbol{n}^{N_2}}\hat{u}_{l,m,n} \mathcal{X}_l(x) \mathcal{Y}_m(y) \mathcal{Z}_n(z),  
        

.. math::
   :label: _auto11

          
         = \sum_{\boldsymbol{\textsf{k}} \in \boldsymbol{k}}\hat{u}_{\boldsymbol{\textsf{k}}} v_{\boldsymbol{\textsf{k}}}(\boldsymbol{x}),
        
        

where :math:`\hat{u}_{l,m,n}` are components of the expansion coefficients for :math:`u` and
the second form, :math:`\{\hat{u}_{\boldsymbol{\textsf{k}}}\}_{\boldsymbol{\textsf{k}}\in\boldsymbol{k}}`, is a shorter,
simplified notation, with sans-serif :math:`\boldsymbol{\textsf{k}}=(l, m, n)`.
The expansion coefficients are the unknowns in the spectral Galerkin method.

We now formulate a variational problem using the Galerkin method: Find :math:`u \in
W^{\boldsymbol{N}}` such that

.. math::
   :label: eq:3d:varform

        
        \int_{\Omega} \nabla^2 u \, \overline{v} \, w\, \boldsymbol{dx} = \int_{\Omega} f \,
        \overline{v}\, w\, \boldsymbol{dx} \quad
        \forall v \, \in \, W^{\boldsymbol{N}}. 
        

Here :math:`\boldsymbol{dx}=dxdydz`, and the overline represents a complex conjugate, which is needed here because
the Fourier exponentials are complex functions.
The weighted integrals, weighted by :math:`w(\boldsymbol{x})`, are called inner products, and a common notation is

.. math::
   :label: _auto12

        
        \int_{\Omega} u \, \overline{v} \, w\, \boldsymbol{dx} = \langle u, v\rangle _w.
        
        

The integral can either be computed exactly, or with quadrature. The advantage
of the latter is that it is generally faster, and that non-linear terms may be
computed just as quickly as linear. For a linear problem, it does not make much of a difference, if any at all. Approximating the integral with quadrature, we obtain

.. math::
   :label: _auto13

        
        \int_{\Omega} u \, \overline{v} \, w\, \boldsymbol{dx} \approx \langle u, v
        \rangle_w^{\boldsymbol{N}},  
        
        

.. math::
   :label: _auto14

          
        \approx \sum_{i=0}^{N_0-1} \sum_{j=0}^{N_1-1}\sum_{k=0}^{N_2-1} u(x_i, y_j, z_k) \overline{v}(x_i, y_j, z_k) w(x_i, y_j, z_k),
        
        

where :math:`w(\boldsymbol{x})` now are the quadrature weights. The quadrature points
:math:`\{x_i\}_{i=0}^{N_0-1}` are specific to the chosen basis, and even within basis there
are two different choices based on which quadrature rule is selected, either
Gauss or Gauss-Lobatto. The quadrature points for the Fourier bases are the
uniform :math:`\{y_j\}_{j=0}^{N_1-1}=2\pi j / N_1` and :math:`\{z_k\}_{k=0}^{N_2-1} = 2 \pi
k/N_2`.

Inserting for test function :eq:`eq:3d:u` and trialfunction
:math:`v_{p,q,r} = \mathcal{X}_{p} \mathcal{Y}_q \mathcal{Z}_r` on the
left hand side of :eq:`eq:3d:varform`, we get

.. math::
        \begin{align*}
        \langle \nabla^2u, v \rangle_w^{\boldsymbol{N}} &= \left\langle \nabla^2\sum_{l\in \boldsymbol{l}^{N_0}}
        \sum_{m\in \boldsymbol{m}^{N_1}}\sum_{n\in \boldsymbol{n}^{N_2}}\hat{u}_{l, m, n}
        \mathcal{X}_{l} \mathcal{Y}_m \mathcal{Z}_n,
        \mathcal{X}_{p} \mathcal{Y}_q \mathcal{Z}_r \right\rangle_w^{\boldsymbol{N}}, \\ 
            &= \left[\left(\mathcal{X}_l^{''}, \mathcal{X}_p \right)_w^N - (m^2+n^2)\left(\mathcal{X}_l, \mathcal{X}_p \right)_w^N  \right]\delta_{mq} \delta_{nr} \hat{u}_{l,m,n}, \\ 
            &= \left( A_{pl} - (m^2 + n^2)B_{pl}\right) \hat{u}_{l,q,r},
        \end{align*}

where the notation :math:`(\cdot, \cdot)_w^{N_0}`

.. math::
   :label: _auto15

        
        B_{pl} = \left( \mathcal{X}_l, \mathcal{X}_p \right)_w^{N_0} = \sum_{i=0}^{N_0-1} \mathcal{X}_l(x_i)
        \mathcal{X}_p(x_i) w(x_i),
        
        

is used to represent an :math:`L_2` inner product along only the first, nonperiodic,
direction. The delta functions above come from integrating over the two periodic
directions, where we use constant weight functions :math:`w=1/(2\pi)` in the
inner products

.. math::
   :label: _auto16

        
        \int_0^{2\pi} \mathcal{Y}_m(y) \overline{\mathcal{Y}}_q(y) \frac{1}{2\pi} dy = \delta_{mq},
        
        

.. math::
   :label: _auto17

          
        \int_0^{2\pi} \mathcal{Z}_n(z) \overline{\mathcal{Z}}_r(z) \frac{1}{2\pi} dz = \delta_{nr},
        
        

The Kronecker delta-function :math:`\delta_{ij}` is one for :math:`i=j` and
zero otherwise.

The right hand side of Eq. :eq:`eq:3d:varform` is computed as

.. math::
   :label: _auto18

        
        \tilde{f}_{p,q,r} = \left\langle f, \mathcal{X}_{p}
        \mathcal{Y}_q \mathcal{Z}_r  \right \rangle_w^{\boldsymbol{N}},
        
        

where a tilde is used because this is not a complete transform of the function
:math:`f`, but only an inner product.

The linear system of equations to solve for the expansion coefficients can now
be found as follows

.. math::
   :label: eq:AB

        
        \left(A_{lj} - (m^2+n^2)B_{lj}\right) \hat{u}_{j,m,n} =
        \tilde{f}_{l,m,n}\quad \forall \, (l,m,n) \in \boldsymbol{k}. 
        

Now, when :math:`\hat{u}_{\boldsymbol{\textsf{k}}}` is found by solving this linear system over the
entire computational mesh, it may be
transformed to real space :math:`u(\boldsymbol{x})` using :eq:`eq:3d:u`. Note that the matrices
:math:`A \in \mathbb{R}^{N_0-2 \times N_0-2}` and :math:`B \in \mathbb{R}^{N_0-2 \times N_0-2}` differs for Legendre or Chebyshev bases, but
for either case they have a
special structure that allows for a solution to be found very efficiently
in the order of :math:`\mathcal{O}(N_0-2)` operations given :math:`m` and :math:`n`, see
:cite:`shen1` and :cite:`shen95`. Fast solvers for :eq:`eq:AB` are implemented in ``shenfun`` for both bases.

Method of manufactured solutions
--------------------------------

In this demo we will use the method of manufactured
solutions to demonstrate spectral accuracy of the ``shenfun`` bases. To
this end we choose a smooth analytical function that satisfies the given boundary
conditions:

.. math::
   :label: eq:3d:u_e

        
        u_e(x, y, z) = \left(\cos(4x) + \sin(2y) + \sin(4z)\right)(1-x^2). 
        

Sending :math:`u_e` through the Laplace operator, we obtain the right hand side

.. math::
   :label: eq:3d:solution

        
         \nabla^2 u_e(x,y,z) = -16(1 - x^2) \cos(4 x) + 16 x \sin(4 x) - 2 \cos(4 x)
                          - (1-x^2)(4 \sin(2y) + 16\sin(4z)).  
        

Now, setting :math:`f_e(\boldsymbol{x}) = \nabla^2 u_e(\boldsymbol{x})` and solving for :math:`\nabla^2
u(\boldsymbol{x}) = f_e(\boldsymbol{x})`, we can compare the numerical solution :math:`u(\boldsymbol{x})` with
the analytical solution :math:`u_e(\boldsymbol{x})` and compute error norms.

Implementation
==============

Preamble
--------

We will solve the Poisson problem using the `shenfun <https://github.com/spectralDNS/shenfun>`__ Python module. The first thing needed
is then to import some of this module's functionality
plus some other helper modules, like `Numpy <https://numpy.org>`__ and `Sympy <https://sympy.org>`__:

.. code-block:: python

    from sympy import symbols, cos, sin, exp, lambdify
    import numpy as np
    from shenfun.tensorproductspace import TensorProductSpace
    from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \ 
        project, Dx, Basis
    from mpi4py import MPI

We use ``Sympy`` for the manufactured solution and ``Numpy`` for testing. MPI for
Python (``mpi4py``) is required for running the solver with MPI.

Manufactured solution
---------------------

The exact solution :math:`u_e(x, y, z)` and the right hand side :math:`f_e(x, y, z)` are created using ``Sympy`` as follows

.. code-block:: python

    x, y, z = symbols("x,y,z")
    ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-x**2)
    fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)
    
    # Lambdify for faster evaluation
    ul = lambdify((x, y, z), ue, 'numpy')
    fl = lambdify((x, y, z), fe, 'numpy')

These solutions are now valid for a continuous domain. The next step is thus to
discretize, using the computational mesh

.. math::
        (x_i, y_j, z_k)\, \forall \, (i, j, k) \in [0, 1,\ldots, N_0-1] \times [0, 1, \ldots, N_1-1] \times [0, 1, \ldots, N_2-1]

and a finite number of basis functions.

Note that it is not mandatory to use ``Sympy`` for the manufactured solution. Since the
solution is known :eq:`eq:3d:solution`, we could just as well simply use ``Numpy``
to compute :math:`f_e`. However, with ``Sympy`` it is much
easier to experiment and quickly change the solution.

Discretization and MPI
----------------------

We create three bases with given size, one for each dimension of the problem.
From these three bases a :class:`.TensorProductSpace` is created.

.. code-block:: python

    # Size of discretization
    N = [14, 15, 16]
    
    SD = Basis(N[0], 'Chebyshev', bc=(0, 0))
    #SD = Basis(N[0], 'Legendre', bc=(0, 0))
    K1 = Basis(N[1], 'Fourier', dtype='D')
    K2 = Basis(N[2], 'Fourier', dtype='d')
    T = TensorProductSpace(comm, (SD, K1, K2), axes=(0, 1, 2))
    X = T.local_mesh()

Note that we can either choose a Legendre or a Chebyshev basis for the
nonperiodic direction. The
:class:`.TensorProductSpace` class takes an MPI communicator as first argument and the
computational mesh is distributed internally using the ``pencil`` method. The
``T.local_mesh`` method returns the mesh local to each processor. The ``axes``
keyword determines the order of transforms going back and forth between real and
spectral space. With ``axes=(0, 1, 2)`` and a forward transform (from real space
to spectral, i.e., from :math:`u` to :math:`\hat{u}`) axis 2 is transformed first and then 1
and 0, respectively.

The manufactured solution is created with Dirichlet boundary conditions in the
:math:`x`-direction, and for this reason ``SD`` is the first basis in ``T``. We could just
as well have put the nonperiodic direction along either :math:`y`- or :math:`z`-direction,
though, but this would then require that the order of the transformed axes be
changed as well. For example, putting the Dirichlet direction along :math:`y`, we
would need to create the tensorproductspace as

.. code-block:: python

    T = TensorProductSpace(comm, (K1, SD, K2), axes=(1, 0, 2))

such that the Dirichlet direction is the last to be transformed. The reason for
this is that only the Dirichlet direction leads to matrices that need to be
inverted (or solved). And for this we need the entire data array along the Dirichlet
direction to be local to the processor. If the ``SD`` basis is the last to be
transformed, then the data will be aligned in this direction, whereas the other
two directions may both, or just one of them, be distributed.

Note that ``X`` is a list containing local values of the arrays :math:`\{x_i\}_{i=0}^{N_0-1}`,
:math:`\{y_j\}_{j=0}^{N_1-0}` and :math:`\{z_k\}_{k=0}^{N_2-1}`. For example, using 4
procesors and a processor mesh of shape :math:`2\times 2`, then the local slices for
each processor in spectral space are

.. code-block:: python

    >>> print(comm.Get_rank(), T.local_slice())
    3 [slice(0, 14, None), slice(8, 15, None), slice(5, 9, None)]
    1 [slice(0, 14, None), slice(0, 8, None), slice(5, 9, None)]
    2 [slice(0, 14, None), slice(8, 15, None), slice(0, 5, None)]
    0 [slice(0, 14, None), slice(0, 8, None), slice(0, 5, None)]

where the global shape is :math:`\boldsymbol{N}=(14, 15, 9)` after taking advantage of
Hermitian symmetry in the :math:`z`-direction. So, all processors have the complete first dimension available locally, as they
should. Furthermore, processor three owns the slices from :math:`8:15` and :math:`5:9` along
axes :math:`y` and :math:`z`, respectively. Processor 2 owns slices :math:`0:8` and :math:`0:5` etc. In
real space the mesh is distributed differently. First of all the global mesh
shape is :math:`\boldsymbol{N}=(14, 15, 16)`, and it is distributed along the first two
dimensions. The local slices can be inspected as

.. code-block:: python

    >>> print(comm.Get_rank(), T.local_slice(False))
    0 [slice(0, 7, None), slice(0, 8, None), slice(0, 16, None)]
    1 [slice(0, 7, None), slice(8, 15, None), slice(0, 16, None)]
    2 [slice(7, 14, None), slice(0, 8, None), slice(0, 16, None)]
    3 [slice(7, 14, None), slice(8, 15, None), slice(0, 16, None)]

Since two directions are distributed, both in spectral and real space, we say
that we have a two-dimensional decomposition (here a :math:`2\times 2` shaped
processor mesh) and the
MPI distribution is of type *pencil*. It is also possible to choose a *slab*
decomposition, where only one dimension of the array is distributed. This choice
needs to be made when creating the tensorproductspace as

.. code-block:: python

    T = TensorProductSpace(comm, (SD, K1, K2), axes=(0, 1, 2), slab=True)

which will lead to a mesh that is distributed along :math:`x`-direction in real space
and :math:`y`-direction in spectral space. The local slices are

.. code-block:: python

    >>> print(comm.Get_rank(), T.local_slice()) # spectral space
    1 [slice(0, 14, None), slice(4, 8, None), slice(0, 9, None)]
    2 [slice(0, 14, None), slice(8, 12, None), slice(0, 9, None)]
    0 [slice(0, 14, None), slice(0, 4, None), slice(0, 9, None)]
    3 [slice(0, 14, None), slice(12, 15, None), slice(0, 9, None)]
    >>> print(comm.Get_rank(), T.local_slice(False)) # real space
    3 [slice(11, 14, None), slice(0, 15, None), slice(0, 16, None)]
    0 [slice(0, 4, None), slice(0, 15, None), slice(0, 16, None)]
    2 [slice(8, 11, None), slice(0, 15, None), slice(0, 16, None)]
    1 [slice(4, 8, None), slice(0, 15, None), slice(0, 16, None)]

Note that the *slab* decomposition is usually the fastest choice. However, the maximum
number of processors with *slab* is :math:`\min \{N_0, N_1\}`, whereas a *pencil*
approach can be used with up to :math:`\min \{N_1(N_2/2+1), N_0 N_1\}` processors.

Variational formulation
-----------------------

The variational problem :eq:`eq:3d:varform` can be assembled using ``shenfun``'s
form language, which is perhaps surprisingly similar to FEniCS.

.. code-block:: python

    u = TrialFunction(T)
    v = TestFunction(T)
    K = T.local_wavenumbers()
    # Get f on quad points
    fj = Array(T, buffer=fl(*X))
    # Compute right hand side of Poisson equation
    f_hat = inner(v, fj)
    # Get left hand side of Poisson equation
    matrices = inner(v, div(grad(u)))

The Laplacian operator is recognized as ``div(grad)``. The ``matrices`` object is a
dictionary representing the left hand side of :eq:`eq:AB`, and there are two
keys: (``ADDmat``, ``BDDmat``). The value of ``matrices["ADDmat"]`` is an object of
type :class:`.SpectralMatrix`,
which is ``shenfun``'s type for a matrix. This matrix represents :math:`A_{lj}`, see
:eq:`eq:AB`, and it has an attribute ``scale`` that is
equal to :math:`(2\pi)^2` (also see :eq:`eq:AB`).  The other key in matrices
is ``BDDmat``, and the value here is a :class:`.SpectralMatrix` representing :math:`B_{lj}` from
:eq:`eq:AB`. This matrix has an attribute ``scale`` that is equal to :math:`m^2+n^2`.
This ``scale`` is stored as a numpy array of shape :math:`(1, 15, 9)`, representing the
set
:math:`\{m^2+n^2: (m, n) \in \boldsymbol{m}^{N_1} \times \boldsymbol{n}^{N_2}\}`. Note that :math:`\boldsymbol{n}^{N_2}` is stored
simply as an array of length :math:`N_2/2+1` (here 9), since the transform in direction :math:`z`
takes a real signal and transforms it taking advantage of Hermitian symmetry,
see `rfft <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.fft.rfft.html>`__.

Solve linear equations
----------------------

Finally, solve linear equation system and transform solution from spectral
:math:`\hat{u}_{\boldsymbol{\textsf{k}}}` vector to the real space :math:`u(\boldsymbol{x})` and then check how the solution corresponds with the exact solution :math:`u_e`.

.. code-block:: python

    # Create Helmholtz linear algebra solver
    H = Solver(**matrices)
    
    # Solve and transform to real space
    u_hat = Function(T)           # Solution spectral space
    u_hat = H(u_hat, f_hat)       # Solve
    uq = T.backward(u_hat)
    
    # Compare with analytical solution
    uj = ul(*X)
    error = comm.reduce(np.linalg.norm(uj-uq)**2)
    if comm.Get_rank() == 0:
        print("Error=%2.16e" %(np.sqrt(error)))

Convergence test
----------------

A complete solver is given in Sec. :ref:`sec:3d:complete`. This solver is created
such that it takes in two commandline arguments and prints out the
:math:`L_2`-errornorm of the solution in the end. We can use this to write a short
script that performs a convergence test. The solver is run like

.. code-block:: text

    >>> python dirichlet_poisson3D.py 32 legendre
    Error=6.5955040031498912e-10

for a discretization of size :math:`\boldsymbol{N}= N^3 = 32^3` and for the Legendre basis.
Alternatively, change ``legendre`` to ``chebyshev`` for the Chebyshev basis.

We set up the solver to run for a list of :math:`N=[8, 10, \ldots, 38]`, and collect
the errornorms in arrays to be plotted. Such a script can be easily created
with the `subprocess <https://docs.python.org/3/library/subprocess.html>`__ module

.. code-block:: python

    import subprocess
    from numpy import log, array
    from matplotlib import pyplot as plt
    
    N = range(8, 40, 2)
    error = {}
    for basis in ('legendre', 'chebyshev'):
        error[basis] = []
        for i in range(len(N)):
            output = subprocess.check_output("python dirichlet_poisson3D.py {} {}".format(N[i], basis), shell=True)
            exec(output) # Error is printed as "Error=%2.16e"%(np.linalg.norm(uj-ua))
            error[basis].append(Error)
            if i == 0:
                print("Error          hmin           r       ")
                print("%2.8e %2.8e %2.8f"%(error[basis][-1], 1./N[i], 0))
            if i > 0:
                print("%2.8e %2.8e %2.8f"%(error[basis][-1], 1./N[i], log(error[basis][-1]/error[basis][-2])/log(N[i-1]/N[i])))
    

The error can be plotted using `matplotlib <https://matplotlib.org>`__, and the
generated figure is shown in the summary's Fig. :ref:`fig:3d:ct0`. The spectral
convergence is evident and we can see that after :math:`N=25` roundoff errors dominate
as the errornorm trails off around :math:`10^{-13}`.

.. code-block:: python

    plt.figure(figsize=(6, 4))
    for basis, col in zip(('legendre', 'chebyshev'), ('r', 'b')):
        plt.semilogy(N, error[basis], col, linewidth=2)
    plt.title('Convergence of Poisson solvers 3D')
    plt.xlabel('N')
    plt.ylabel('Error norm')
    plt.legend(('Legendre', 'Chebyshev'))
    plt.savefig('poisson3D_errornorm.png')
    plt.show()

.. FIGURE: [poisson1D_errornorm.png] Convergence test of Legendre and Chebyshev 1D Poisson solvers.

.. _sec:3d:complete:

Complete solver
---------------
A complete solver, that can use either Legendre or Chebyshev bases, and any quadrature size
chosen as a command-line argument, is shown below.

.. code-block:: text

    >>> python dirichlet_poisson3D.py 36 legendre

or similarly with ``chebyshev`` instead of ``legendre``.

.. code-block:: python

    import sys, os
    import importlib
    from sympy import symbols, cos, sin, lambdify
    import numpy as np
    from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \ 
        Function, Basis, TensorProductSpace
    import time
    from mpi4py import MPI
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    
    comm = MPI.COMM_WORLD
    
    assert len(sys.argv) == 3
    assert sys.argv[-1].lower() in ('legendre', 'chebyshev')
    assert isinstance(int(sys.argv[-2]), int)
    
    # Collect basis and solver from either Chebyshev or Legendre submodules
    family = sys.argv[-1].lower()
    base = importlib.import_module('.'.join(('shenfun', family)))
    Solver = base.la.Helmholtz
    
    # Use sympy to compute a rhs, given an analytical solution
    a = -0
    b = 0
    x, y, z = symbols("x,y,z")
    ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-z**2) + a*(1 + z)/2. + b*(1 - z)/2.
    fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)
    
    # Lambdify for faster evaluation
    ul = lambdify((x, y, z), ue, 'numpy')
    fl = lambdify((x, y, z), fe, 'numpy')
    
    # Size of discretization
    N = int(sys.argv[-2])
    N = [N, N, N]
    
    SD = Basis(N[0], family=family, bc=(a, b))
    K1 = Basis(N[1], family='F', dtype='D')
    K2 = Basis(N[2], family='F', dtype='d')
    T = TensorProductSpace(comm, (K1, K2, SD), axes=(0, 1, 2), slab=True)
    X = T.local_mesh()
    u = TrialFunction(T)
    v = TestFunction(T)
    
    K = T.local_wavenumbers()
    
    # Get f on quad points
    fj = Array(T, buffer=fl(*X))
    
    # Compute right hand side of Poisson equation
    f_hat = inner(v, fj)
    if family == 'legendre':
        f_hat *= -1.
    
    # Get left hand side of Poisson equation
    if family == 'chebyshev':
        matrices = inner(v, div(grad(u)))
    else:
        matrices = inner(grad(v), grad(u))
    
    # Create Helmholtz linear algebra solver
    H = Solver(**matrices)
    
    # Solve and transform to real space
    u_hat = Function(T)           # Solution spectral space
    t0 = time.time()
    u_hat = H(u_hat, f_hat)       # Solve
    uq = T.backward(u_hat, fast_transform=False)
    
    # Compare with analytical solution
    uj = ul(*X)
    error = comm.reduce(np.linalg.norm(uj-uq)**2)
    if comm.Get_rank() == 0:
        print("Error=%2.16e" %(np.sqrt(error)))

.. ======= Bibliography =======

.. bibliography:: papers.bib
   :notcited:
