.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - Lid driven cavity
========================

:Authors: Mikael Mortensen (mikaem at math.uio.no)
:Date: Apr 14, 2020

*Summary.* The lid driven cavity is a classical benchmark for Navier Stokes solvers.
This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve the lid
driven cavity problem with full spectral accuracy using a mixed (coupled) basis
in a 2D tensor product domain. The demo also shows how to use mixed
tensor product spaces for vector valued equations. Note that the regular
lid driven cavity, where the top wall has constant velocity and the
remaining three walls are stationary, has a singularity at the two
upper corners, where the velocity is discontinuous.
Due to their global nature, spectral methods
are usually not very good at handling problems with discontinuities, and
for this reason we will also look at a regularized lid driven cavity,
where the top lid moves according to :math:`(1-x)^2(1+x)^2`, thus removing
the corner discontinuities.

.. _fig:drivencavity:

.. figure:: https://raw.githack.com/spectralDNS/spectralutilities/master/figures/DrivenCavity.png

   Velocity vectors for :math:`Re=100`

.. _demo:navierstokes:

Navier Stokes equations
-----------------------

The nonlinear steady Navier Stokes equations are given in strong form as

.. math::
        \begin{align*}
        \nu \nabla^2 \bs{u} - \nabla p &= \nabla \cdot \bs{u} \bs{u} \quad \text{in }  \Omega , \\ 
        \nabla \cdot \bs{u} &= 0 \quad \text{in } \Omega  \\ 
        \int_{\Omega} p dx &= 0 \\ 
        \bs{u}(x, y=1) = (1, 0) \, &\text{ or }\, \bs{u}(x, y=1) = ((1-x)^2(1+x)^2, 0) \\ 
        \bs{u}(x, y=-1) &= (0, 0) \\ 
        \bs{u}(x=\pm 1, y) &= (0, 0)
        \end{align*}

where :math:`\bs{u}, p` and :math:`\nu` are, respectively, the
fluid velocity vector, pressure and kinematic viscosity. The domain
:math:`\Omega = [-1, 1]^2` and the nonlinear term :math:`\bs{u} \bs{u}` is the
outer product of vector :math:`\bs{u}` with itself. Note that the final
:math:`\int_{\Omega} p dx = 0` is there because there is no Dirichlet boundary
condition on the pressure and the system of equations would otherwise be
ill conditioned.

We want to solve these steady nonlinear Navier Stokes equations with the Galerkin
method, using the `shenfun <https://github.com/spectralDNS/shenfun>`__ Python
package. The first thing we need to do then is to import all of shenfun's
functionality

.. code-block:: python

    import matplotlib.pyplot as plt
    from shenfun import *
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

Note that MPI for Python (`mpi4py <https://bitbucket.org/mpi4py/mpi4py>`__)
is a requirement for shenfun, but the current solver cannot be used with more
than one processor.

.. _sec:bases:

Bases and tensor product spaces
-------------------------------

With the Galerkin method we need basis functions for both velocity and
pressure, as well as for the
nonlinear right hand side. A Dirichlet basis will be used for velocity,
whereas there is no boundary restriction on the pressure basis. For both
two-dimensional bases we will use one basis function for the :math:`x`-direction,
:math:`\mathcal{X}_k(x)`, and one for the :math:`y`-direction, :math:`\mathcal{Y}_l(y)`. And
then we create two-dimensional basis functions like

.. math::
   :label: eq:nstestfunction

        
        v_{kl}(x, y) = \mathcal{X}_k(x) \mathcal{Y}_l(y),  
        

and solutions (trial functions) as

.. math::
   :label: eq:nstrialfunction

        
            u(x, y) = \sum_{k}\sum_{l} \hat{u}_{kl} v_{kl}(x, y). 
        

For the homogeneous Dirichlet boundary condition the basis functions
:math:`\mathcal{X}_k(x)` and :math:`\mathcal{Y}_l(y)` are chosen as composite
Legendre polynomials (we could also use Chebyshev):

.. math::
   :label: eq:D0

        
        \mathcal{X}_k(x) = L_k(x) - L_{k+2}(x), \quad \forall \, k \in \bs{k}^{N_0-2},  
        

.. math::
   :label: eq:D1

          
        \mathcal{Y}_l(y) = L_l(y) - L_{l+2}(y), \quad \forall \, l \in \bs{l}^{N_1-2}, 
        

where :math:`\bs{k}^{N_0-2} = (0, 1, \ldots, N_0-3)`, :math:`\bs{l}^{N_1-2} = (0, 1, \ldots, N_1-3)`
and :math:`N = (N_0, N_1)` is the number
of quadrature points in each direction. Note that :math:`N_0` and :math:`N_1` do not need
to be the same. The basis funciton :eq:`eq:D0` satisfies
the homogeneous Dirichlet boundary conditions at :math:`x=\pm 1` and :eq:`eq:D1` the same
at :math:`y=\pm 1`. As such, the basis function :math:`v_{kl}(x, y)` satisfies the homogeneous Dirichlet boundary
condition for the entire domain.

With shenfun we create these homogeneous spaces, :math:`D_0^{N_0}(x)=\text{span}\{L_k-L_{k+2}\}_{k=0}^{N_0-2}` and
:math:`D_0^{N_1}(y)=\text{span}\{L_l-L_{l+2}\}_{l=0}^{N_1-2}` as

.. code-block:: python

    N = (51, 51)
    family = 'Legendre' # or use 'Chebyshev'
    quad = 'LG'         # for Chebyshev use 'GC' or 'GL'
    D0X = Basis(N[0], family, quad=quad, bc=(0, 0))
    D0Y = Basis(N[1], family, quad=quad, bc=(0, 0))

The spaces are here the same, but we will use ``D0X`` in the :math:`x`-direction and
``D0Y`` in the :math:`y`-direction. But before we use these bases in
tensor product spaces, they remain identical as long as :math:`N_0 = N_1`.

Special attention is required by the moving lid. To get a solution
with nonzero boundary condition at :math:`y=1` we need to add one more basis function
that satisfies that solution. In general, a nonzero boundary condition
can be added on both sides of the domain using the following basis

.. math::
   :label: _auto1

        
        \mathcal{Y}_l(y) = L_l(y) - L_{l+2}(y), \quad \forall \, l \in \bs{l}^{N_1-2}. 
        
        

.. math::
   :label: _auto2

          
        \mathcal{Y}_{N_1-2}(y) = (L_0+L_1)/2 \quad \left(=(1+y)/2\right), 
        
        

.. math::
   :label: _auto3

          
        \mathcal{Y}_{N_1-1}(y) = (L_0-L_1)/2 \quad \left(=(1-y)/2\right).
        
        

And then the unknown component :math:`N_1-2` decides the value at :math:`y=1`, whereas
the unknown at :math:`N_1-1` decides the value at :math:`y=-1`. Here we only need to
add the :math:`N_1-2` component, but for generality this is implemented in shenfun
using both additional basis functions. We create the space
:math:`D_1^{N_1}(y)=\text{span}\{\mathcal{Y}_l(y)\}_{l=0}^{N_1-1}` as

.. code-block:: python

    D1Y = Basis(N[1], family, quad=quad, bc=(1, 0))

where ``bc=(1, 0)`` fixes the values for :math:`y=1` and :math:`y=-1`, respectively.
For a regularized lid driven cavity the velocity of the top lid is
:math:`(1-x)^2(1+x)^2` and not unity. To implement this boundary condition
instead, we can make use of `sympy <https://www.sympy.org>`__ and
quite straight forward do

.. code-block:: python

    import sympy
    x = sympy.symbols('x')
    #D1Y = Basis(N[1], family, quad=quad, bc=((1-x)**2*(1+x)**2, 0))

Uncomment the last line to run the regularized boundary conditions.
Otherwise, there is no difference at all between the regular and the
regularized lid driven cavity implementations.

The pressure basis that comes with no restrictions for the boundary is a
little trickier. The reason for this has to do with
inf-sup stability. The obvious choice of basis functions are the
regular Legendre polynomials :math:`L_k(x)` in :math:`x` and :math:`L_l(y)` in the
:math:`y`-directions. The problem is that for the natural choice of
:math:`(k, l) \in \bs{k}^{N_0} \times \bs{l}^{N_1}`
there are nullspaces and the problem is not well-defined. It turns out
that the proper choice for the pressure basis is simply the regular
Legendre basis functions, but for
:math:`(k, l) \in \bs{k}^{N_0-2} \times \bs{l}^{N_1-2}`.
The bases :math:`P^{N_0}(x)=\text{span}\{L_k(x)\}_{k=0}^{N_0-3}` and
:math:`P^{N_1}(y)=\text{span}\{L_l(y)\}_{l=0}^{N_1-3}` are created as

.. code-block:: python

    PX = Basis(N[0], family, quad=quad)
    PY = Basis(N[1], family, quad=quad)
    PX.slice = lambda: slice(0, N[0]-2)
    PY.slice = lambda: slice(0, N[1]-2)

Note that we still use these spaces with the same :math:`N_0 \cdot N_1`
quadrature points in real space, but the two highest frequencies have
been set to zero.

We have now created all relevant function spaces for the problem at hand.
It remains to combine these spaces into tensor product spaces, and to
combine tensor product spaces into mixed (coupled) tensor product
spaces. From the Dirichlet bases we create two different tensor
product spaces, whereas one is enough for the pressure

.. math::
   :label: _auto4

        
        V_{1}^{\bs{N}}(\bs{x}) = D_0^{N_0}(x) \otimes D_1^{N_1}(y), 
        
        

.. math::
   :label: _auto5

          
        V_{0}^{\bs{N}}(\bs{x}) = D_0^{N_0}(x) \otimes D_0^{N_1}(y), 
        
        

.. math::
   :label: _auto6

          
        P^{\bs{N}}(\bs{x}) = P^{N_0}(x) \otimes P^{N_1}(y).
        
        

With shenfun the tensor product spaces are created as

.. code-block:: python

    V1 = TensorProductSpace(comm, (D0X, D1Y))
    V0 = TensorProductSpace(comm, (D0X, D0Y))
    P = TensorProductSpace(comm, (PX, PY))

These tensor product spaces are all scalar valued.
The velocity is a vector, and a vector requires a mixed basis like
:math:`W_1^{\bs{N}} = V_1^{\bs{N}} \times V_0^{\bs{N}}`. The mixed basis is created
in shenfun as

.. code-block:: python

    W1 = MixedTensorProductSpace([V1, V0])
    W0 = MixedTensorProductSpace([V0, V0])

Note that the second mixed basis, :math:`W_0^{\bs{N}} = V_0^{\bs{N}} \times V_0^{\bs{N}}`, uses
homogeneous boundary conditions throughout.

.. _sec:mixedform:

Mixed variational form
----------------------

We now formulate a variational problem using the
Galerkin method: Find
:math:`\bs{u} \in W_1^{\bs{N}}` and :math:`p \in P^{\bs{N}}` such that

.. math::
   :label: eq:nsvarform

        
        \int_{\Omega} (\nu \nabla^2 \bs{u} - \nabla p ) \cdot \bs{v} \, dxdy = \int_{\Omega} (\nabla \cdot \bs{u}\bs{u}) \cdot \bs{v}\, dxdy \quad\forall \bs{v} \, \in \, W_0^{\bs{N}},  
        

.. math::
   :label: _auto7

          
        \int_{\Omega} \nabla \cdot \bs{u} \, q \, dxdy = 0 \quad\forall q \, \in \, P^{\bs{N}}.
        
        

Note that we are using test functions :math:`\bs{v}` with homogeneous
boundary conditions.

The first obvious issue with Eq :eq:`eq:nsvarform` is the nonlinearity.
In other words we will
need to linearize and iterate to be able to solve these equations with
the Galerkin method. To this end we will introduce the solution on
iteration :math:`k \in [0, 1, \ldots]` as :math:`\bs{u}^k` and compute the nonlinearity
using only known solutions
:math:`\int_{\Omega} (\nabla \cdot \bs{u}^k\bs{u}^k) \cdot \bs{v}\, dxdy`.
Using further integration by parts we end up with the equations to solve
for iteration number :math:`k+1` (using :math:`\bs{u} = \bs{u}^{k+1}` and :math:`p=p^{k+1}`
for simplicity)

.. math::
   :label: eq:nsvarform2

        
        -\int_{\Omega} \nu \nabla \bs{u} \, \colon \nabla \bs{v} \, dxdy + \int_{\Omega} p \nabla \cdot \bs{v} \, dxdy = \int_{\Omega} (\nabla \cdot \bs{u}^k\bs{u}^k) \cdot \bs{v}\, dxdy \quad\forall \bs{v} \, \in \, W_0^{\bs{N}},  
        

.. math::
   :label: _auto8

          
        \int_{\Omega} \nabla \cdot \bs{u} \, q \, dxdy = 0 \quad\forall q \, \in \, P^{\bs{N}}.
        
        

Note that the nonlinear term may also be integrated by parts and
evaluated as :math:`\int_{\Omega}-\bs{u}^k\bs{u}^k  \, \colon \nabla \bs{v} \, dxdy`. All
boundary integrals disappear since we are using test functions with
homogeneous boundary conditions.

Since we are to solve for :math:`\bs{u}` and :math:`p` at the same time, we formulate a
mixed (coupled) problem: find :math:`(\bs{u}, p) \in W_1^{\bs{N}} \times P^{\bs{N}}`
such that

.. math::
   :label: _auto9

        
        a((\bs{u}, p), (\bs{v}, q)) = L((\bs{v}, q)) \quad \forall (\bs{v}, q) \in W_0^{\bs{N}} \times P^{\bs{N}},
        
        

where bilinear (:math:`a`) and linear (:math:`L`) forms are given as

.. math::
   :label: _auto10

        
            a((\bs{u}, p), (\bs{v}, q)) = -\int_{\Omega} \nu \nabla \bs{u} \, \colon \nabla \bs{v} \, dxdy + \int_{\Omega} p \nabla \cdot \bs{v} \, dxdy + \int_{\Omega} \nabla \cdot \bs{u} \, q \, dxdy, 
        
        

.. math::
   :label: _auto11

          
            L((\bs{v}, q); \bs{u}^{k}) = \int_{\Omega} (\nabla \cdot \bs{u}^{k}\bs{u}^{k}) \cdot \bs{v}\, dxdy.
        
        

Note that the bilinear form will assemble to a block matrix, whereas the right hand side
linear form will assemble to a block vector. The bilinear form does not change
with the solution and as such it does not need to be reassembled inside
an iteration loop.

The algorithm used to solve the equations are:

  * Set :math:`k = 0`

  * Guess :math:`\bs{u}^0 = (0, 0)`

  * while not converged:

    * assemble :math:`L((\bs{v}, q); \bs{u}^{k})`

    * solve :math:`a((\bs{u}, p), (\bs{v}, q)) = L((\bs{v}, q); \bs{u}^{k})` for :math:`\bs{u}^{k+1}, p^{k+1}`

    * compute error = :math:`\int_{\Omega} (\bs{u}^{k+1}-\bs{u}^{k})^2 \, dxdy`

    * if error :math:`<` some tolerance then converged = True

    * :math:`k` += :math:`1`

Implementation of solver
------------------------

We will now implement the coupled variational problem described in previous
sections. First of all, since we want to solve for the velocity and pressure
in a coupled solver, we have to
create a mixed tensor product space :math:`VQ = W_1^{\bs{N}} \times P^{\bs{N}}` that
couples velocity and pressure

.. code-block:: python

    VQ = MixedTensorProductSpace([W1, P])    # Coupling velocity and pressure

We can now create test- and trialfunctions for the coupled space :math:`VQ`,
and then split them up into components afterwards:

.. code-block:: python

    up = TrialFunction(VQ)
    vq = TestFunction(VQ)
    u, p = up
    v, q = vq


.. note::
   The test function ``v`` is using homogeneous Dirichlet boundary conditions even
   though it is derived from ``VQ``, which contains ``W1``. It is currently not (and will
   probably never be) possible to use test functions with inhomogeneous
   boundary conditions.




With the basisfunctions in place we may assemble the different blocks of the
final coefficient matrix. For this we also need to specify the kinematic
viscosity, which is given here in terms of the Reynolds number:

.. code-block:: python

    Re = 100.
    nu = 2./Re
    A = inner(grad(v), -nu*grad(u))
    G = inner(div(v), p)
    D = inner(q, div(u))


.. note::
   The inner products may also be assembled with one single line, as
   
   .. code-block:: text
   
       AA = inner(grad(v), -nu*grad(u)) + inner(div(v), p) + inner(q, div(u))
   
   But note that this requires addition, not subtraction, of inner products,
   and it is not possible to move the negation to ``-inner(grad(v), nu*grad(u))``.
   This is because the :func:`.inner` function returns a list of
   tensor product matrices of type :class:`.TPMatrix`, and you cannot
   negate a list.




The assembled subsystems ``A, G`` and ``D`` are lists containg the different blocks of
the complete, coupled, coefficient matrix. ``A`` actually contains 4
tensor product matrices of type :class:`.TPMatrix`. The first two
matrices are for vector component zero of the test function ``v[0]`` and
trial function ``u[0]``, the
matrices 2 and 3 are for components 1. The first two matrices are as such for

.. code-block:: text

      A[0:2] = inner(grad(v[0]), -nu*grad(u[0]))

Breaking it down the inner product is mathematically

.. math::
   :label: eq:partialeq1

        
        
        \int_{\Omega}-\nu \left(\frac{\partial \bs{v}[0]}{\partial x}, \frac{\partial \bs{v}[0]}{\partial y}\right) \cdot \left(\frac{\partial \bs{u}[0]}{\partial x}, \frac{\partial \bs{u}[0]}{\partial y}\right) dx dy .
        

We can now insert for test function :math:`\bs{v}[0]`

.. math::
   :label: _auto12

        
        \bs{v}[0]_{kl} = \mathcal{X}_k \mathcal{Y}_l, \quad (k, l) \in \bs{k}^{N_0-2} \times \bs{l}^{N_1-2}
        
        

and trialfunction

.. math::
   :label: _auto13

        
        \bs{u}[0]_{mn} = \sum_{m=0}^{N_0-3} \sum_{n=0}^{N_1-1} \hat{\bs{u}}[0]_{mn} \mathcal{X}_m \mathcal{Y}_n,
        
        

where :math:`\hat{\bs{u}}` are the unknown degrees of freedom for the velocity vector.
Notice that the sum over the second
index runs all the way to :math:`N_1-1`, whereas the other indices runs to either
:math:`N_0-3` or :math:`N_1-3`. This is because of the additional basis functions required
for the inhomogeneous boundary condition.

Inserting for these basis functions into :eq:`eq:partialeq1`, we obtain after a few trivial
manipulations

.. math::
   :label: _auto14

        
         -\sum_{m=0}^{N_0-3} \sum_{n=0}^{N_1-1} \nu \Big( \underbrace{\int_{-1}^{1} \frac{\partial \mathcal{X}_k(x)}{\partial x} \frac{\partial \mathcal{X}_m}{\partial x} dx \int_{-1}^{1} \mathcal{Y}_l \mathcal{Y}_n dy}_{A[0]} +  \underbrace{\int_{-1}^{1} \mathcal{X}_k(x) X_m(x) dx \int_{-1}^{1} \frac{\partial \mathcal{Y}_l}{\partial y} \frac{\partial \mathcal{Y}_n}{\partial y} dy}_{A[1]}  \Big) \hat{\bs{u}}[0]_{mn}.
        
        

We see that each tensor product matrix (both A[0] and A[1]) is composed as
outer products of two smaller matrices, one for each dimension.
The first tensor product matrix, A[0], is

.. math::
   :label: _auto15

        
            \underbrace{\int_{-1}^{1} \frac{\partial \mathcal{X}_k(x)}{\partial x} \frac{\partial \mathcal{X}_m}{\partial x} dx}_{c_{km}} \underbrace{\int_{-1}^{1} \mathcal{Y}_l \mathcal{Y}_n dy}_{f_{ln}}
        
        

where :math:`C\in \mathbb{R}^{N_0-2 \times N_1-2}` and :math:`F \in \mathbb{R}^{N_0-2 \times N_1}`.
Note that due to the inhomogeneous boundary conditions this last matrix :math:`F`
is actually not square. However, remember that all contributions from the two highest
degrees of freedom (:math:`\hat{\bs{u}}[0]_{m,N_1-2}` and :math:`\hat{\bs{u}}[0]_{m,N_1-1}`) are already
known and they can, as such, be  moved directly over to the right hand side of the
linear algebra system that is to be solved. More precisely, we can split the
tensor product matrix into two contributions and obtain

.. math::
        \sum_{m=0}^{N_0-3}\sum_{n=0}^{N_1-1} c_{km}f_{ln} \hat{\bs{u}}[0]_{m, n} = \sum_{m=0}^{N_0-3}\sum_{n=0}^{N_1-3}c_{km}f_{ln}\hat{\bs{u}}[0]_{m, n} + \sum_{m=0}^{N_0-3}\sum_{n=N_1-2}^{N_1-1}c_{km}f_{ln}\hat{\bs{u}}[0]_{m, n}, \quad \forall (k, l) \in \bs{k}^{N_0-2} \times \bs{l}^{N_1-2},

where the first term on the right hand side is square and the second term is known and
can be moved to the right hand side of the linear algebra equation system.

All the parts of the matrices that are to be moved to the right hand side
can be extracted from A, G and D as follows

.. code-block:: python

    # Extract the boundary matrices
    bc_mats = extract_bc_matrices([A, G, D])

These matrices are applied to the solution below (see ``BlockMatrix BM``).
Furthermore, this leaves us with square submatrices (A, G, D), which make up a
symmetric block matrix

.. math::
   :label: eq:nsbmatrix

        M =
          \begin{bmatrix}
              A[0]+A[1] & 0 & G[0] \\ 
              0 & A[2]+A[3] & G[1] \\ 
              D[0] & D[1] & 0
          \end{bmatrix}

This matrix, and the matrix responsible for the boundary degrees of freedom,
can be assembled from the pieces we already have as

.. code-block:: python

    M = BlockMatrix(A+G+D)
    BM = BlockMatrix(bc_mats)

We now have all the matrices we need in order to solve the Navier Stokes equations.
However, we also need some work arrays for iterations and we need to
assemble the constant boundary contribution to the right hand side

.. code-block:: python

    # Create Function to hold solution
    uh_hat = Function(VQ)
    ui_hat = uh_hat[0]
    D1Y.bc.apply_after(ui_hat[0], True) # Fixes the values of the boundary dofs
    
    # New solution (iterative)
    uh_new = Function(VQ)
    ui_new = uh_new[0]
    D1Y.bc.apply_after(ui_new[0], True)
    
    # Compute the constant contribution to rhs due to nonhomogeneous boundary conditions
    bh_hat0 = Function(VQ)
    bh_hat0 = BM.matvec(-uh_hat, bh_hat0) # Negative because moved to right hand side
    bi_hat0 = bh_hat0[0]
    

Note that ``bh_hat0`` now contains the part of the right hand side that is
due to the non-symmetric part of assembled matrices. The line with
``D1Y.bc.apply_after(ui_hat[0], True)`` ensures the known boundary values of
the solution are fixed for ``ui_hat``.

The nonlinear right hand side also requires some additional attention.
Nonlinear terms are usually computed in physical space before transforming
to spectral. For this we need to evaluate the velocity vector on the
quadrature mesh. We also need a rank 2 Array to hold the outer
product :math:`\bs{u}\bs{u}`. The required arrays and spaces are
created as

.. code-block:: python

    bh_hat = Function(VQ)
    
    # Create arrays to hold velocity vector solution
    ui = Array(W1)
    
    # Create work arrays for nonlinear part
    QT = MixedTensorProductSpace([W1, W0])  # for uiuj
    uiuj = Array(QT)
    uiuj_hat = Function(QT)

The right hand side :math:`L((\bs{v}, q);\bs{u}^{k});` is computed in its
own function ``compute_rhs`` as

.. code-block:: python

    def compute_rhs(ui_hat, bh_hat):
        global ui, uiuj, uiuj_hat, V1, bh_hat0
        bh_hat.fill(0)
        ui = W1.backward(ui_hat, ui)
        uiuj = outer(ui, ui, uiuj)
        uiuj_hat = uiuj.forward(uiuj_hat)
        bi_hat = bh_hat[0]
        #bi_hat = inner(v, div(uiuj_hat), output_array=bi_hat)
        bi_hat = inner(grad(v), -uiuj_hat, output_array=bi_hat)
        bh_hat += bh_hat0
        return bh_hat

Here :func:`.outer` is a shenfun function that computes the
outer product of two vectors and returns the product in a rank two
array (here ``uiuj``). With ``uiuj`` forward transformed to ``uiuj_hat``
we can assemble the linear form either as ``inner(v, div(uiuj_hat)`` or
``inner(grad(v), -uiuj_hat)``. Also notice that the constant contribution
from the inhomogeneous boundary condition, ``bh_hat0``,
is added to the right hand side vector.

Now all that remains is to guess an initial solution and solve
iteratively until convergence. For initial solution we simply set the
velocity and pressure to zero and solve the Stokes equations:

.. code-block:: python

    from scipy.sparse.linalg import splu
    uh_hat, Ai = M.solve(bh_hat0, u=uh_hat, constraints=((2, 0, 0),), return_system=True) # Constraint for component 2 of mixed space
    Alu = splu(Ai)
    uh_new[:] = uh_hat

Note that the :class:`.BlockMatrix` given by ``M`` has a solve method that sets up
a sparse coefficient matrix ``Ai`` of size :math:`\mathbb{R}^{3(N_0-2)(N_1-2) \times 3(N_0-2)(N_1-2)}`,
and then solves using `scipy.sparse.linalg.spsolve <http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve>`__.
The matrix ``Ai`` is then pre-factored for reuse with `splu <http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.splu.html#scipy.sparse.linalg.splu>`__.
Also note that the ``constraints=((2, 0, 0),)`` keyword argument
ensures that the pressure integrates to zero, i.e., :math:`\int_{\Omega} pdxdy=0`.
Here the number 2 tells us that block component 2 in the mixed space
(the pressure) should be integrated, dof 0 should be fixed, and it
should be fixed to 0.

With an initial solution from the Stokes equations we are ready to start iterating.
However, for convergence it is necessary to add some underrelaxation :math:`\alpha`,
and update the solution each time step as

.. math::
        \begin{align*}
        \hat{\bs{u}}^{k+1} &= \alpha \hat{\bs{u}}^* + (1-\alpha)\hat{\bs{u}}^{k},\\ 
        \hat{p}^{k+1} &= \alpha \hat{p}^* + (1-\alpha)\hat{p}^{k},
        \end{align*}

where :math:`\hat{\bs{u}}^*` and :math:`\hat{p}^*` are the newly computed velocity
and pressure returned from ``M.solve``. Without underrelaxation the solution
will quickly blow up. The iteration loop goes as follows

.. code-block:: python

    converged = False
    count = 0
    alfa = 0.5
    while not converged:
        count += 1
        bh_hat = compute_rhs(ui_hat, bh_hat)
        uh_new = M.solve(bh_hat, u=uh_new, constraints=((2, 0, 0),), Alu=Alu) # Constraint for component 2 of mixed space
        error = np.linalg.norm(ui_hat-ui_new)
        uh_hat[:] = alfa*uh_new + (1-alfa)*uh_hat
        converged = abs(error) < 1e-10 or count >= 10000
        print('Iteration %d Error %2.4e' %(count, error))
    
    up = uh_hat.backward()
    u, p = up
    
    X = V0.local_mesh(True)
    plt.figure()
    plt.quiver(X[0], X[1], u[0], u[1])

The last three lines plots the velocity vectors that are shown
in Figure :ref:`fig:drivencavity`. The solution is apparently nice
and smooth, but hidden underneath are Gibbs oscillations from the
corner discontinuities. This is painfully obvious when switching from
Legendre to Chebyshev polynomials. With Chebyshev the same plot looks
like Figure :ref:`fig:drivencavitycheb`. However, choosing instead the
regularized lid, the solutions will be nice and smooth, both for
Legendre and Chebyshev polynomials.

.. _fig:drivencavitycheb:

.. figure:: https://raw.githack.com/spectralDNS/spectralutilities/master/figures/DrivenCavityCheb.png

   *Velocity vectors for Re=100 using Chebyshev*

.. _sec:nscomplete:

Complete solver
---------------

A complete solver can be found in demo `NavierStokesDrivenCavity.py <https://github.com/spectralDNS/shenfun/blob/master/demo/NavierStokesDrivenCavity.py>`__.
