Introduction
============

Spectral Galerkin
-----------------

The spectral Galerkin method solves partial differential equations through
a special form of the `method of weighted residuals <https://en.wikiversity.org/wiki/Introduction_to_finite_elements/Weighted_residual_methods>`_ (WRM). As a Galerkin method it
is very similar to the `finite element method <https://en.wikipedia.org/wiki/Finite_element_method>`_ (FEM). The most distinguishable
feature is that it uses global shape functions, where FEM uses local. This
feature leads to highly accurate results with very few shape functions, but
the downside is much less flexibility when it comes to computational
domain than FEM.

Consider the Poisson equation with a right hand side function :math:`f(\boldsymbol{x})`

.. math::
   :label: eq:poisson1

    -\nabla^2 u(\boldsymbol{x}) = f(\boldsymbol{x}) \quad \text{for } \, \boldsymbol{x} \in \Omega.

To solve this equation, we will eventually need to supplement
appropriate boundary conditions. However, for now just assume that any valid
boundary conditions (Dirichlet, Neumann, periodic).

With the method of weighted residuals we attempt to find :math:`u(\boldsymbol{x})`
using an approximation, :math:`u_N`, to the solution

.. math::
   :label: eq:mwr_u

    u(\boldsymbol{x}) \approx u_N(\boldsymbol{x}) = \sum_{k=0}^{N-1} \hat{u}_k \phi_k(\boldsymbol{x}).

Here the :math:`N` expansion coefficients :math:`\hat{u}_k` are unknown
and :math:`\{\phi_k\}_{k\in \mathcal{I}^N}, \mathcal{I}^N = 0, 1, \ldots, N-1` are
*trial* functions. Inserting for :math:`u_N` in Eq. :eq:`eq:poisson1` we get
a residual

.. math::
   :label: eq:residual

    R_N(\boldsymbol{x}) = \nabla^2 u_N(\boldsymbol{x}) + f(\boldsymbol{x}) \neq 0.

With the WRM we now force this residual to zero in an average sense using
*test* function :math:`v(\boldsymbol{x})` and *weight* function
:math:`w(\boldsymbol{x})`

.. math::
   :label: eq:wrm_test

    \left(R_N, v \right)_w := \int_{\Omega} R_N(\boldsymbol{x}) \, \overline{v}(\boldsymbol{x}) \, w(\boldsymbol{x}) d\boldsymbol{x} = 0,

where :math:`\overline{v}` is the complex conjugate of :math:`v`. If we
now choose the test functions from the same space as the trial functions,
i.e., :math:`V_N=span\{\phi_k\}_{k\in \mathcal{I}^N}`,
then the WRM becomes the Galerkin method, and we get :math:`N` equations for
:math:`N` unknowns :math:`\{\hat{u}_k\}_{k\in \mathcal{I}^N}`

.. math::
   :label: eq:galerkin

    \sum_{j\in \mathcal{I}^N} \underbrace{\left(-\nabla^2 \phi_j, \phi_k \right)_w}_{A_{kj}} \hat{u}_j = \left( f, \phi_k \right)_w, \text{ for } k \in \mathcal{I}^N.

Note that this is a regular linear system of algebra equations

.. math::

    A_{kj} \hat{u}_{j} = \tilde{f}_k,

where the matrix :math:`A \in \mathbb{R}^{N \times N}`.

The choice of basis functions :math:`v(\boldsymbol{x})` is highly central to the method.
For the Galerkin method to be *spectral*, the basis is usually chosen as linear
combinations of Chebyshev, Legendre, Laguerre, Hermite, Jacobi or trigonometric functions.
In one spatial dimension typical choices for :math:`\phi_k` are

.. math::

   \phi_k(x) &= T_k(x) \\
   \phi_k(x) &= T_k(x) - T_{k+2}(x) \\
   \phi_k(x) &= L_k(x) \\
   \phi_k(x) &= L_k(x) - L_{k+2}(x) \\
   \phi_k(x) &= \exp(\imath k x)

where :math:`T_k, L_k` are the :math:`k`'th Chebyshev polynomial of the first
kind and the :math:`k`'th Legendre polynomial, respectively. Note that the
second and fourth functions above satisfy the homogeneous Dirichlet boundary
conditions :math:`\phi_k(\pm 1) = 0`, and as such these basis functions may be
used to solve the Poisson equation :eq:`eq:poisson1` with homogeneous Dirichlet
boundary conditions. Similarly, two basis functions that satisfy homogeneous
Neumann boundary condition :math:`u'(\pm 1)=0` are

.. math::

    \phi_k &= T_k-\left(\frac{k}{k+2}\right)^2T_{k+2} \\
    \phi_k &= L_k-\frac{k(k+1)}{(k+2)(k+3)}L_{k+2}

Shenfun contains classes for working with several such bases, to be used for
different equations and boundary conditions.

Complete demonstration programs that solves the Poisson equation
:eq:`eq:poisson1`, and some other problems can be found by following these
links

    * :ref:`Demo - 1D Poisson's equation`
    * :ref:`Demo - 3D Poisson's equation`
    * :ref:`Demo - Helmholtz equation in polar coordinates`
    * :ref:`Demo - Cubic nonlinear Klein-Gordon equation`
    * :ref:`Demo - Kuramato-Sivashinsky equation`
    * :ref:`Demo - Stokes equations`
    * :ref:`Demo - Lid driven cavity`
    * :ref:`Demo - Rayleigh Benard`

Tensor products
---------------

If the problem is two-dimensional, then we need two basis functions, one per
dimension. If we call the basis function along :math:`x`-direction :math:`\mathcal{X}(x)`
and along :math:`y`-direction :math:`\mathcal{Y}(y)`, a test function can then be
computed as

.. math::

   v(x, y) = \mathcal{X}(x) \mathcal{Y}(y).

If we now have a problem that has Dirichlet boundaries in the :math:`x`-direction
and periodic boundaries in the :math:`y`-direction, then we can choose
:math:`\mathcal{X}_k(x) = T_k-T_{k+2}`, for :math:`k \in \mathcal{I}^{N-2}` (
with :math:`N-2` because :math:`T_{k+2}` then equals :math:`T_{N}` for :math:`k=N-2`),
:math:`\mathcal{Y}_l(y) = \exp(\imath l y)` for :math:`l \in \mathcal{I}^M`
and a tensor product test function is then

.. math::
   :label: eq:v2D

   v_{kl}(x, y) = (T_k(x) - T_{k+2}(x)) \exp(\imath l y), \text{ for } (k, l) \in \mathcal{I}^{N-2} \times \mathcal{I}^M.

In other words, we choose one test function per spatial dimension and create
global basis functions by taking the outer products (or tensor products) of these individual
test functions. Since global basis functions simply are the tensor products of
one-dimensional basis functions, it is trivial to move to even higher-dimensional spaces.
The multi-dimensional basis functions then form a basis for a multi-dimensional
tensor product space. The associated domains are similarily formed by taking
Cartesian products of the one-dimensional domains. For example, if the one-dimensional
domains in :math:`x`- and :math:`y`-directions are :math:`[-1, 1]` and :math:`[0, 2\pi]`, then
the two-dimensional domain formed from these two are :math:`[-1, 1] \times [0, 2\pi]`,
where :math:`\times` represents a Cartesian product.

The one-dimensional domains are discretized using the quadrature points of the
chosen basis functions. If the meshes in :math:`x`- and :math:`y`-directions are
:math:`x = \{x_i\}_{i\in \mathcal{I}^N}` and :math:`y = \{y_j\}_{j\in \mathcal{I}^M}`,
then a Cartesian product mesh is :math:`x \times y`. With index and set builder
notation it is given as

.. math::
    :label: eq:tensormesh

    x \times y = \left\{(x_i, y_j) \,|\, (i, j) \in \mathcal{I}^N \times \mathcal{I}^M\right\}.

With shenfun a user chooses the appropriate bases for each dimension of the
problem, and may then combine these bases into tensor product spaces and Cartesian
product domains. For
example, to create the required spaces for the aforementioned domain, with Dirichlet in
:math:`x`- and periodic in :math:`y`-direction, we need the following:

.. math::

    N, M &= (16, 16) \\
    B^N(x) &= \text{span}\{T_k(x)-T_{k+2}(x)\}_{k\in \mathcal{I}^{N-2}} \\
    B^M(y) &= \text{span}\{\exp(\imath l y)\}_{l\in \mathcal{I}^M} \\
    V(x, y) &= B^N(x) \otimes B^M(y)

where :math:`\otimes` represents a tensor product.

This can be implemented in `shenfun` as follows::

    from shenfun import Basis, TensorProductSpace
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    N, M = (16, 16)
    BN = Basis(N, 'Chebyshev', bc=(0, 0))
    BM = Basis(M, 'Fourier', dtype='d')
    V = TensorProductSpace(comm, (BN, BM))

Note that the Chebyshev basis is created using :math:`N` and not :math:`N-2`. The
chosen boundary condition ``bc=(0, 0)`` ensures that only :math:`N-2` bases will be used.
The Fourier basis ``BM`` has been defined for real inputs to a
forward transform, which is ensured by the ``dtype`` keyword being set to ``d``
for double. ``dtype``
specifies the data type that is input to the ``forward`` method, or the
data type of the solution in physical space. Setting
``dtype='D'`` indicates that this datatype will be complex. Note that it
will not trigger an error, or even lead to wrong results, if ``dtype`` is
by mistake set to ``D``. It is merely less efficient to work with complex data
arrays where double precision is sufficient. See Sec :ref:`sec:gettingstarted`
for more information on getting started with using bases.

Shenfun is parallelized with MPI through the `mpi4py-fft`_ package.
If we store the current example in ``filename.py``, then it can be run
with more than one processor, e.g., like::

    mpirun -np 4 python filename.py

In this case the tensor product space ``V`` will be distributed
with the *slab* method (since the problem is 2D) and it
can here use a maximum of 9 CPUs. The maximum is 9 since the last dimension is
transformed from 16 real numbers to 9 complex, using the Hermitian symmetry of
real transforms, i.e., the shape of a transformed array in the V space will be
(14, 9). You can read more about MPI in the later section :ref:`MPI`.

Tribute
-------

Shenfun is named as a tribute to Prof. Jie Shen, as it contains many
tools for working with his modified Chebyshev and Legendre bases, as
described here:

    * Jie Shen, SIAM Journal on Scientific Computing, 15 (6), 1489-1505 (1994) (JS1)
    * Jie Shen, SIAM Journal on Scientific Computing, 16 (1), 74-87, (1995) (JS2)

Shenfun has implemented classes for the bases described in these papers,
and within each class there are methods for fast transforms, inner
products and for computing matrices arising from bilinear forms in the
spectral Galerkin method.

.. _shenfun: https:/github.com/spectralDNS/shenfun
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _Demo for the nonlinear Klein-Gordon equation: https://rawgit.com/spectralDNS/shenfun/master/docs/src/KleinGordon/kleingordon_bootstrap.html
.. _Demo for the Kuramato-Sivashinsky equation: https://rawgit.com/spectralDNS/shenfun/master/docs/src/KuramatoSivashinsky/kuramatosivashinsky_bootstrap.html
.. _Demo for Poisson equation in 1D with inhomogeneous Dirichlet boundary conditions: https://rawgit.com/spectralDNS/shenfun/master/docs/src/Poisson/poisson_bootstrap.html
.. _Demo for Poisson equation in 3D with Dirichlet in one and periodicity in remaining two dimensions: https://rawgit.com/spectralDNS/shenfun/master/docs/src/Poisson3D/poisson3d_bootstrap.html
.. _Shenfun paper: https://rawgit.com/spectralDNS/shenfun/master/docs/shenfun_bootstrap.html
