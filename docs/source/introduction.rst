Introduction
============

Spectral Galerkin
-----------------

The spectral Galerkin method uses the method of weighted residuals, and
solves PDEs by first creating variational forms from inner products,

.. math::
    :label: intro:varform

    (u, v)_w = \int_{\Omega} u(\boldsymbol{x}) \overline{v}(\boldsymbol{x}) w(\boldsymbol{x}) d\boldsymbol{x} 

where :math:`\Omega` is the computational domain, :math:`u` is a trial 
function, :math:`v` a test function (overline indicates a complex conjugate),
and :math:`w` is a weight function. The bold :math:`\boldsymbol{x}` represents 
:math:`(x,y,z)` for a 3D inner product, but shenfun may be used for any number 
of dimensions.

Consider the Poisson equation with homogeneous Dirichlet boundary conditions
and a right hand side function :math:`f(\boldsymbol{x})`

.. math::
   :label: eq:poisson1

    -\nabla^2 u(\boldsymbol{x}) &= f(\boldsymbol{x}) \quad \text{for } \, \boldsymbol{x} \in \Omega \\
              u(\boldsymbol{x}) &= 0\, \text{for }\, \boldsymbol{x} \in \Gamma

To solve this problem with the spectral Galerking method we need to create a 
variational form by first multiplying the entire equation by a test function 
:math:`\overline{v}` that satisfies the boundary conditions: :math:`v(\boldsymbol{x}) = 0`
for :math:`\boldsymbol{x} \, \in \Gamma`. We also multiply by an associated 
weight function :math:`w`, which is chosen on basis on obtaining orthogonal
inner products. For basis functions composed of Fourier or Legendre polynomials
the weight function is simply a constant, but for Chebyshev the required weight
function is :math:`1/\sqrt{1-x^2}`. After multiplying :eq:`eq:poisson1` with 
:math:`v(\boldsymbol{x}) w(\boldsymbol{x})` we then integrate over the domain 
to obtain the variational form

.. math::
   :label: eq:poisson2

    (-\nabla^2 u, v)_w = (f, v)_w   

The solution to the Poisson equation is found in the Hilbert space :math:`H^1(\Omega)`
and the variational problem is solved as: find :math:`u \in H^1` such that

.. math::
   :label: eq:poisson3

    (-\nabla^2 u, v)_w = (f, v)_w \quad \forall v \, \in H^1_0   

where :math:`H^1_0` represents :math:`H^1` restricted to the homogeneous Dirichlet
boundary condition.

Equation :eq:`eq:poisson3` is formulated in continuous space. To solve this
equation on a computer we need to discretize using a finite number of test
functions :math:`v`, and we need to evaluate the inner products numerically
using quadrature. 

The choice of :math:`v(\boldsymbol{x})` is highly sentral to the method. 
In one spatial dimension typical choices for :math:`v` are

.. math::

   v_k(x) &= T_k(x) \\
   v_k(x) &= T_k(x) - T_{k+2}(x) \\
   v_k(x) &= L_k(x) \\
   v_k(x) &= L_k(x) - L_{k+2}(x) \\ 
   v_k(x) &= \exp(\imath k x)
   
where :math:`T_k, L_k` are the :math:`k`'th Chebyshev polynomial of the first kind
and the :math:`k`'th Legendre polynomial, respectively. Note that the second
and fourth functions above satisfy the homogeneous Dirichlet boundary conditions 
:math:`v(\pm 1) = 0`, and as such these basis functions may be used to solve
the Poisson equation :eq:`eq:poisson1`.

Tensor products
---------------

If the problem is two-dimensional, then we need two basis functions, one per
dimension. If we call the basis function along :math:`x`-direction :math:`\mathcal{X}(x)`
and along :math:`y`-direction as :math:`\mathcal{Y}(y)`, the test function is then 
computed as

.. math::

   v(x, y) = \mathcal{X}(x) \mathcal{Y}(y)

If we now have a problem with Dirichlet in :math:`x`-direction and periodic in
:math:`y`-direction, then we can choose :math:`\mathcal{X}_k(x) = T_k-T_{k+2}`,
:math:`\mathcal{Y}_l(y) = \exp(\imath l y)` and a test function is then

.. math::
   :label: eq:v2D

   v_{k, l}(x, y) = (T_k(x) - T_{k+2}(x)) \exp(\imath l y)

In other words, we choose one test function per dimension and create
global basis functions by taking the outer products of these individual
test functions. Moving to even more dimensions is then trivial, as
global basis functions simply are the products of one-dimensional basis
functions. Combining one-dimensional bases like this results in
tensor product spaces, with tensor product meshes. If the one-dimensional
meshes in :math:`x`- and :math:`y`-directions are :math:`x = \{x_m\}_{m=0}^{N-1}`
and :math:`y = \{y_n\}_{n=0}^{M-1}`, then a tensor product mesh :math:`X` is
the outer product of these two vectors

.. math::

    X_{m, n} = x_m y_n, \text{for } m=0,1,\ldots, N-1, \, n=0,1,\ldots,M-1

Likewise, a tensor product basis is given in :eq:`eq:v2D`. 

With shenfun a user chooses the appropriate bases for each dimension of the
problem, and may then combine these bases into tensor product spaces. For
example, to create a basis for the aforementioned domain, with Dirichlet in
:math:`x`- and periodic in :math:`y`-direction, a user may proceed
as follows

>>> from shenfun import Basis, TensorProductSpace
>>> from mpi4py import MPI
>>> comm = MPI.COMM_WORLD
>>> N = (14, 16)
>>> B0 = Basis(N[0], 'Chebyshev', bc=(0, 0))
>>> B1 = Basis(N[1], 'Fourier', dtype='d')
>>> V = TensorProductSpace(comm, (B0, B1))

where the Fourier basis ``B1`` is for real-to-complex transforms, which is
ensured by the ``dtype`` keyword being set to ``d`` for double. ``dtype``
specifies the data type that is input to the ``forward`` method, or the
data type of the solution in physical space. Setting
``dtype='D'`` indicates that this datatype will be complex. Note that it
will not trigger an error, or even lead to wrong results, if ``dtype`` is
by mistake set to ``D``. It is merely less efficient to work with complex data
arrays where double precision is sufficient. 

The tensor product space ``V`` will be distributed with the *slab* method and it
can here use a maximum of 9 CPUs (9 since the last dimension is
transformed from 16 real data to 9 complex, using the Hermitian symmetry of
real transforms, i.e., the shape of a transformed array in the V space will be
(14, 9)).

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

