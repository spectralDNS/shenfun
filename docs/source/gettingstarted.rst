.. _sec:gettingstarted:

Getting started
===============

Basic usage
-----------

Shenfun consists of classes and functions whoose purpose are to make it easier
to implement PDE's with spectral methods in simple tensor product domains. The
most important everyday tools are

	* :class:`.TensorProductSpace`
	* :class:`.MixedTensorProductSpace`
	* :class:`.TrialFunction`
	* :class:`.TestFunction`
	* :class:`.Function`
	* :class:`.Array`
	* :func:`.inner`
	* :func:`.div`
	* :func:`.grad`
	* :func:`.project`
	* :func:`.Basis`

A good place to get started is by creating a :func:`.Basis`. There are five families of
bases: Fourier, Chebyshev, Legendre, Laguerre, Hermite and Jacobi. All bases are
defined on a one-dimensional
domain, with their own basis functions and quadrature points. For example, we have
the regular Chebyshev basis :math:`\{T_k\}_{k=0}^{N-1}`, where :math:`T_k` is the
:math:`k`'th Chebyshev polynomial of the first kind. To create such a basis with
8 quadrature points  (i.e., :math:`\{T_k\}_{k=0}^{7}`) do::

    from shenfun import Basis
    N = 8
    T = Basis(N, 'Chebyshev', bc=None)

Here ``bc=None`` is used to indicate that there are no boundary conditions associated
with this basis, which is the default, so it could just as well have been left out.
To create
a regular Legendre basis (i.e., :math:`\{L_k\}_{k=0}^{N-1}`, where :math:`L_k` is the
:math:`k`'th Legendre polynomial), just replace
``Chebyshev`` with ``Legendre`` above. And to create a Fourier basis, just use
``Fourier``.

The basis :math:`T = \{T_k\}_{k=0}^{N-1}` has many useful methods associated
with it, and we may experiment a little. A :class:`.Function` ``u`` using basis
:math:`T` has expansion

.. math::
   :label: eq:sum8

    u(x) = \sum_{k=0}^{7} \hat{u}_k T_k(x)

and an instance of this function (initialized with :math:`\{\hat{u}_k\}_{k=0}^7=0`)
is created in shenfun as::

    u = Function(T)

Consider now for exampel the polynomial :math:`2x^2-1`, which happens to be
exactly equal to :math:`T_2(x)`. We
can create this polynomial using `sympy <www.sympy.org>`_ ::

    import sympy as sp
    x = sp.Symbol('x')
    u = 2*x**2 - 1  # or simply u = sp.chebyshevt(2, x)

The Sympy function ``u`` can now be evaluated on the quadrature points of basis
:math:`T`::

    xj = T.mesh()
    ue = Array(T)
    ue[:] = [u.subs(x, xx) for xx in xj]
    print(xj)
      [ 0.98078528  0.83146961  0.55557023  0.19509032 -0.19509032 -0.55557023
       -0.83146961 -0.98078528]
    print(ue)
      [ 0.92387953  0.38268343 -0.38268343 -0.92387953 -0.92387953 -0.38268343
        0.38268343  0.92387953]

We see that ``ue`` is an :class:`.Array` on the basis ``T``, and not a
:class:`.Function`. The :class:`.Array` and :class:`Function` classes
are both subclasses of Numpy's `ndarray <https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.html>`_, and represent the two arrays associated
with the spectral Galerkin function, like :eq:`eq:sum8`.
The :class:`.Function` represent the entire spectral Galerkin function, with
array values corresponding to the expansion coefficients :math:`\hat{u}`.
The :class:`.Array` represent the spectral Galerkin function evaluated
on the quadrature mesh of the basis ``T``, i.e., here
:math:`u(x_i), \forall \, i \in 0, 1, \ldots, 7`.

We now want to find the :class:`.Function` ``uh`` corresponding to
:class:`.Array` ``ue``. Considering :eq:`eq:sum8`, this corresponds to finding
:math:`\hat{u}_k` if the left hand side :math:`u(x_j)` is known for
all quadrature points :math:`x_j`.

Since we already know that ``ue`` is
equal to the second Chebyshev polynomial, we should get an array of
expansion coefficients equal to :math:`\hat{u} = (0, 0, 1, 0, 0, 0, 0, 0)`.
We can compute ``uh`` either by using :func:`project` or a forward transform::

    uh = Function(T)
    uh = T.forward(ue, uh)
    # or
    # uh = ue.forward(uh)
    # or
    # uh = project(ue, T)
    print(uh)
      [-1.38777878e-17  6.72002101e-17  1.00000000e+00 -1.95146303e-16
        1.96261557e-17  1.15426347e-16 -1.11022302e-16  1.65163507e-16]

So we see that the projection works to machine precision.

The projection is mathematically: find :math:`u_h \in T`, such that

.. math::

    (u_h - u, v)_w = 0 \quad \forall v \in T,

where :math:`v` is a test function, :math:`u_h` is a trial function and the
notation :math:`(\cdot, \cdot)_w` was introduced in :eq:`eq:wrm_test`. Using
now :math:`v=T_k` and :math:`u_h=\sum_{j=0}^7 \hat{u}_j T_j`, we get

.. math::

    (\sum_{j=0}^7 \hat{u}_j T_j, T_k)_w &= (u, T_k)_w, \\
    \sum_{j=0}^7 (T_j, T_k)_w \hat{u}_j &= (u, T_k)_w,

for all :math:`k \in 0, 1, \ldots, 7`. This can be rewritten on matrix form as

.. math::

    B_{kj} \hat{u}_j = \tilde{u}_k,

where :math:`B_{kj} = (T_j, T_k)_w`, :math:`\tilde{u}_k = (u, T_k)_w` and
summation is implied by the repeating :math:`j` indices. Since the
Chebyshev polynomials are orthogonal the mass matrix :math:`B_{kj}` is
diagonal. We can assemble both :math:`B_{kj}` and :math:`\tilde{u}_j`
with shenfun, and at the same time introduce the :class:`.TestFunction`,
:class:`.TrialFunction` classes and the :func:`.inner` function::

    from shenfun import TestFunction, TrialFunction, inner
    u = TrialFunction(T)
    v = TestFunction(T)
    B = inner(u, v)
    u_tilde = inner(ue, v)
    print(B)
      {0: array([3.14159265, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
       1.57079633, 1.57079633, 1.57079633])}
    print(u_tilde)
      [-4.35983562e-17  1.05557843e-16  1.57079633e+00 -3.06535096e-16
        3.08286933e-17  1.81311282e-16 -1.74393425e-16  2.59438230e-16]

The :func:`.inner` function represents the inner product and it expects
one test function, and possibly one trial function. If, as here, it also
contains a trial function, then a matrix is returned. If :func:`.inner`
contains one test, but no trial function, then an array is returned.

Note that the matrix :math:`B` is stored using shenfun's
:class:`.SpectralMatrix` class, which is a subclass of Python's dictionary,
where the keys are the diagonals and the values are the diagonal entries.
The matrix :math:`B` is seen to have only one diagonal (the principal)
:math:`\{B_{ii}\}_{i=0}^{7}`.

With the matrix comes a `solve` method and we can solve for :math:`\hat{u}`
through::

    u_hat = Function(T)
    u_hat = B.solve(u_tilde, u=u_hat)
    print(u_hat)
      [-1.38777878e-17  6.72002101e-17  1.00000000e+00 -1.95146303e-16
        1.96261557e-17  1.15426347e-16 -1.11022302e-16  1.65163507e-16]

which obviously is exactly the same as we found using :func:`.project`
or the `T.forward` function.

Note that :class:`.Array` merely is a subclass of Numpy's ``ndarray``,
whereas :class:`.Function` is a subclass
of both Numpy's ``ndarray`` *and* the :class:`.BasisFunction` class. The
latter is used as a base class for arguments to bilinear and linear forms,
and is as such a base class also for :class:`.TrialFunction` and
:class:`.TestFunction`. An instance of the :class:`.Array` class cannot
be used in forms, except from regular inner products of test function
vs an :class:`.Array`. To illustrate, lets create some forms, where
all except the last one is ok::

    T = Basis(12, 'Legendre')
    u = TrialFunction(T)
    v = TestFunction(T)
    uf = Function(T)
    ua = Array(T)
    A = inner(v, u)   # Mass matrix
    c = inner(v, ua)  # ok, a scalar product
    d = inner(v, uf)  # ok, a scalar product (slower than above)
    df = Dx(uf, 0, 1) # ok
    da = Dx(ua, 0, 1) # Not ok

        AssertionError                            Traceback (most recent call last)
        <ipython-input-14-3b957937279f> in <module>
        ----> 1 da = inner(v, Dx(ua, 0, 1))

        ~/MySoftware/shenfun/shenfun/forms/operators.py in Dx(test, x, k)
             82         Number of derivatives
             83     """
        ---> 84     assert isinstance(test, (Expr, BasisFunction))
             85
             86     if isinstance(test, BasisFunction):

        AssertionError:

So it is not possible to perform operations that involve differentiation on an
:class:`.Array` instance. This is because the ``ua`` does not contain more
information than its values and its TensorProductSpace. A :class:`.BasisFunction`
instance, on the other hand, can be manipulated with operators like :func:`.div`
:func:`.grad` in creating instances of the :class:`.Expr` class, see
:ref:`operators`.

Note that any rules for efficient use of Numpy ``ndarrays``, like vectorization,
also applies to :class:`.Function` and :class:`.Array` instances.

.. _operators:

Operators
---------

Operators act on any single instance of a :class:`.BasisFunction`, which can
be :class:`.Function`, :class:`.TrialFunction` or :class:`.TestFunction`. The
implemented operators are:

	* :func:`.div`
	* :func:`.grad`
	* :func:`.curl`
	* :func:`.Dx`

Operators are used in variational forms assembled using :func:`.inner`
or :func:`.project`, like::

    A = inner(grad(u), grad(v))

which assembles a stiffness matrix A. Note that the two expressions fed to
inner must have consistent rank. Here, for example, both ``grad(u)`` and
``grad(v)`` have rank 1 of a vector.


Multidimensional problems
-------------------------

As described in the introduction, a multidimensional problem is handled using
tensor product spaces, that are outer products of one-dimensional bases. We
create tensor product spaces using the class :class:`.TensorProductSpace`::

    N, M = (12, 16)
    C0 = Basis(N, 'L', bc=(0, 0), scaled=True)
    K0 = Basis(M, 'F', dtype='d')
    T = TensorProductSpace(comm, (C0, K0))

The tensor product mesh will now be :math:`[-1, 1] \times [0, 2\pi]`. We use
classes :class:`.Function`, :class:`.TrialFunction` and :class:`TestFunction`
exactly as before::

    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(grad(u), grad(v))

However, now ``A`` will be a tensor product matrix, or more correctly,
the sum of two tensor product matrices. This can be seen if we look at
the equations beyond the code. In this case we are using a composite
Legendre basis for the first direction and Fourier exponentials for
the second, and the tensor product basis function is

.. math::

    v_{kl}(x, y) &= \frac{1}{\sqrt{4k+6}}(L_k(x) - L_{k+2}(x)) \exp(\imath l y), \\
                 &= \Psi_k(x) \phi_l(y),

where :math:`L_k` is the :math:`k`'th Legendre polynomial,
:math:`\psi_k = (L_k-L_{k+2})/\sqrt{4k+6}` and :math:`\phi_l = \exp(\imath l y)` are used
for simplicity in later derivations. The trial function becomes

.. math::

    u(x, y) = \sum_k \sum_l \hat{u}_{kl} v_{kl}

and the inner product is

.. math::
    :label: eq:poissons

    (\nabla u, \nabla v)_w &= \int_{-1}^{1} \int_{0}^{2 \pi} \nabla u \cdot \nabla v dxdy, \\
                           &= \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}\frac{\partial v}{\partial y} dxdy, \\
                           &= \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} dxdy + \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial u}{\partial y} \frac{\partial v}{\partial y} dxdy,

showing that it is the sum of two tensor product matrices. However, each one of these two
terms contains the outer product of smaller matrices. To see this we need to insert for the
trial and test functions (using :math:`v_{mn}` for test):

.. math::
     \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} dxdy &= \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial}{\partial x} \left( \sum_k \sum_l \hat{u}_{kl} \Psi_k(x) \phi_l(y) \right) \frac{\partial}{\partial x} \left( \Psi_m(x) \phi_n(y)  \right)dxdy, \\
          &= \sum_k \sum_l \underbrace{ \int_{-1}^{1}  \frac{\partial \Psi_k(x)}{\partial x} \frac{\partial \Psi_m(x)}{\partial x} dx}_{A_{mk}} \underbrace{ \int_{0}^{2 \pi} \phi_l(y) \phi_{n}(y) dy}_{B_{nl}} \, \hat{u}_{kl},

where :math:`A \in \mathbb{R}^{N-2 \times N-2}` and :math:`B \in \mathbb{R}^{M \times M}`.
The tensor product matrix :math:`A_{mk} B_{nl}` (or in matrix notation :math:`A \otimes B`)
is the first item of the two
items in the list that is returned by ``inner(grad(u), grad(v))``. The other
item is of course the second term in the last line of :eq:`eq:poissons`:

.. math::
     \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial u}{\partial y} \frac{\partial v}{\partial y} dxdy &= \int_{-1}^{1} \int_{0}^{2 \pi} \frac{\partial}{\partial y} \left( \sum_k \sum_l \hat{u}_{kl} \Psi_k(x) \phi_l(y) \right) \frac{\partial}{\partial y} \left(\Psi_m(x) \phi_n(y) \right) dxdy \\
          &= \sum_k \sum_l \underbrace{ \int_{-1}^{1}  \Psi_k(x) \Psi_m(x) dx}_{C_{mk}} \underbrace{ \int_{0}^{2 \pi} \frac{\partial \phi_l(y)}{\partial y} \frac{ \phi_{n}(y) }{\partial y} dy}_{D_{nl}} \, \hat{u}_{kl}

The tensor product matrices :math:`A_{mk} B_{nl}` and :math:`C_{mk}D_{nl}` are both instances
of the :class:`.TPMatrix` class. Together they lead to linear algebra systems
like:

.. math::
    :label: eq:multisystem

    (A_{mk}B_{nl} + C_{mk}D_{nl}) \hat{u}_{kl} = \tilde{f}_{mn},

where

.. math::

    \tilde{f}_{mn} = (v, f)_w,

for some right hand side :math:`f`, see, e.g., :eq:`eq:poissonmulti`. Note that
an alternative formulation here is

.. math::

    A \hat{u} B^T + C \hat{u} D^T = \tilde{f}

where :math:`\hat{u}` and :math:`\tilde{f}` are treated as regular matrices
(:math:`\hat{u} \in \mathbb{R}^{N-2 \times M}` and :math:`\tilde{f} \in \mathbb{R}^{N-2 \times M}`).
This formulation is utilized to derive efficient solvers for tensor product bases
in multiple dimensions using the matrix decomposition
method in :cite:`shen1` and :cite:`shen95`.

Note that in our case the equation system :eq:`eq:multisystem` can be greatly simplified since
three of the submatrices (:math:`A_{mk}, B_{nl}` and :math:`D_{nl}`) are diagonal.
Even more, two of them equals the identity matrix

.. math::

    A_{mk} &= \delta_{mk}, \\
    B_{nl} &= \delta_{nl},

whereas the last one can be written in terms of the identity
(no summation on repeating indices)

.. math::

    D_{nl} = -nl\delta_{nl}.

Inserting for this in :eq:`eq:multisystem` and simplifying by requiring that
:math:`l=n` in the second step, we get

.. math::
    :label: eq:matfourier

    (\delta_{mk}\delta_{nl} - ln C_{mk}\delta_{nl}) \hat{u}_{kl} &= \tilde{f}_{mn}, \\
    (\delta_{mk} - l^2 C_{mk}) \hat{u}_{kl} &= \tilde{f}_{ml}.

Now if we keep :math:`l` fixed this latter equation is simply a regular
linear algebra problem to solve for :math:`\hat{u}_{kl}`, for all :math:`k`.
Of course, this solve needs to be carried out for all :math:`l`.

Note that there is a generic solver available for the system
:eq:`eq:multisystem` in :class:`.SolverGeneric2NP` that makes no
assumptions on diagonality. However, this solver will, naturally, be
quite a bit slower than a tailored solver that takes advantage of
diagonality. For the Poisson equation such solvers are available for
both Legendre and Chebyshev bases, see the extended demo :ref:`Demo - 3D Poisson equation`
or the demo programs `dirichlet_poisson2D.py <https://github.com/spectralDNS/shenfun/blob/master/demo/dirichlet_poisson2D.py>`_
and `dirichlet_poisson3D.py <https://github.com/spectralDNS/shenfun/blob/master/demo/dirichlet_poisson3D.py>`_.

Coupled problems
----------------

With Shenfun it is possible to solve equations coupled and implicit using the
:class:`.MixedTensorProductSpace` class for multidimensional problems and
:class:`.MixedBasis` for one-dimensional problems. As an example, lets consider
a mixed formulation of the Poisson equation. The Poisson equation is given as
always as

.. math::
    :label: eq:poissonmulti

    \nabla^2 u(\boldsymbol{x}) = f(\boldsymbol{x}), \quad \text{for} \quad \boldsymbol{x} \in \Omega,

but now we recast the problem into a mixed formulation

.. math::

    \sigma(\boldsymbol{x})- \nabla u (\boldsymbol{x})&= 0,  \quad \text{for} \quad \boldsymbol{x} \in \Omega, \\
    \nabla \cdot \sigma (\boldsymbol{x})&= f(\boldsymbol{x}), \quad \text{for} \quad \boldsymbol{x} \in \Omega.

where we solve for the vector :math:`\sigma` and scalar :math:`u` simultaneously. The
domain :math:`\Omega` is taken as a multidimensional tensor product, with
one inhomogeneous direction. Here we will consider the 2D domain
:math:`\Omega=[-1, 1] \times [0, 2\pi]`, but the code is more or less identical for
a 3D problem. For boundary conditions we use Dirichlet in the :math:`x`-direction and
periodicity in the :math:`y`-direction:

.. math::

    u(\pm 1, y) &= 0 \\
    u(x, 2\pi) &= u(x, 0)

Note that there is no boundary condition on :math:`\sigma`, only on :math:`u`.
For this reason we choose a Dirichlet basis :math:`SD` for :math:`u` and a regular
Legendre or Chebyshev :math:`ST` basis for :math:`\sigma`. Since :math:`\sigma` is
a vector we use a :class:`.VectorTensorProductSpace` :math:`VT = ST \times ST` and
finally a :class:`.MixedTensorProductSpace` :math:`Q = VT \times TD` for the coupled and
implicit treatment of :math:`(\sigma, u)`::

    N, M = (16, 24)
    family = 'Legendre'
    SD = Basis(N[0], family, bc=(0, 0))
    ST = Basis(N[0], family)
    K0 = Basis(N[1], 'Fourier', dtype='d')
    TD = TensorProductSpace(comm, (SD, K0), axes=(0, 1))
    TT = TensorProductSpace(comm, (ST, K0), axes=(0, 1))
    VT = VectorTensorProductSpace(TT)
    Q = MixedTensorProductSpace([VT, TD])

In variational form the problem reads: find :math:`(\sigma, u) \in Q`
such that

.. math::
    :label: eq:coupled

    (\sigma, \tau)_w - (\nabla u, \tau)_w &= 0, \quad \forall \tau \in VT, \\
    (\nabla \cdot \sigma, v)_w  &= (f, v)_w \quad \forall v \in TD

To implement this we use code that is very similar to regular, uncoupled
problems. We create test and trialfunction::

    gu = TrialFunction(Q)
    tv = TestFunction(Q)
    sigma, u = gu
    tau, v = tv

and use these to assemble all blocks of the variational form :eq:`eq:coupled`::

    # Assemble equations
    A00 = inner(sigma, tau)
    if family.lower() == 'legendre':
        A01 = inner(u, div(tau))
    else:
        A01 = inner(-grad(u), tau)
    A10 = inner(div(sigma), v)

Note that we here can use integration by parts for Legendre, since the weight function
is a constant, and as such get the term :math:`(-\nabla u, \tau)_w = (u, \nabla \cdot \tau)_w`
(boundary term is zero due to homogeneous Dirichlet boundary conditions).

We collect all assembled terms in a :class:`.BlockMatrix`::

    H = BlockMatrix(A00+A01+A10)

This block matrix ``H`` is then simply (for Legendre)

.. math::
    :label: eq:coupledH

    \begin{bmatrix}
        (\sigma, \tau)_w & (u, \nabla \cdot \tau)_w \\
        (\nabla \cdot \sigma, v)_w & 0
    \end{bmatrix}

Note that each item in :eq:`eq:coupledH` is a collection of instances of the
:class:`.TPMatrix` class, and for similar reasons as given around :eq:`eq:matfourier`,
we get also here one regular block matrix for each Fourier wavenumber.
The sparsity pattern is the same for all matrices except for wavenumber 0.
The (highly sparse) sparsity pattern for block matrix :math:`H` with
wavenumber :math:`\ne 0` is shown in the image below

.. image:: Sparsity.png

A complete demo for the coupled problem discussed here can be found in
`MixedPoisson.py <https://github.com/spectralDNS/shenfun/blob/master/demo/MixedPoisson.py>`_
and a 3D version is in `MixedPoisson3D.py <https://github.com/spectralDNS/shenfun/blob/master/demo/MixedPoisson3D.py>`_.

.. include:: integrators.rst
.. include:: mpi.rst
.. include:: postprocessing.rst
