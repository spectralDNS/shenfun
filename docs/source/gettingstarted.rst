.. _sec:gettingstarted:

Getting started
===============

Basic usage
-----------

Shenfun consists of classes and functions whoose purpose are to make it easier
to implement PDE's with spectral methods in simple tensor product domains. The
most important everyday tools are

	* :class:`.TensorProductSpace`
	* :class:`.TrialFunction`
	* :class:`.TestFunction`
	* :class:`.Function`
	* :class:`.Array`
	* :func:`.inner`
	* :func:`.div`
	* :func:`.grad`
	* :func:`.project`
	* :func:`.Basis`

A good place to get started is by creating a :func:`.Basis`. There are three families of
bases: Fourier, Chebyshev and Legendre. All bases are defined on a one-dimensional
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

    from sympy import Symbol
    x = Symbol('x')
    u = 2*x**2 - 1

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

    (u_h - u, v)_w = 0 \quad \forall v \in T

where :math:`v` is a test function, :math:`u_h` is a trial function and the
notation :math:`(\cdot, \cdot)_w` was introduced in :eq:`eq:wrm_test`. Using
now :math:`v=T_k` and :math:`u_h=\sum_{j=0}^7 \hat{u}_j T_j`, we get

.. math::

    (\sum_{j=0}^7 \hat{u}_j T_j, T_k)_w &= (u, T_k)_w \\
    \sum_{j=0}^7 (T_j, T_k)_w \hat{u}_j &= (u, T_k)_w

for all :math:`k \in 0, 1, \ldots, 7`. This can be rewritten on matrix form as

.. math::

    B_{kj} \hat{u}_j = \tilde{u}_k

where :math:`B_{kj} = (T_j, T_k)_w`, :math:`\tilde{u}_k = (u, T_k)_w` and
summation is implied by the repeating :math:`j` indices. Since the
Chebyshev polynomials are orthogonal the mass matrix :math:`B_{kj}` is
diagonal. We can assemble both :math:`B_{kj}` and :math:`\tilde{u}_j`
with `shenfun`, and at the same time introduce the :class:`.TestFunction`,
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

Note that the matrix :math:`B` is stored using `shenfun`'s
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

    K0 = Basis(12, 'F', dtype='D')
    K1 = Basis(12, 'F', dtype='d')
    T = TensorProductSpace(comm, (K0, K1))
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
        <ipython-input-35-de4aac99d23b> in <module>()
        ----> 1 da = Dx(ua, 0, 1)

        ~/MySoftware/shenfun/shenfun/forms/operators.py in Dx(test, x, k)
             85             Number of derivatives
             86     """
        ---> 87     assert isinstance(test, (Expr, BasisFunction))
             88
             89     if isinstance(test, BasisFunction):

So it is not possible to perform operations that involve differentiation on an
:class:`.Array` instance. This is because the ``ua`` does not contain more
information than its values and its TensorProductSpace. A :class:`.BasisFunction`
instance, on the other hand, can be manipulated with operators like :func:`.div`
:func:`.grad` in creating instances of the :class:`.Expr` class.

Any rules for efficient use of Numpy ``ndarrays``, like vectorization, also
applies to :class:`.Function` and :class:`.Array` instances.

.. include:: mpi.rst
.. include:: postprocessing.rst
.. include:: integrators.rst
