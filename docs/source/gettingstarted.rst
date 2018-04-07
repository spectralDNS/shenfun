Getting started
===============

Shenfun consists of classes and functions whoose purpose are to make it easy to implement
PDE's in simple tensor product domains. The most important everyday tools are

	* :class:`.TensorProductSpace`
	* :class:`.TrialFunction`
	* :class:`.TestFunction`
	* :class:`.Function`
	* :func:`.div`
	* :func:`.grad`
	* :func:`.project`
	* :func:`.Basis`

To get started you need to create a :func:`Basis`. There are three families of
bases: Fourier, Chebyshev and Legendre. All bases are defined on a one-dimensional
domain, with their own basis functions and quadrature points. For example, we have 
the regular Chebyshev basis :math:`\{T_k\}_{k=0}^{N-1}`, where :math:`T_k` is the 
:math:`k`'th Chebyshev polynomial of the first kind. To create such a basis with
8 quadrature points  (i.e., :math:`\{T_k\}_{k=0}^{7}`) do::

    from shenfun import Basis
    N = 8
    T = Basis(N, 'Chebyshev', plan=True, bc=None)

Here `bc=None` is used to indicate that there are no boundary conditions associated
with this basis, which is the default, so it could just as well have been left out.
The `plan=True` is included to indicate that the `Basis` class can go ahead and
plan its forward and backward transforms, that are to be used later on.

The basis :math:`T` has many useful methods associated with it, and we may
experiment a little. A function using basis :math:`T` has expansion

.. math::
   :label: eq:sum8

    u(x) = \sum_{k=0}^{7} \hat{u}_k T_k

Consider now for exampel the polynomial :math:`2x^2-1`, which is
exactly equal to :math:`T_2(x)`. We know that nn expansion in the :math:`T`
basis should be exactly equal to :math:`\hat{u} = (0, 0, 1, 0, 0, 0, 0, 0)`. We
can create the polynomial using `sympy <www.sympy.org>`_ ::

    from sympy import Symbol
    x = Symbol('x')
    u = 2*x**2 - 1

The Sympy function `u` can now be evaluated on the quadrature points of basis
`T`::

    xj = T.mesh(N)
    ue = Function(T, False)
    ue[:] = [u.subs(x, xx) for xx in xj]
    print(xj)
      [ 0.98078528  0.83146961  0.55557023  0.19509032 -0.19509032 -0.55557023
       -0.83146961 -0.98078528]
    print(ue)
      [ 0.92387953  0.38268343 -0.38268343 -0.92387953 -0.92387953 -0.38268343
        0.38268343  0.92387953]

We see that `ue` is a :class:`.Function` on the basis `T`, and the `False` is there
to indicate that this function lives in the real physical space. That is, it is
the left hand side :math:`u(x)` of :eq:`eq:sum8`. If we change from `False` to `True`,
the we get :math:`\hat{u}` on the right hand side::

    u_hat = Function(T, True)

We now want the expansion of :class:`.Function` `ue` in `T`, that is, we want to
compute the :math:`\hat{u}` corresponding to `ue`. Since we know that `ue` is
equal to the second Chebyshev polynomial, we should get
:math:`\hat{u} = (0, 0, 1, 0, 0, 0, 0, 0)`. We can compute `u_hat` either
by using :func:`project` or a forward transform::

    u_hat = T.forward(ue, u_hat)
    # or
    # u_hat = project(ue, T, output_array=u_hat)
    print(u_hat)
      [-1.38777878e-17  6.72002101e-17  1.00000000e+00 -1.95146303e-16
        1.96261557e-17  1.15426347e-16 -1.11022302e-16  1.65163507e-16]

So we see that the projection works to machince precision.

