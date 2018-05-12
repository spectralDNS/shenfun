MPI
===

Shenfun makes use of the Message Passing Interface (MPI) to solve problems on
distributed memory architectures. OpenMP is also possible to enable for FFTs.

Dataarrays in Shenfun are distributed using a `new and completely generic method <https://arxiv.org/abs/1804.09536>`_, that allows for any index of a multidimensional array to be
distributed. To illustrate, lets consider a :class:`.TensorProductSpace`
of three dimensions, such that the arrays living in this space will be 
3-dimensional. We create two spaces that are identical, except from the MPI
decomposition, and we use 4 CPUs (``mpirun -np 4 python mpitest.py``, if we
store the code in this section as ``mpitest.py``)::

    from shenfun import *
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    N = (20, 40, 60)
    K0 = Basis(N[0], 'F', dtype='D', domain=(0, 1))
    K1 = Basis(N[1], 'F', dtype='D', domain=(0, 2))
    K2 = Basis(N[2], 'F', dtype='d', domain=(0, 3))
    T0 = TensorProductSpace(comm, (K0, K1, K2), axes=(0, 1, 2), slab=True)
    T1 = TensorProductSpace(comm, (K0, K1, K2), axes=(1, 0, 2), slab=True)

Here the keyword ``slab`` determines that only *one* index set of the 3-dimensional
arrays living in ``T0`` or ``T1`` should be distributed. The defaul is to use
two, which corresponds to a so-called pencil decomposition. The ``axes``-keyword 
determines the order of which transforms are conducted, starting from last to 
first in the given tuple. Note that ``T0`` now will give arrays in real physical 
space that are distributed in the first index, whereas ``T1`` will give arrays 
that are distributed in the second. This is because 0 and
1 are the first items in the tuples given to ``axes``.

We can now create some :class:`.Function` s on these spaces::

    u0 = Function(T0, False, val=comm.Get_rank())
    u1 = Function(T1, False, val=comm.Get_rank())

such that ``u0`` and ``u1`` have values corresponding to their communicating 
processors rank in the ``COMM_WORLD`` group (the group of all CPUs).

Note that both the :class:`.TensorProductSpace` s have functions with expansion

.. math::
   :label: u_fourier
        
        u(x, y, z) = \sum_{n=-N/2}^{N/2-1}\sum_{m=-N/2}^{N/2-1}\sum_{l=-N/2}^{N/2-1}
        \hat{u}_{l,m,n} e^{\imath (lx + my + nz)}.

where :math:`u(x, y, z)` is the continuous solution in real physical space, and :math:`\hat{u}`
are the spectral expansion coefficients. If we evaluate expansion :eq:`u_fourier`
on the real physical mesh, then we get

.. math::
   :label: u_fourier_d
        
        u(x_i, y_j, z_k) = \sum_{n=-N/2}^{N/2-1}\sum_{m=-N/2}^{N/2-1}\sum_{l=-N/2}^{N/2-1}
        \hat{u}_{l,m,n} e^{\imath (lx_i + my_j + nz_k)}.

The function :math:`u(x_i, y_j, z_k)` corresponds to the functions ``u0, u1``, whereas
we have not yet computed the array :math:`\hat{u}`. We could get :math:`\hat{u}` as::

    u0_hat = Function(T0)
    u0_hat = T0.forward(u0, u0_hat)

Now, ``u0`` and ``u1`` have been created on the same mesh, which is a structured
mesh of shape :math:`(20, 40, 60)`. However, since they have different MPI
decomposition, the values used to fill them on creation will differ. We can
visualize the arrays in Paraview using some postprocessing tools, to be further
described in Sec :ref:`Postprocessing`::


    h5file0 = HDF5Writer('my0file.h5', ['u0'], T0)
    h5file1 = HDF5Writer('my1file.h5', ['u1'], T1)

    h5file0.write_tstep(0, u0)
    h5file1.write_tstep(0, u1)

    h5file0.close()
    h5file1.close()

    generate_xdmf('my0file.h5')
    generate_xdmf('my1file.h5')

And when ``my0file.xdmf`` and ``my1file.xdmf`` are opened in Paraview, we
can see the different distributions. The function ``u0`` is shown first, and
we see that it has different values along the short first dimension. The
second figure is evidently distributed along the second dimension. Both
arrays are non-distributed in the third and final dimension, which is
fortunate, because this axis will be the first to be transformed in, e.g.,
``u0_hat = T0.forward(u0, u0_hat)``.

.. image:: datastructures0.png
    :width: 250px
    :height: 200px

.. image:: datastructures1.png
    :width: 250px
    :height: 200px

We can now decide to distribute not just one, but the first two axes using 
a pencil decomposition instead. This is achieved simply by dropping the
slab keyword::

    T2 = TensorProductSpace(comm, (K0, K1, K2), axes=(0, 1, 2))
    u2 = Function(T2, False, val=comm.Get_rank())
    pencilfile = HDF5Writer('pencilfile.h5', ['u0'], T2)
    pencilfile.write_tstep(0, u2)
    pencilfile.close()
    generate_xdmf('pencilfile.h5')

Running again with 4 CPUs the array ``u2`` will look like:

.. _pencil:

.. image:: datastructures_pencil0.png
    :width: 250px
    :height: 200px

The local slices into the global array may be obtained through::

    >>> print(comm.Get_rank(), T2.local_slice(False))
    0 [slice(0, 10, None), slice(0, 20, None), slice(0, 60, None)]
    1 [slice(0, 10, None), slice(20, 40, None), slice(0, 60, None)]
    2 [slice(10, 20, None), slice(0, 20, None), slice(0, 60, None)]
    3 [slice(10, 20, None), slice(20, 40, None), slice(0, 60, None)]

In spectral space the distribution will be different. This is because the
discrete Fourier transforms are performed one axis at the time, and for
this to happen the dataarrays need to be realigned to get entire axis available
for each processor. Naturally, for the array in the pencil example 
:ref:`(see image) <pencil>`, we can only perform an
FFT over the third and longest axis, because only this axis is locally available to all
processors. To do the other directions, the dataarray must be realigned and this
is done internally by the :class:`.TensorProductSpace` class. 
The shape of the datastructure in spectral space, that is
the shape of :math:`\hat{u}`, can be obtained as::

    >>> print(comm.Get_rank(), T2.local_slice(True))
    0 [slice(0, 20, None), slice(0, 20, None), slice(0, 16, None)]
    1 [slice(0, 20, None), slice(0, 20, None), slice(16, 31, None)]
    2 [slice(0, 20, None), slice(20, 40, None), slice(0, 16, None)] 
    3 [slice(0, 20, None), slice(20, 40, None), slice(16, 31, None)]

Evidently, the spectral space is distributed in the last two axes, whereas
the first axis is locally avalable to all processors. Tha dataarray
is said to be aligned in the first dimension.
