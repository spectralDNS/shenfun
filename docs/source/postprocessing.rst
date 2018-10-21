.. _Postprocessing:

Post processing
---------------

MPI is great because it means that you can run Shenfun on pretty much
as many CPUs as you can get your hands on. However, MPI makes it more
challenging to do visualization, in particular with Python and Matplotlib. For
this reason there is a :mod:`.utilities` module with helper classes for dumping dataarrays
to `HDF5 <https://www.hdf5.org>`_ or `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_

The helper functions and classes are

    * :class:`.HDF5File`
    * :class:`.NCFile`
    * :func:`.ShenfunFile`

where :func:`.ShenfunFile` is a common interface, returning an instance of
either :class:`.HDF5File` or :class:`.NCFile`, depending on choice.

For example, to create an HDF5 writer for a 3D
TensorProductSpace with Fourier bases in all directions::

    from shenfun import *
    from mpi4py import MPI
    N = (24, 25, 26)
    K0 = Basis(N[0], 'F', dtype='D')
    K1 = Basis(N[1], 'F', dtype='D')
    K2 = Basis(N[2], 'F', dtype='d')
    T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2))
    fl = ShenfunFile('myh5file', T, backend='hdf5', mode='w')

The file instance `fl` will now have two method that can be used to either ``write``
dataarrays to file, or ``read`` them back again.

    * ``fl.write``
    * ```fl.read``

With the ``HDF5`` backend we can write
both arrays from physical space (:class:`.Array`), as well as spectral space
(:class:`.Function`). However, the ``NetCDF4`` backend cannot handle complex dataarrays,
and as such it can only be used for real physical dataarrays.

In addition to storing complete dataarrays, we can also store any slices of the arrays.
To illustrate, this is how to store three snapshots of the ``u`` array, along with
some *global* 2D and 1D slices::

    u = Array(T)
    u[:] = np.random.random(T.forward.input_array.shape)
    d = {'u': [u, (u, np.s_[4, :, :]), (u, np.s_[4, 4, :])]}
    fl.write(0, d)
    u[:] = 2
    fl.write(1, d)
    fl.close()

The :class:`.ShenfunFile` may also be used for the :class:`.MixedTensorProductSpace`,
or :class:`.VectorTensorProductSpace`, that are collections of the scalar
:class:`.TensorProductSpace`. We can create a :class:`.MixedTensorProductSpace`
consisting of two TensorProductSpaces, and an accompanying writer class as::

    TT = MixedTensorProductSpace([T, T])
    fl_m = ShenfunFile('mixed', TT, backend='hdf5', mode='w')

Let's now consider a transient problem where we step a solution forward in time.
We create a solution array from the :class:`.Array` class, and update the array
inside a while loop::

    TT = VectorTensorProductSpace(T)
    fl_m = ShenfunFile('mixed', TT, backend='hdf5', mode='w')
    uf = Array(TT)
    tstep = 0
    du = {'uf': (uf,
                (uf, [slice(None), 4, slice(None), slice(None)]),
                (uf, [0, slice(None), slice(None), 10]))}
    while tstep < 3:
        fl_m.write(tstep, du, forward_output=False)
        tstep += 1
    fl_m.close()

Note that on each time step the first two arrays
``uf`` and ``(uf, [slice(None), 4, slice(None), slice(None)])``
are vectors, and as such of global shape ``(3, 24, 25, 26)`` and ``(3, 25, 26)``,
respectively. The final dumped array ``(uf, [0, slice(None), slice(None), 10])``
is a scalar since we choose only to store component 0, and the global shape is
``(24, 25)``.

Note that the slices in the above dictionaries
are *global* views of the global arrays, that may or may not be distributed
over any number of processors.

After running the above, the different arrays will be found in groups
stored in `myyfile.h5` with directory tree structure as::

    myh5file.h5/
    ├─ u/
    |  ├─ 1D/
    |  |  └─ 0_20_slice/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 3
    |  ├─ 2D/
    |  |  └─ 0_slice_slice/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 2
    |  └─ 3D/
    |     ├─ 0
    |     ├─ 1
    |     └─ 2
    └─ mesh/
       ├─ x0
       ├─ x1
       └─ x2

Likewise, the `mixed.h5` file will at the end of the loop look like::

    mixed.h5/
    ├─ uf/
    |  ├─ 2D/
    |  |  └─ slice_slice_10/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 3
    |  ├─ 2D_Vector/
    |  |  └─ 4_slice_slice/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 2
    |  └─ 3D_Vector/
    |     ├─ 0
    |     ├─ 1
    |     └─ 2
    └─ mesh/
       ├─ x0
       ├─ x1
       └─ x2

Note that the mesh is stored as well as the results. The three mesh arrays are
all 1D arrays, representing the domain for each basis in the TensorProductSpace.
Also note that these routines work with any number of CPUs and dimensions.

With NetCDF4 the layout is somewhat different. For ``mixed`` above,
if we were using backend ``netcdf`` instead of ``hdf5``,
we would get a datafile where ``ncdump -h mixed.nc`` would result in::

    netcdf mixed {
    dimensions:
            time = UNLIMITED ; // (3 currently)
            x = 24 ;
            y = 25 ;
            z = 26 ;
            dim = 3 ;
    variables:
            double time(time) ;
            double x(x) ;
            double y(y) ;
            double z(z) ;
            int64 dim(dim) ;
            double uf(time, dim, x, y, z) ;
            double uf_4_slice_slice(time, dim, y, z) ;
            double uf_slice_slice_10(time, x, y) ;

    // global attributes:
                    :ndim = 3LL ;
                    :shape = 3LL, 24LL, 25LL, 26LL ;
    }


Note that it is also possible to store vector arrays as scalars. For NetCDF4 this
is necessary for direct visualization using `Visit <https://www.visitusers.org>`_.
To store vectors as scalars, simply use::

    fl_m.write(tstep, du, forward_output=False, as_scalar=True)

ParaView
********

The stored datafiles can be visualized in `ParaView <www.paraview.org>`_.
However, ParaView cannot understand the content of these HDF5-files without
a little bit of help. We have to explain that these data-files contain
structured arrays of such and such shape. The way to do this is through
the simple XML descriptor `XDMF <www.xdmf.org>`_. To this end there is a
function imported from `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_
called ``generate_xdmf`` that can be called with any one of the
generated hdf5-files::

    generate_xdmf('myh5file.h5')
    generate_xdmf('mixed.h5')

This results in some light xdmf-files being generated for the 2D and 3D arrays in
the hdf5-file:

    * ``myh5file.xdmf``
    * ``myh5file_0_slice_slice.xdmf``
    * ``mixed.xdmf``
    * ``mixed_4_slice_slice.xdmf``
These xdmf-files can be opened and inspected by ParaView. Note that 1D arrays are
not wrapped, and neither are 4D.

An annoying feature of Paraview is that it views a three-dimensional array of
shape :math:`(N_0, N_1, N_2)` as transposed compared to shenfun. That is,
for Paraview the *last* axis represents the :math:`x`-axis, whereas
shenfun (like most others) considers the first axis to be the :math:`x`-axis. So when opening a
three-dimensional array in Paraview one needs to be aware. Especially when
plotting vectors. Assume that we are working with a Navier-Stokes solver
and have a three-dimensional :class:`VectorTensorProductSpace` to represent
the fluid velocity::

    from mpi4py import MPI
    from shenfun import *

    comm = MPI.COMM_WORLD
    N = (32, 64, 128)
    V0 = Basis(N[0], 'F', dtype='D')
    V1 = Basis(N[1], 'F', dtype='D')
    V2 = Basis(N[2], 'F', dtype='d')
    T = TensorProductSpace(comm, (V0, V1, V2))
    TV = VectorTensorProductSpace(T)
    U = Array(TV)
    U[0] = 0
    U[1] = 1
    U[2] = 2

To store the resulting :class:`.Array` ``U`` we can create an instance of the
:class:`.HDF5File` class, and store using keyword ``as_scalar=True``::

    hdf5file = ShenfunFile("NS", TV, backend='hdf5', mode='w')
    ...
    file.write(0, {'u': [U]}, as_scalar=True)
    file.write(1, {'u': [U]}, as_scalar=True)

Generate an xdmf file through::

    generate_xdmf('NS.h5')

and open the generated ``NS.xdmf`` file in Paraview. You will then see three scalar
arrays ``u0, u1, u2``, each one of shape ``(32, 64, 128)``, for the vector
component in what Paraview considers the :math:`z`, :math:`y` and :math:`x` directions,
respectively. Other than the swapped coordinate axes there is no difference.
But be careful if creating vectors in Paraview with the Calculator. The vector
should be created as::

    u0*kHat+u1*jHat+u2*iHat
