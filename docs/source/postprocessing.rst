.. _Postprocessing:

Post processing
===============

MPI is great because it means that you can run Shenfun on pretty much
as many CPUs as you can get your hands on. However, MPI makes it more
challenging to do visualization, in particular with Python and Matplotlib. For
this reason there is a :mod:`.utilities` module with helper classes for dumping dataarrays
to `HDF5 <https://www.hdf5.org>`_ or `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_

The helper classes are

    * :class:`.HDF5Writer`
    * :class:`.NCWriter`

An instance of either class is created with a name of the file that is going
to store the data, names of the arrays to store, and an instance of a
:class:`.TensorProductSpace`. For example, to create an HDF5 writer for a 3D
TensorProductSpace with Fourier bases in all directions::

    from shenfun import *
    from mpi4py import MPI
    N = (24, 25, 26)
    K0 = Basis(N[0], 'F', dtype='D')
    K1 = Basis(N[1], 'F', dtype='D')
    K2 = Basis(N[2], 'F', dtype='d')
    T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2))
    h5file = HDF5Writer('myh5file.h5', ['u'], T)

The instance `h5file` will now have two methods that can be used to dump
dataarrays, either the complete array, or slices into the domain

    * :meth:`.HDF5Writer.write_tstep`
    * :meth:`.HDF5Writer.write_slice_tstep`

These methods assume, as their names suggest, that the problem to be solved is
timedependent. As such it is easy to use for a transient problem. 

The :class:`.HDFWriter` class may also be used for the :class:`.MixedTensorProductSpace`,
or :class:`.VectorTensorProductSpace`, that are collections of the scalar
:class:`.TensorProductSpace`. We can create a :class:`.MixedTensorProductSpace`
consisting of two TensorProductSpaces, and an accompanying writer class as::

    TT = MixedTensorProductSpace([T, T])
    h5file_m = HDF5Writer('mixed.h5', ['u', 'f'], TT)

Let's now consider a transient problem where we step a solution forward in time. 
We create solution arrays from :class:`.Function` s, and update these Functions
inside a while loop::

    u = Function(T)
    uf = Function(TT)
    tstep = 0
    while tstep < 3:
        ... solve for u and uf
        h5file.write_tstep(tstep, u)
        h5file.write_slice_tstep(tstep, [0, slice(None), slice(None)], u)
        h5file.write_slice_tstep(tstep, [0, 20, slice(None)], u)
        
        h5file_m.write_step(tstep, uf)
        h5file_m.write_slice_tstep(tstep, [4, slice(None), slice(None)], uf)
        h5file_m.write_slice_tstep(tstep, [slice(None), 10, 10], uf)
        tstep += 1

During the 3 time steps we will with `h5file` dump 3 dense 3D arrays, 3
2D arrays (``u[0, :, :]``) and 3 1D arrays (``u[0, 20, :]``)
to the file `myh5file.h5`. The different arrays will be found in groups
stored in `myh5file.h5` with directory tree structure as::

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
    ├─ f/
    |  ├─ 1D/
    |  |  └─ slice_10_10/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 3
    |  ├─ 2D/
    |  |  └─ 4_slice_slice/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 2
    |  └─ 3D/
    |     ├─ 0
    |     ├─ 1
    |     └─ 2
    ├─ u/
    |  ├─ 1D/
    |  |  └─ slice_10_10/
    |  |     ├─ 0
    |  |     ├─ 1
    |  |     └─ 3
    |  ├─ 2D/
    |  |  └─ 4_slice_slice/
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

Note that the mesh is stored as well as the results. The three mesh arrays are
all 1D arrays, representing the domain for each basis in the TensorProductSpace.
Also note that these routines work with any number of CPUs and dimensions.

ParaView
--------

The stored datafiles can be visualized in `ParaView <www.paraview.org>`_. 
However, ParaView cannot understand the content of these HDF5-files without
a little bit of help. We have to explain that these data-files contain
structured arrays of such and such shape. The way to do this is through 
the simple XML descriptor `XDMF <www.xdmf.org>`_. To this end there is a
function called :func:`.generate_xdmf` that can be called with any of the
generated hdf5-files::

    generate_xdmf('myh5file.h5')
    generate_xdmf('mixed.h5')

This results in some light files being generated for the 2D and 3D arrays in
the hdf5-file: ``myh5file.xdmf, myh5file_0_slice_slice.xdmf,
mixed.xdmf, mixed_4_slice_slice.xdmf``. These ``xdmf``-files can be opened 
and inspected by ParaView. Note that 1D arrays are not wrapped, and neither are
4D.


