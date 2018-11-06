Installation
============

Shenfun has a few dependencies

::

    mpi4py
    FFTW
    mpi4py-fft
    cython
    numpy
    sympy
    scipy

that are mostly straight-forward to install, or already installed in
most Python environments. The first two are usually most troublesome.
Basically, for `mpi4py`_ you need to have a working MPI installation,
whereas `FFTW`_ is avalable on most high performance computer systems.
If you are using `conda`_, then both these libraries can be installed
directly from the `conda-forge`_ channel

::

    conda install -c conda-forge mpi4py mpich fftw

which installs `mpich`_ as a dependency of `mpi4py`_. You can get `openmpi`_
instead by specifying so in the line above. If you do not use `conda`_,
then you need to make sure that MPI and FFTW are installed by some
other means.

If not already present, the remaining dependencies can be easily
installed using `pip`_ or `conda`_. To install remaining dependencies as
well as ``shenfun`` do

::

    pip install shenfun

whereas the following will install the latest version from github

::

    pip install git+https://github.com/spectralDNS/shenfun.git@master

From the top directory you can also do

::

    pip install .

You can also build yourself using conda-build (from top directory after
cloning or forking)

::

    conda build -c conda-forge -c spectralDNS conf/conda
    conda create --name shenfun -c conda-forge -c spectralDNS shenfun --use-local
    source activate shenfun

You may also use precompiled binaries in the `spectralDNS channel`_. Use for exampel

::

    conda create --name shenfun -c conda-forge -c spectralDNS shenfun
    source activate shenfun

which installs both shenfun, and all required dependencies,
most of which are pulled in from the conda-forge channel. There are
binaries compiled for both OSX and linux, for either Python version 2.7
or 3.6. To specify the Python version as 3.6 instead of default (used
above) you can for exampel do

::

    conda create --name shenfun_py3 -c conda-forge -c spectralDNS python=3.6 shenfun
    source activate shenfun_py3

Extra dependencies
------------------

For storing and retrieving data you need either `HDF5`_ or `netCDF4`_, compiled
with support for MPI (see :ref:`Postprocessing`). Unfortunately, the libraries that
are available on `conda-forge`_ are compiled without MPI and cannot be used.
But parallel versions have been made available on the `spectralDNS channel`_

::

    conda install -c spectralDNS h5py-parallel netcdf4-parallel

which installs the two required Python modules `h5py`_ and `netCDF4`_. Otherwise,
see the respective packages for how to install with support for MPI.


.. _github: https://github.com/spectralDNS/shenfun
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _cython: http://cython.org
.. _spectralDNS channel: https://anaconda.org/spectralDNS
.. _conda: https://conda.io/docs/
.. _conda-forge: https://conda-forge.org
.. _FFTW: http://www.fftw.org
.. _pip: https://pypi.org/project/pip/
.. _HDF5: https://www.hdfgroup.org
.. _netCDF4: http://unidata.github.io/netcdf4-python/
.. _h5py: https://www.h5py.org
.. _mpich: https://www.mpich.org
.. _openmpi: https://www.open-mpi.org
