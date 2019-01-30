Installation
============

Shenfun has a few dependencies

    * `mpi4py`_
    * `FFTW`_
    * `mpi4py-fft`_
    * `cython`_
    * `numpy`_
    * `sympy`_
    * `scipy`_

that are mostly straight-forward to install, or already installed in
most Python environments. The first two are usually most troublesome.
Basically, for `mpi4py`_ you need to have a working MPI installation,
whereas `FFTW`_ is available on most high performance computer systems.
If you are using `conda`_, then both these libraries can be installed
directly from the `conda-forge`_ channel. In an appropriate conda
environment you can then

::

    conda install -c conda-forge mpi4py mpich fftw numpy cython

This installs `mpich`_ as a dependency of `mpi4py`_. You can switch
``mpich`` with ``openmpi`` in the line above to get `openmpi`_
instead. But then note that `h5py`_ needs to be compiled with `openmpi`_
as well. If you do not use `conda`_,
then you need to make sure that MPI and FFTW are installed by some
other means.

If not already present, the remaining dependencies can be easily
installed using `pip`_ or `conda`_. However, if mixing `pip`_ and
`conda`_, make sure that `pip`_ is installed into the working conda
environment first

::

    conda install pip

To install remaining dependencies as well as ``shenfun`` from `pypi`_

::

    pip install shenfun

whereas the following will install the latest version from github

::

    pip install git+https://github.com/spectralDNS/shenfun.git@master

You can also build ``shenfun`` from the top directory, after cloning
or forking

::

    pip install .

or using `conda-build`_ with the recipes in folder ``conf/conda``

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
or 3.6. To specify the Python version as 3.6 instead of default
you can for exampel do

::

    conda create --name shenfun_py3 -c conda-forge -c spectralDNS python=3.6 shenfun
    source activate shenfun_py3

Additional dependencies
-----------------------

For storing and retrieving data you need either `HDF5`_ or `netCDF4`_, compiled
with support for MPI (see :ref:`Postprocessing`). `HDF5`_ is already available
with parallel support on `conda-forge`_ and can be installed (with the mpich
backend for MPI) as

::

    conda install -c conda-forge h5py=*=mpi_mpich_*

A parallel version of `netCDF4`_ cannot be found on conda-forge, but a precompiled
version has been made available on the `spectralDNS channel`_

::

    conda install -c spectralDNS netcdf4-parallel

Note that parallel HDF5 and h5py often are available as modules on
supercomputers. Otherwise, see the respective packages for how to install
with support for MPI.

Test installation
-----------------

After installing (from source) it may be a good idea to run all the tests
located in the `tests <https://github.com/spectralDNS/shenfun/tree/master/tests>`_
folder. The tests are run with `pytest <https://docs.pytest.org/en/latest/>`_
from the main directory of the source code

::

    python -m pytest tests/

However, note that you may need to install pytest into the correct
environment as well. A common mistake is to run a version of pytest that has
already been installed in a different conda environment, perhaps using a different Python
version.

The tests are run automatically on every commit to github, see

.. image:: https://travis-ci.org/spectralDNS/shenfun.svg?branch=master
    :target: https://travis-ci.org/spectralDNS/shenfun
.. image:: https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg
    :target: https://circleci.com/gh/spectralDNS/shenfun


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
.. _numpy: https://www.numpy.org
.. _sympy: https://www.sympy.org
.. _scipy: https://www.scipy.org
.. _conda-build: https://conda.io/docs/commands/build/conda-build.html
.. _pypi: https://pypi.org/project/shenfun/