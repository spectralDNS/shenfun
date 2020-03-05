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
    * `h5py`_

that are mostly straight-forward to install, or already installed in
most Python environments. The first two are usually most troublesome.
Basically, for `mpi4py`_ you need to have a working MPI installation,
whereas `FFTW`_ is available on most high performance computer systems.
If you are using `conda`_, then all you need to install a fully functional
shenfun, with all the above dependencies, is

::

    conda install -c conda-forge shenfun

You probably want to install into a fresh environment, though, which
can be achieved with

::

    conda create --name shenfun -c conda-forge shenfun
    conda activate shenfun

Note that this gives you shenfun with default settings. This means that
you will probably get the openmpi backend. To make sure that shenfun is
is installed with mpich instead do

::

    conda create --name shenfun -c conda-forge shenfun mpich

If you do not use `conda`_, then you need to make sure that MPI
and FFTW are installed by some other means. You can then install
any version of shenfun hosted on `pypi`_ using `pip`_

::

    pip install shenfun

whereas the following will install the latest version from github

::

    pip install git+https://github.com/spectralDNS/shenfun.git@master

You can also build ``shenfun`` yourselves from the top directory,
after cloning or forking

::

    pip install .

or using `conda-build`_ with the recipes in folder ``conf/conda``

::

    conda build -c conda-forge -c spectralDNS conf/conda
    conda create --name shenfun -c conda-forge -c spectralDNS shenfun --use-local
    conda activate shenfun

Optimization
------------

Shenfun contains a few routines (essentially linear algebra solvers
and matrix vector products) that are difficult to vectorize with numpy,
and for this reason they have been implemented in either (or both of)
`Numba`_ or `Cython`_. The user may choose which implementation
to use through the environment variable ``SHENFUN_OPTIMIZATION``. The default
is to use cython, but it is possible to enable either one by making
the appropriate choice in the active terminal

::

    export SHENFUN_OPTIMIZATION={CYTHON,NUMBA}

Additional dependencies
-----------------------

For storing and retrieving data you need either `HDF5`_ or `netCDF4`_, compiled
with support for MPI (see :ref:`Postprocessing`). Both `HDF5`_  and `netCDF4`_ 
are already available with parallel support on `conda-forge`_, and, if they were 
not installed at the same time as shenfun, they can be installed as

::

    conda install -c conda-forge h5py=*=mpi* netcdf4=*=mpi*

Note that parallel HDF5 and NetCDF4 often are available as modules on
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

However, note that for conda you need to install pytest into the correct
environment as well. A common mistake is to run a version of pytest that has
already been installed in a different conda environment, perhaps using a
different Python version.

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
.. _numba: https://www.numba.org
.. _sympy: https://www.sympy.org
.. _scipy: https://www.scipy.org
.. _conda-build: https://conda.io/docs/commands/build/conda-build.html
.. _pypi: https://pypi.org/project/shenfun/