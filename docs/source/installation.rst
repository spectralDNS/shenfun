Installation
============

Shenfun has quite a few dependencies and as such it is not completely
straight-forward to install. For example, it depends on the non-standard
`mpiFFT4py`_  and `mpi4py-fft`_, and it requires `cython`_ to optimize a
few routines. However, most problematic is `pyFFTW`_ for the serial fast 
Fourier transforms. Here it is not possible to use a standard installation,
because *pyFFTW* is very slow at incorporating new pull requests. Currently
we need to use the fork by `David Wells`_ to get to the fast discrete 
cosine transforms.

If all dependencies are in place, then shenfun can be installed by cloning 
or forking the repository at `github`_ and then with regular python distutils

::

    python setup.py install --prefix="path used for installation. Must be on the PYTHONPATH"

or in-place using

::

    python setup.py build_ext --inplace

However, due to the non-standard dependencies, a much easier installation is 
achieved by going through `conda`_ and the `spectralDNS channel`_ on Anaconda
Cloud. When you build with conda, all the correct dependencies will automatically
be pulled in. From the top directory, after cloning, build shenfun yourself with

::

    conda build -c conda-forge -c spectralDNS conf/conda
    conda create --name shenfun -c conda-forge -c spectralDNS shenfun --use-local
    source activate shenfun

You may also use precompiled binaries in the `spectralDNS channel`_. Use for exampel

::

    conda create --name shenfun -c conda-forge -c spectralDNS shenfun
    source activate shenfun

which installs both shenfun, mpiFFT4py and all required dependencies,
most of which are pulled in from the conda-forge channel. There are
binaries compiled for both OSX and linux, for either Python version 2.7
or 3.6. To specify the Python version as 3.6 instead of default (used
above) you can for exampel do

::

    conda create --name shenfun_py3 -c conda-forge -c spectralDNS python=3.6 shenfun
    source activate shenfun_py3

.. _github: https://github.com/spectralDNS/shenfun
.. _mpiFFT4py: https://github.com/spectralDNS/mpiFFT4py
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _cython: http://cython.org
.. _pyFFTW: https://github.com/pyFFTW/pyFFTW
.. _David Wells: https://github.com/drwells/pyFFTW/tree/r2r-try-two
.. _spectralDNS channel: https://anaconda.org/spectralDNS
.. _conda: https://conda.io/docs/

