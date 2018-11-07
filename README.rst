Shenfun
=======
.. image:: https://api.codacy.com/project/badge/Grade/dc9c6e8e33c34382b76d38916852b36b
    :target: https://app.codacy.com/app/mikaem/shenfunutm_source=github.com&utm_medium=referral&utm_content=spectralDNS/shenfun&utm_campaign=badger
.. image:: https://travis-ci.org/spectralDNS/shenfun.svg?branch=master
    :target: https://travis-ci.org/spectralDNS/shenfun
.. image:: https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg
    :target: https://circleci.com/gh/spectralDNS/shenfun
.. image:: https://codecov.io/gh/spectralDNS/shenfun/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spectralDNS/shenfun

Description
-----------
Shenfun is a high performance computing platform for solving partial differential equations by the spectral Galerkin method. The user interface to shenfun is very similar to `FEniCS <https://fenicsproject.org>`_, but applications are limited to multidimensional tensor product grids. The code is parallelized with MPI through the `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_ package.

The demo folder contains several examples for the Poisson, Helmholtz and Biharmonic equations. For extended documentation and installation instructions see `ReadTheDocs <http://shenfun.readthedocs.org>`_. See also this `paper <https://rawgit.com/spectralDNS/shenfun/master/docs/demos/mekit17/pub/shenfun_bootstrap.html>`_.

Installation
------------

Shenfun can be installed using either `pip <https://pypi.org/project/pip/>`_ or `conda <https://conda.io/docs/>`_, see `installation chapter on readthedocs <https://shenfun.readthedocs.io/en/latest/installation.html>`_.

Contact
-------
For comments, issues, bug-reports and requests, please use the issue tracker of the current repository. Otherwise the principal author can be reached at::

    Mikael Mortensen
    mikaem at math.uio.no
    http://folk.uio.no/mikaem/
    Department of Mathematics
    University of Oslo
    Norway
