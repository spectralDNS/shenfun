.. shenfun documentation master file, created by
   sphinx-quickstart on Tue Mar 27 18:30:26 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to shenfun's documentation!
===================================

`Shenfun`_ is a high performance computing platform for solving partial
differential equations by the spectral Galerkin method.
The user interface to shenfun is very similar to `FEniCS`_,
but applications are limited to multidimensional tensor product grids.
The code is parallelized with MPI through the `mpi4py-fft`_
package.

Document build status
---------------------

.. image:: https://readthedocs.org/projects/shenfun/badge/?version=latest
   :target: http://shenfun.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   gettingstarted
   installation

.. toctree::
   :caption: Demos:
   :maxdepth: 1

   poisson
   kleingordon
   poisson3d
   kuramatosivashinsky
   zrefs

.. toctree::
   :caption: Indices and tables

   indices

.. _shenfun: https:/github.com/spectralDNS/shenfun
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _Fenics: https://fenicsproject.org
