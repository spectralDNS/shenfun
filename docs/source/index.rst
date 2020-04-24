.. shenfun documentation master file, created by
   sphinx-quickstart on Tue Mar 27 18:30:26 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to shenfun's documentation!
===================================

Shenfun is a high performance computing platform for solving partial
differential equations (PDEs) by the spectral Galerkin method.
The user interface to shenfun is very similar to `FEniCS`_,
but applications are limited to multidimensional tensor product grids,
using either Cartesian or curvilinear grids (polar/cylindrical/spherical).
The code is parallelized with MPI through the `mpi4py-fft`_ package.

Shenfun enables fast development of efficient and accurate PDE solvers
(spectral order and accuracy), in the comfortable high-level Python
language. The spectral accuracy is ensured by using high-order
*global* orthogonal basis functions (Fourier, Legendre, Chebyshev, Laguerre and Hermite),
as opposed to finite element codes that are using low-order
*local* basis functions. Efficiency is ensured through vectorization
(`Numpy`_), parallelization (`mpi4py`_) and by moving critical routines to
`Cython`_ or `Numba`_. Shenfun has been used to run turbulence simulations (Direct
Numerical Simulations) on thousands of processors on high-performance
supercomputers, see the `spectralDNS`_ repository.

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
   howtocite
   howtocontribute

.. toctree::
   :caption: Demos:
   :maxdepth: 1

   poisson
   kleingordon
   poisson3d
   polarhelmholtz
   sphericalhelmholtz
   kuramatosivashinsky
   stokes
   drivencavity
   rayleighbenard
   zrefs

.. toctree::
   :caption: Indices and tables

   indices

.. _shenfun: https:/github.com/spectralDNS/shenfun
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _Fenics: https://fenicsproject.org
.. _Numpy: https:/www.numpy.org
.. _Numba: https://numba.pydata.org
.. _Cython: https://cython.org
.. _spectralDNS: https://github.com/spectralDNS/spectralDNS
