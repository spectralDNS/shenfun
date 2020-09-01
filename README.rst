Shenfun
=======
.. image:: https://api.codacy.com/project/badge/Grade/dc9c6e8e33c34382b76d38916852b36b
    :target: https://app.codacy.com/app/mikaem/shenfunutm_source=github.com&utm_medium=referral&utm_content=spectralDNS/shenfun&utm_campaign=badger
.. image:: https://dev.azure.com/spectralDNS/shenfun/_apis/build/status/spectralDNS.shenfun?branchName=master
    :target: https://dev.azure.com/spectralDNS/shenfun
.. image:: https://github.com/spectralDNS/shenfun/workflows/github-CI/badge.svg?branch=master
    :target: https://github.com/spectralDNS/shenfun
.. image:: https://codecov.io/gh/spectralDNS/shenfun/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spectralDNS/shenfun
.. image:: https://anaconda.org/conda-forge/shenfun/badges/platforms.svg
    :target: https://anaconda.org/conda-forge/shenfun
.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/spectralDNS/shenfun/master?filepath=binder


Try it in a jupyter hub using Binder

|binder|

Description
-----------
Shenfun is a high performance computing platform for solving partial differential equations (PDEs) by the spectral Galerkin method. The user interface to shenfun is very similar to `FEniCS <https://fenicsproject.org>`_, but applications are limited to multidimensional tensor product grids, using either Cartesian or curvilinear grids (e.g., but not limited to, polar, cylindrical, spherical or parabolic). The code is parallelized with MPI through the `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_ package.

Shenfun enables fast development of efficient and accurate PDE solvers (spectral order and accuracy), in the comfortable high-level Python language. The spectral accuracy is ensured by using high-order *global* orthogonal basis functions (Fourier, Legendre, Chebyshev, Laguerre, Hermite and Jacobi), as opposed to finite element codes that are using low-order *local* basis functions. Efficiency is ensured through vectorization (`Numpy <https://www.numpy.org/>`_), parallelization (`mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_) and by moving critical routines to `Cython <https://cython.org/>`_ or `Numba <https://numba.pydata.org>`_. Shenfun has been used to run turbulence simulations (Direct Numerical Simulations) on thousands of processors on high-performance supercomputers, see the `spectralDNS <https://github.com/spectralDNS/spectralDNS>`_ repository.

The demo folder contains several examples for the Poisson, Helmholtz and Biharmonic equations. For extended documentation and installation instructions see `ReadTheDocs <http://shenfun.readthedocs.org>`_. For interactive demos, see the `jupyter book <https://mikaem.github.io/shenfun-demos>`_. Note that shenfun currently comes with the possibility to use two non-periodic directions (see `biharmonic demo <https://github.com/spectralDNS/shenfun/blob/master/demo/biharmonic2D_2nonperiodic.py>`_), and equations may be solved coupled and implicit (see `MixedPoisson.py <https://github.com/spectralDNS/shenfun/blob/master/demo/MixedPoisson.py>`_). 

Note that there is now also support for solving equations using curvilinear coordinates. For example, it is possible to solve equations on a `sphere <https://github.com/spectralDNS/shenfun/blob/master/demo/sphere_helmholtz.py>`_ (using spherical coordinates), on the surface of a `torus <https://github.com/spectralDNS/shenfun/blob/master/binder/Torus.ipynb>`_, or along any `curved line in 2D/3D <https://github.com/spectralDNS/shenfun/blob/master/demo/curvilinear_poisson1D.py>`_, like for the coil show below. Actually, any new coordinates may be defined by the user as long as the coordinates lead to separable systems of equations. After defining new coordinates, operators like div, grad and curl work automatically with the new curvilinear coordinates. See also `this notebook on the sphere <https://github.com/spectralDNS/shenfun/blob/master/binder/sphere2-helmholtz.ipynb>`_ or an illustration of the `vector Laplacian <https://github.com/spectralDNS/shenfun/blob/master/binder/vector-laplacian.ipynb>`_.

.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/smallcoil2.png
    :alt: Solution of Poisson's equation on a Coil
.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/spherewhite4.png
    :alt: Solution of Poisson's equation on a spherical shell
.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/torus2.png
    :alt: Solution of Poisson's equation on the surface of a torus

For a more psychedelic experience, have a look at the `simulation <https://github.com/spectralDNS/shenfun/blob/master/demo/Ginzburg_Landau_sphere_IRK3.py>`_ of the Ginzburg-Landau equation on the sphere (click for Youtube-video):

.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/GLimage.png
    :target: https://youtu.be/odsIoHVcqek
    :alt: Ginzburg-Landau spherical coordinates

Shenfun can also be used to approximate analytical functions with global spectral basis `functions <https://mikaem.github.io/shenfun-demos/content/functions.html>`_, and to integrate over highly complex domains, like the seashell below, see `this demo <https://mikaem.github.io/shenfun-demos/content/surfaceintegration.html>`_.

.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/seashell3.png
    :alt: The surface of a seashell

Installation
------------

Shenfun can be installed using either `pip <https://pypi.org/project/pip/>`_ or `conda <https://conda.io/docs/>`_, see `installation chapter on readthedocs <https://shenfun.readthedocs.io/en/latest/installation.html>`_.

Dependencies
------------

    * `Python <https://www.python.org/>`_ 3.3 or above. Test suits are run with Python 3.6, 3.7 and 3.8.
    * A functional MPI 2.x/3.x implementation like `MPICH <https://www.mpich.org>`_ or `Open MPI <https://www.open-mpi.org>`_ built with shared/dynamic libraries.
    * `FFTW <http://www.fftw.org/>`_ version 3, also built with shared/dynamic libraries.
    * Python modules:
        * `Numpy <https://www.numpy.org/>`_
        * `Scipy <https://www.scipy.org/>`_
        * `Sympy <https://www.sympy.org>`_
        * `Cython <https://cython.org/>`_
        * `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_
        * `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_

Contact
-------
For comments, issues, bug-reports and requests, please use the issue tracker of the current repository, or see `How to contribute? <https://shenfun.readthedocs.io/en/latest/howtocontribute.html>`_ at readthedocs. Otherwise the principal author can be reached at::

    Mikael Mortensen
    mikaem at math.uio.no
    http://folk.uio.no/mikaem/
    Department of Mathematics
    University of Oslo
    Norway
