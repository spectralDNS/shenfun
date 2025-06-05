Shenfun
=======
.. image:: https://app.codacy.com/project/badge/Grade/bd772b3ca7134651a9225d8051db8c41
    :target: https://www.codacy.com/gh/spectralDNS/shenfun/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=spectralDNS/shenfun&amp;utm_campaign=Badge_Grade
.. image:: https://dev.azure.com/spectralDNS/shenfun/_apis/build/status/spectralDNS.shenfun?branchName=master
    :target: https://dev.azure.com/spectralDNS/shenfun
.. image:: https://github.com/spectralDNS/shenfun/actions/workflows/main.yml/badge.svg?branch=master
    :target: https://github.com/spectralDNS/shenfun
.. image:: https://codecov.io/gh/spectralDNS/shenfun/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/spectralDNS/shenfun
.. image:: https://anaconda.org/conda-forge/shenfun/badges/platforms.svg
    :target: https://anaconda.org/conda-forge/shenfun

Description
-----------
Shenfun is a high performance computing platform for solving partial differential equations (PDEs) by the spectral Galerkin method. The user interface to shenfun is very similar to `FEniCS <https://fenicsproject.org>`_, but applications are limited to multidimensional tensor product grids, using either Cartesian or curvilinear grids (e.g., but not limited to, polar, cylindrical, spherical or parabolic). The code is parallelized with MPI through the `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_ package.

Shenfun enables fast development of efficient and accurate PDE solvers (spectral order and accuracy), in the comfortable high-level Python language. The spectral accuracy is ensured by using high-order *global* orthogonal basis functions (Fourier, Legendre, Chebyshev first and second kind, Ultraspherical, Jacobi, Laguerre and Hermite), as opposed to finite element codes that are using low-order *local* basis functions. Efficiency is ensured through vectorization (`Numpy <https://www.numpy.org/>`_), parallelization (`mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_) and by moving critical routines to `Cython <https://cython.org/>`_ or `Numba <https://numba.pydata.org>`_. Shenfun has been used to run turbulence simulations (Direct Numerical Simulations) on thousands of processors on high-performance supercomputers, see the `spectralDNS <https://github.com/spectralDNS/spectralDNS>`_ repository.

The demo folder contains several examples for the Poisson, Helmholtz and Biharmonic equations. For extended documentation and installation instructions see `ReadTheDocs <http://shenfun.readthedocs.org>`_. For interactive demos, see the `jupyter book <https://mikaem.github.io/shenfun-demos>`_. Note that shenfun currently comes with the possibility to use two non-periodic directions (see `biharmonic demo <https://github.com/spectralDNS/shenfun/blob/master/demo/biharmonic2D_2nonperiodic.py>`_), and equations may be solved coupled and implicit (see `MixedPoisson.py <https://github.com/spectralDNS/shenfun/blob/master/demo/MixedPoisson.py>`_).

Note that shenfun works with curvilinear coordinates. For example, it is possible to solve equations on a `sphere <https://github.com/spectralDNS/shenfun/blob/master/demo/sphere_helmholtz.py>`_ (using spherical coordinates), on the surface of a `torus <https://github.com/spectralDNS/shenfun/blob/master/docs/notebooks/Torus.ipynb>`_, on a `Möbius strip <https://mikaem.github.io/shenfun-demos/content/moebius.html>`_ or along any `curved line in 2D/3D <https://github.com/spectralDNS/shenfun/blob/master/demo/curvilinear_poisson1D.py>`_. Actually, any new coordinates may be defined by the user as long as the coordinates lead to a system of equations with separable coefficients. After defining new coordinates, operators like div, grad and curl work automatically with the new curvilinear coordinates. See also `this notebook on the sphere <https://github.com/spectralDNS/shenfun/blob/master/docs/notebooks/sphere-helmholtz.ipynb>`_ or an illustration of the `vector Laplacian <https://github.com/spectralDNS/shenfun/blob/master/docs/notebooks/vector-laplacian.ipynb>`_.

.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/moebius8_trans.png
    :target: https://mikaem.github.io/shenfun-demos/content/moebius.html
    :alt: The eigenvector of the 8'th smallest eigvalue on a Möbius strip
.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/smallcoil2.png
    :alt: Solution of Poisson's equation on a Coil
.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/spherewhite4.png
    :target: https://mikaem.github.io/shenfun-demos/content/sphericalhelmholtz.html
    :alt: Solution of Poisson's equation on a spherical shell
.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/torus2.png
    :target: https://github.com/spectralDNS/shenfun/blob/master/docs/notebooks/Torus.ipynb
    :alt: Solution of Poisson's equation on the surface of a torus


For a more psychedelic experience, have a look at the `simulation <https://github.com/spectralDNS/shenfun/blob/master/demo/Ginzburg_Landau_sphere_IRK3.py>`_ of the Ginzburg-Landau equation on the sphere (click for Youtube-video):

.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/GLimage.png
    :target: https://youtu.be/odsIoHVcqek
    :alt: Ginzburg-Landau spherical coordinates

Shenfun can also be used to approximate analytical functions with global spectral basis `functions <https://mikaem.github.io/shenfun-demos/content/functions.html>`_, and to integrate over highly complex domains, like the seashell below, see `this demo <https://mikaem.github.io/shenfun-demos/content/surfaceintegration.html>`_.

.. image:: https://cdn.jsdelivr.net/gh/spectralDNS/spectralutilities@master/figures/seashell3.png
    :alt: The surface of a seashell

Some recent papers using Shenfun
--------------------------------

- `Posterior comparison of model dynamics in several hybrid turbulence model forms <https://pubs.aip.org/aip/pof/article/36/10/105148/3315660/Posterior-comparison-of-model-dynamics-in-several>`_ Towery, Colin A. Z. and Sáenz, Juan A. and Livescu, Daniel, Physics of Fluids 36, 105148 (2024)
- `Effective control of two-dimensional Rayleigh–Bénard convection: Invariant multi-agent reinforcement learning is all you need <https://pubs.aip.org/aip/pof/article/35/6/065146/2900730>`_ C. Vignon, J. Rabault, J. Vasanth, F. Alcántara-Ávila, M. Mortensen, R. Vinuesa, Physics of Fluids 35, 065146 (2023)
- `Solving Partial Differential Equations with Equivariant Extreme Learning Machines <https://www.researchgate.net/profile/Sebastian-Peitz/publication/380897446_Solving_Partial_Differential_Equations_with_Equivariant_Extreme_Learning_Machines/links/66544d0fbc86444c7205cbdb/Solving-Partial-Differential-Equations-with-Equivariant-Extreme-Learning-Machines.pdf>`_, H. Harder, J. Rabault, R. Vinuesa, M. Mortensen, S. Peitz. preprint (2024)
- `A global spectral-Galerkin investigation of a Rayleigh–Taylor instability in plasma using an MHD–Boussinesq model <https://pubs.aip.org/aip/adv/article/13/10/105319/2917415>`_  A. Piterskaya, Wojciech J. Miloch, M. Mortensen, AIP Advances 13, 105319 (2023)
- `A Generic and Strictly Banded Spectral Petrov–Galerkin Method for Differential Equations with Polynomial Coefficients <https://epubs.siam.org/doi/full/10.1137/22M1492842>`_ M. Mortensen, SIAM J. on Scientific Computing, 45, 1, A123-A146, (2023)
- `Variance representations and convergence rates for data-driven approximations of Koopman operators <https://arxiv.org/abs/2402.02494>`_ F. M. Philipp, M. Schaller, S. Boshoff, S. Peitz, F. Nüske, K. Worthmann, preprint (2024)
- `Partial observations, coarse graining and equivariance in Koopman operator theory for large-scale dynamical systems <https://arxiv.org/abs/2307.15325>`_, S. Peitz, H. Harder, F. Nüske, F. Philipp, M. Schaller, K. Worthmann, preprint (2024)
- `Koopman-Based Surrogate Modelling of Turbulent Rayleigh-Bénard Convection <https://arxiv.org/abs/2405.06425>`_ T. Markmann, M. Straat, B. Hammer, preprint (2024)
- `Shenfun: High performance spectral Galerkin computing platform <https://joss.theoj.org/papers/10.21105/joss.01071.pdf>`_, M. Mortensen, Journal of Open Source Software, 3(31), 1071 (2018)


Installation
------------

Shenfun can be installed using either `pip <https://pypi.org/project/pip/>`_ or `conda <https://conda.io/docs/>`_, see `installation chapter on readthedocs <https://shenfun.readthedocs.io/en/latest/installation.html>`_.

Dependencies
------------

    * `Python <https://www.python.org/>`_ 3.7 or above. Test suits are run with Python 3.10, 3.11 and 3.12.
    * A functional MPI 2.x/3.x implementation like `MPICH <https://www.mpich.org>`_ or `Open MPI <https://www.open-mpi.org>`_ built with shared/dynamic libraries.
    * `FFTW <http://www.fftw.org/>`_ version 3, also built with shared/dynamic libraries.
    * Python modules:
        * `Numpy <https://www.numpy.org/>`_
        * `Scipy <https://www.scipy.org/>`_
        * `Sympy <https://www.sympy.org>`_
        * `Cython <https://cython.org/>`_
        * `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_
        * `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_


Try it in a codespace
---------------------
The easiest way to try out shenfun is to create your own codespace. Press the green codespace button on this page and wait for a couple of minutes while everything in `environment.yml` gets installed. Then write in the terminal of the codespace editor:: 

     source activate shenfun
     echo -e "PYTHONPATH=/workspaces/shenfun" > .env
     export PYTHONPATH=/workspaces/shenfun

and you are set to run any of the demo programs, or for example try to follow the detailed instructions in the `documentation <https://shenfun.readthedocs.io/en/latest/gettingstarted.html>`_. We assume that you know how to run a Python program. Please note that if you want to use for example IPython or Jupyter in the codespace, then these need to be installed into the shenfun environment.

Contact
-------
For comments, issues, bug-reports and requests, please use the issue tracker of the current repository, or see `How to contribute? <https://shenfun.readthedocs.io/en/latest/howtocontribute.html>`_ at readthedocs. Otherwise the principal author can be reached at::

    Mikael Mortensen
    mikaem at math.uio.no
    https://mikaem.github.io/
    Department of Mathematics
    University of Oslo
    Norway
