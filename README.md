# shenfun

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/dc9c6e8e33c34382b76d38916852b36b)](https://app.codacy.com/app/mikaem/shenfun?utm_source=github.com&utm_medium=referral&utm_content=spectralDNS/shenfun&utm_campaign=badger)
[![Build Status](https://travis-ci.org/spectralDNS/shenfun.svg?branch=master)](https://travis-ci.org/spectralDNS/shenfun)
[![CircleCI](https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg)](https://circleci.com/gh/spectralDNS/shenfun)
[![codecov](https://codecov.io/gh/spectralDNS/shenfun/branch/master/graph/badge.svg)](https://codecov.io/gh/spectralDNS/shenfun)

Description
-----------
Shenfun is a toolbox for automating the spectral Galerkin method.  The user interface to shenfun is very similar to FEniCS (fenicsproject.org), but works only for tensor product grids and the spectral Galerking method. The code is parallelized with MPI through the [*mpi4py-fft*](https://bitbucket.org/mpi4py/mpi4py-fft) package.

The demo folder contains several examples for the Poisson, Helmholtz and Biharmonic equations. For extended documentation and installation instructions see [*ReadTheDocs*](http://shenfun.readthedocs.org). See also this [*paper*](https://rawgit.com/spectralDNS/shenfun/master/docs/demos/mekit17/pub/shenfun_bootstrap.html).

About
-----
Shenfun is developed by
     
     - Mikael Mortensen, University of Oslo

Contact
-------
For comments, issues, bug-reports and requests, please use the issue tracker of the current repository. Otherwise I can be reached at http://folk.uio.no/mikaem/contact.html

    Mikael Mortensen
    mikaem at math.uio.no
    http://folk.uio.no/mikaem/
    Department of Mathematics
    University of Oslo
    Norway
