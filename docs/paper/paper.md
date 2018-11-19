---
title: 'Shenfun: High performance spectral Galerkin computing platform'
tags:
 - Spectral Galerkin
 - Fourier
 - Chebyshev
 - Legendre
 - MPI
 - Python
authors:
 - name: Mikael Mortensen
   orcid:  0000-0002-3293-7573
   affiliation: "1"
affiliations:
 - name: University of Oslo, Department of Mathematics
   index: 1
date: 7 November 2018
bibliography: paper.bib
---

# Summary

``Shenfun`` is an open-source computing platform for solving partial
differential equations (PDEs) by the spectral Galerkin method. The user
interface to ``shenfun`` is very similar to FEniCS (https://fenicsproject.org),
but applications are limited to simple, yet multidimensional, tensor
product grids.

``Shenfun`` enables fast development of efficient and accurate PDE solvers (spectral 
order and accuracy), in the comfortable high-level Python language. The spectral 
accuracy is ensured by using high-order *global* orthogonal basis functions 
(Fourier, Legendre and Chebyshev), as opposed to finite element codes like FEniCS
that are using low-order *local* basis functions. Efficiency is ensured through 
vectorization, parallelization (MPI) and by moving critical routines to Cython 
(https://cython.org). 

With ``shenfun`` one can solve a wide range of PDEs, with the limitation that
one dimension can be inhomogeneous (with Dirichlet/Neumann type of boundaries),
whereas the remaining dimensions are required to be periodic. The
periodic dimensions are discretized using Fourier exponentials as basis
functions. For the inhomogeneous direction, we use combinations of
Chebyshev or Legendre polynomials, as described by J. Shen [@shen95; @shen94].

The code is parallelized with MPI through the mpi4py-fft
(https://bitbucket.org/mpi4py/mpi4py-fft) package, and has been run for
thousands of processors on various supercomputers. The parallelization is
automated and highly configurable (slab, pencil), using a new algorithm
[@dalcin18] for global redistribution of multidimensional arrays.

``Shenfun`` is documented, with installation instructions and demo
programs, on readthedocs (http://shenfun.readthedocs.org).
Extended demonstration programs are included for, e.g., the
Poisson, Klein-Gordon and Kuramato-Sivashinsky equations.

``Shenfun`` has been designed as a low entry-level research tool for physicists
[@mortensen17] in need of highly accurate numerical methods for high
performance computing. The primary target for the development has been
turbulence and transition to turbulence, where simulations are extremely
sensitive to disturbances, and numerical diffusion or dispersion are
unacceptable. Spectral methods are well known for their accuracy and
efficiency, taking advantage of fast transform methods, like the Fast Fourier
Transforms (FFTs). Combined with the Galerkin method and Shen's robust
composite bases, this leads to well conditioned linear algebra systems and
numerical schemes that are truly exceptional. Highly efficient direct
solvers [@mortensen17] are provided for Poisson, Helmholtz and Biharmonic
systems of equations, arising naturally with the current bases for a wide
range of problems.

In the spectralDNS repository (https://github.com/spectralDNS/spectralDNS)
there are applications utilizing ``shenfun`` for forced isotropic turbulence
(Navier-Stokes equations), turbulent channel flows [@mortensen17b],
Magnetohydrodynamics (MHD) and Rayleigh-Bénard flow. However, ``shenfun`` is
by no means limited to the equations for fluid and plasma flows and it should
be equally efficient for other scientific applications.

# Acknowledgements

We acknowledge support from the 4DSpace Strategic Research Initiative at the
University of Oslo

# References
