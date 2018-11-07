---
title: 'Shenfun: High performance spectral Galerkin method'
tags:
 - Spectral Galerkin
 - Fourier
 - Chebyshev
 - Legendre
 - MPI
authors:
 - name: Mikael Mortensen
   orcid:  0000-0002-3293-7573
   affiliation: "1"
affiliations:
 - name: University of Oslo, Department of Mathematics
   index: 1
date: 6 November 2018
bibliography: paper.bib
---

# Summary

``Shenfun`` is a toolbox for automating the spectral Galerkin method. The user
interface to ``shenfun`` is very similar to FEniCS (https://fenicsproject.org),
but applications are limited to simple, yet multi-dimensional, tensor
product grids.

With ``shenfun`` one can solve a wide range of PDEs, where one dimension can
be inhomogeneous, with the remaining required to be periodic. The
periodic dimensions are discretized using Fourier exponentials as basis
functions. For the inhomogeneous direction, we use combinations of
Chebyshev or Legendre polynomials, as described by J. Shen [@shen95; @shen94].

The code is parallelized with MPI through the mpi4py-fft
(https://bitbucket.org/mpi4py/mpi4py-fft) package, and can run for
thousands of processors on supercomputers. The parallelization is automated
and highly configurable (slab, pencil), using a new algorithm [@dalcin18]
for global redistribution of multidimensional arrays.

``Shenfun`` is documented, with installation instructions and demo
programs, on readthedocs (http://shenfun.readthedocs.org).
A range of extended demonstration programs are included for, e.g., the
Poisson, Klein-Gordon and Kuramato-Sivashinsky equations.

``Shenfun`` has been designed as a low entry-level research tool for physicists
[@mortensen17] in need of highly accurate numerical methods for high
performance computing. The primary focus has been for turbulence and
transition to turbulence, where simulations are extremely sensitive to
disturbances, and numerical diffusion or dispersion are unacceptable.
Spectral methods are well known for their accuracy and efficiency due to fast
transform methods, e.g., Fast Fourier Transforms (FFTs). Combined with
the Galerkin method, we get numerical methods that are truly state of
the art.

In the spectralDNS repository (https://github.com/spectralDNS/spectralDNS)
there are applications using ``shenfun`` for forced isotropic turbulence
(Navier-Stokes equations), turbulent channel flows [@mortensen17b] and
Rayleigh-Bénard flow. However, ``shenfun`` is by no means limited to the
equations for fluid flow and it should be equally efficient for a wide
range of scientific applications.

# Acknowledgements

We acknowledge support from the 4DSpace Strategic Research Initiative at the
University of Oslo

# References
