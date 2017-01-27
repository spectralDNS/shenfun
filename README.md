# shenfun

[![Build Status](https://travis-ci.org/spectralDNS/shenfun.svg?branch=master)](https://travis-ci.org/spectralDNS/shenfun)
[![CircleCI](https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg)](https://circleci.com/gh/spectralDNS/shenfun)

Description
-----------

Shenfun contains tools for working with Jie Shen's modified Chebyshev and Legendre bases described here:
  * Jie Shen, SIAM Journal on Scientific Computing, 15 (6), 1489-1505 (1994) (JS1)
  * Jie Shen, SIAM Journal on Scientific Computing, 16 (1), 74-87, (1995) (JS2)

Shenfun has implemented classes for the bases described in these papers, and within each class there are methods for fast transforms, scalar products and for computing matrices arising with the spectral Galerkin method. The following bases are defined in submodules `shenfun.chebyshev` and `shendun.legendre`

  * shenfun.chebyshev
    * ChebyshevBasis - Regular Chebyshev 
    * ShenDirichletBasis - Dirichlet boundary conditions
    * ShenNeumannBasis - Neumann boundary conditions (homogeneous)
    * ShenBiharmonicBasis - Homogeneous Dirichlet and Neumann boundary conditions
    
  * shenfun.legendre
    * LegendreBasis - Regular Legendre
    * LegendreDirichletBasis - Dirichlet boundary conditions
    
Matrices that arise with Shen's bases and the spectral Galerkin method are often very sparse. As such, `shenfun` defines it's own sparse matrix class `ShenMatrix` in `shenfun.matrixbase.py`. The matrix class is subclassing a regular Python dictionary, and its keys and values are, respectively, the offsets and the diagonals. For example, we may declare a tridiagonal matrix of shape N x N as

    >>> N = 4
    >>> d = {-1: 1, 0: -2, 1: 1}
    >>> A = SparseMatrix(d, (N, N))

or similarly as

    >>> import numpy as np
    >>> d = {-1: np.ones(N-1), 0: -2*np.ones(N)}
    >>> d[1] = d[-1]  # Symmetric, reuse np.ones array
    >>> A = SparseMatrix(d, (N, N))
    
The matrix is a subclassed dictionary

    >>> A
    {-1: array([ 1.,  1.,  1.]),
      0: array([-2., -2., -2., -2.]),
      1: array([ 1.,  1.,  1.])}
and if you want a regular `Scipy` sparse matrix, just do

    >>> A.diags()
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 10 stored elements (3 diagonals) in DIAgonal format>
    >>> A.diags().toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])

A Dirichlet basis is created as

    >>> from shenfun.chebyshev.bases import ShenDirichletBasis
    >>> SD = ShenDirichletBasis()

Now one may project for example a random vector to this space using

    >>> import numpy as np
    >>> fj = np.random.random(N)
    >>> fk = np.zeros_like(fj)
    >>> fk = SD.forward(fj, fk) # Gets expansion coefficients of Shen Dirichlet basis

and back to real physical space again

    >>> fj = SD.backward(fk, fj)
    
Note that `fj` now will be different than the original `fj` since it now has homogeneous boundary conditions. However, if we transfer back and forth one more time, starting from `fj` which is in the Shen Dirichlet function space, then we come back to the same array:

    >>> fj_copy = fj.copy()
    >>> fk = SD.forward(fj, fk)
    >>> fj = SD.backward(fk, fj)
    >>> assert np.allclose(fj, fj_copy) # Is True

The `SD` class can also be used to compute the scalar product of an array

    >>> fs = np.zeros_like(fj)
    >>> fs = SD.scalar_product(fj, fs)

The bases can also be used to assemble coefficient matrices, like the mass matrix

    >>> mass = ShenMatrix({}, 8, (SD, 0), (SD, 0))

This `mass` matrix will be the same as Eq. (2.5) of JS1:

    >>> mass
    {-2: array([-1.57079633, -1.57079633, -1.57079633, -1.57079633]),
      0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265,  3.14159265, 3.14159265]),
      2: array([-1.57079633, -1.57079633, -1.57079633, -1.57079633])}

and it will be (up to roundoff) the same as `BDDmat` from `shenfun.chebyshev.matrices`

    >>> from shenfun.chebyshev.matrices import BDDmat
    >>> mass = BDDmat(np.arange(8))
    >>> mass
    {-2: array([-1.57079633]),
      0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265,  3.14159265, 3.14159265]),
      2: array([-1.57079633])}

You may notice that `BDDmat` takes advantage of the fact that two diagonals are constant. There are many such precomputed matrices. For example the stiffness matrix `ADDmat`

    >>> from shenfun.chebyshev.matrices import ADDmat
    >>> stiffness1 = ADDmat(np.arange(8))
    >>> stiffness2 = ShenMatrix({}, 8, (SD, 2), (SD, 0), -1.)

Here `stiffness1` and `stiffness2` are equal up to roundoff, but `stiffness2` is automatically generated from Vandermonde type matrices, whereas `stiffness1` uses analytical expressions for the diagonals (see `class ADDmat` in `shenfun.chebyshev.matrices.py`). 

Square matrices have implemented a solve method that is using fast direct LU decomposition or similar (TDMA/PDMA). For example

    >>> fj = np.random.random(8)
    >>> fk = np.zeros_like(fj)
    >>> fk = SD.scalar_product(fj, fk)
    >>> fk = stiffness1.solve(fk)



    



  
