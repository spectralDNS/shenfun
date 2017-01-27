# shenfun

[![Build Status](https://travis-ci.org/spectralDNS/shenfun.svg?branch=master)](https://travis-ci.org/spectralDNS/shenfun)
[![CircleCI](https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg)](https://circleci.com/gh/spectralDNS/shenfun)

Description
-----------

Shenfun contains tools for working with Jie Shen's modified Chebyshev and Legendre bases described here:
  * Jie Shen, SIAM Journal on Scientific Computing, 15 (6), 1489-1505 (1994)
  * Jie Shen, SIAM Journal on Scientific Computing, 16 (1), 74-87, (1995)

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
    >>> A
    {-1: array([ 1.,  1.,  1.]),
      0: array([-2., -2., -2., -2.]),
      1: array([ 1.,  1.,  1.])}
    >>> A.diags()
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 10 stored elements (3 diagonals) in DIAgonal format>
    >>> A.diags().toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])





  
