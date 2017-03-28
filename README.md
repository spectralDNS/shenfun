# shenfun

[![Build Status](https://travis-ci.org/spectralDNS/shenfun.svg?branch=master)](https://travis-ci.org/spectralDNS/shenfun)
[![CircleCI](https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg)](https://circleci.com/gh/spectralDNS/shenfun)

Description
-----------

Shenfun contains tools for working with Jie Shen's modified Chebyshev and Legendre bases described here:
  * Jie Shen, SIAM Journal on Scientific Computing, 15 (6), 1489-1505 (1994) (JS1)
  * Jie Shen, SIAM Journal on Scientific Computing, 16 (1), 74-87, (1995) (JS2)

Shenfun has implemented classes for the bases described in these papers, and within each class there are methods for fast transforms, scalar products and for computing matrices arising from bilinear forms in the spectral Galerkin method. The following bases are defined in submodules `shenfun.chebyshev`, `shenfun.legendre` and `shenfun.fourier`

* shenfun.chebyshev.bases
  * Basis - Regular Chebyshev 
  * ShenDirichletBasis - Dirichlet boundary conditions
  * ShenNeumannBasis - Neumann boundary conditions (homogeneous)
  * ShenBiharmonicBasis - Homogeneous Dirichlet and Neumann boundary conditions

* shenfun.legendre.bases
  * Basis - Regular Legendre
  * ShenDirichletBasis - Dirichlet boundary conditions
  * ShenNeumannBasis - Neumann boundary conditions (homogeneous)
  * ShenBiharmonicBasis - Homogeneous Dirichlet and Neumann boundary conditions

* shenfun.fourier.bases
  * R2CBasis - Real to complex Fourier transforms
  * C2CBasis - Complex to complex transforms

Matrices that arise with Shen's bases and the spectral Galerkin method are often very sparse. As such, `shenfun` defines it's own sparse matrix class `SparseMatrix` and the subclassed `ShenMatrix` in `shenfun.matrixbase.py`. The matrix baseclass `SparseMatrix` is subclassing a regular Python dictionary, and its keys and values are, respectively, the offsets and the diagonals. For example, we may declare a tridiagonal matrix of shape N x N as

```python
    >>> N = 4
    >>> d = {-1: 1, 0: -2, 1: 1}
    >>> A = SparseMatrix(d, (N, N))
```

or similarly as

```python
    >>> import numpy as np
    >>> d = {-1: np.ones(N-1), 0: -2*np.ones(N)}
    >>> d[1] = d[-1]  # Symmetric, reuse np.ones array
    >>> A = SparseMatrix(d, (N, N))
```

The matrix is a subclassed dictionary

```python
    >>> A
    {-1: array([ 1.,  1.,  1.]),
      0: array([-2., -2., -2., -2.]),
      1: array([ 1.,  1.,  1.])
```

and if you want a regular `Scipy` sparse matrix, just do

```python
    >>> A.diags()
    <4x4 sparse matrix of type ‘<class ‘numpy.float64’>’
        with 10 stored elements (3 diagonals) in DIAgonal format>
    >>> A.diags().toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])
```

A Dirichlet basis with 8 quadrature points can be created as

```python
    >>> from shenfun.chebyshev.bases import ShenDirichletBasis
    >>> N = 8
    >>> SD = ShenDirichletBasis(N)
```

Now one may project for example a random vector to this space using

```python
    >>> import numpy as np
    >>> fj = np.random.random(N)
    >>> fk = np.zeros_like(fj)
    >>> fk = SD.forward(fj, fk) # Gets expansion coefficients of Shen Dirichlet basis
```

and back to real physical space again

```python
    >>> fj = SD.backward(fk, fj)
``` 

Note that `fj` now will be different than the original `fj` since it now has homogeneous boundary conditions. However, if we transfer back and forth one more time, starting from `fj` which is in the Shen Dirichlet function space, then we come back to the same array:

```python
    >>> fj_copy = fj.copy()
    >>> fk = SD.forward(fj, fk)
    >>> fj = SD.backward(fk, fj)
    >>> assert np.allclose(fj, fj_copy) # Is True
```

The `SD` class can also be used to compute the scalar product of an array

```python
    >>> fs = np.zeros_like(fj)
    >>> fs = SD.scalar_product(fj, fs)
```

The bases can also be used to assemble inner products of bilinear forms, like the mass matrix

```python
    >>> from shenfun import inner_product
    >>> mass = inner_product((SD, 0), (SD, 0))
```

This `mass` matrix will be the same as Eq. (2.5) of JS1:

```python
    >>> mass
    {-2: array([-1.57079633]),
      0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265,  3.14159265, 3.14159265]),
      2: array([-1.57079633])}
```

You may notice that `mass` takes advantage of the fact that two diagonals are constant. 

The `inner_product` may be used to compute any bilinear form. For example the stiffness matrix `K`

```python
    >>> K = inner_product((SD, 0), (SD, 2))
```
Note that the `inner_product` takes as arguments two bases, where the first represents the test function and the matrix row, whereas the second represents the trial function and the matrix column. The integer in the tuples determines how many times the basis is differentiated.

Square matrices have implemented a solve method that is using fast direct LU decomposition or similar (TDMA/PDMA). For example, to solve the linear system `Ku=b`

```python
    >>> fj = np.random.random(N)
    >>> b = np.zeros_like(fj)
    >>> b = SD.scalar_product(fj, b)
    >>> u = np.zeros_like(b)
    >>> u = K.solve(b, u)
```

All methods are designed to work along any dimension of a multidimensional array. Very little differs in the users interface. Consider, for example, the previous example on a three-dimensional cube 

```python
    >>> fj = np.random.random((N, N, N))
    >>> b = np.zeros_like(fj)
    >>> b = SD.scalar_product(fj, b)
    >>> u = np.zeros_like(b)
    >>> u = K.solve(b, u)
```
where `K` is exactly the same as before, from the 1D example. The matrix solve is applied along the first dimension since this is the default behaviour.


