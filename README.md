# shenfun

[![Build Status](https://travis-ci.org/spectralDNS/shenfun.svg?branch=master)](https://travis-ci.org/spectralDNS/shenfun)
[![CircleCI](https://circleci.com/gh/spectralDNS/shenfun.svg?style=svg)](https://circleci.com/gh/spectralDNS/shenfun)

Description
-----------
Shenfun is a toolbox for automating the spectral Galerkin method.  The user interface to `shenfun` is very similar to FEniCS (fenicsproject.org), but works only for tensor product grids and the spectral Galerking method. The code is parallelized with MPI through the [*mpi4py-fft*](https://bitbucket.org/mpi4py/mpi4py-fft) package.

The spectral Galerkin method uses the method of weighted residuals, and solves PDEs by first creating variational forms from inner products, 
<p align="center">
    <img src="https://www.dropbox.com/s/de2s19s0wyb2mi3/inner_product_3D.png?dl=1" alt="Poisson equation"/>
</p>
<p align="center">

where _omega_ is the computational domain, _u_ is a trial function, _v_ a test function (overline indicates a complex conjugate), and _w_ is a weight function. The bold _**x**_ represents (x,y,z) for a 3D inner product, but *shenfun* may be used for any number of dimensions. 

Consider the Poisson equation 
<p align="center">
    <img src="https://www.dropbox.com/s/vrvkin9dsw7bq57/poisson_3D_2.png?dl=1" alt="Poisson equation"/>
</p>
<p align="center">

We obtain a variational form by multiplying with _vw_ and integrating over the domain

<p align="center">
    <img src="https://www.dropbox.com/s/c7zbebj6n9r426q/poisson_3D_var.png?dl=1" alt="Poisson equation variational form"/>
</p>
<p align="center">

With *shenfun* a user chooses the appropriate bases for each dimension of the problem, and may then combine these bases into tensor product spaces. For example, to create a basis for a triply periodic domain, a user may proceed as follows

```python
   from shenfun import fourier, TensorProductSpace
   from mpi4py import MPI
   
   comm = MPI.COMM_WORLD
   N = (14, 15, 16)
   B0 = fourier.C2CBasis(N[0])
   B1 = fourier.C2CBasis(N[1])
   B2 = fourier.R2CBasis(N(2))
   V = TensorProductSpace(comm, (B0, B1, B2))
```
where `C2CBasis` is a Fourier basis for complex-to-complex transforms, whereas `R2CBasis` is actually the same Fourier basis, but it is used on real input data, and as such it performs real-to-complex transforms. The tensor product space `V` will be distributed with the *pencil* method and it can here use a maximum of 14*9 CPUs (14 being the first dimension, and 9 since the last dimension is transformed from 16 real data to 9 complex, using the Hermitian symmetry of real transforms, i.e., the shape of a transformed array in the V space will be (14, 15, 9)). 

To solve a Poisson problem with the above triply periodic tensor product space, one may assemble the coefficient matrix as

```python
   from shenfun import TestFunction, TrialFunction, inner, div, grad
   
   u = TrialFunction(V)
   v = TestFunction(V)
   A = inner(-div(grad(u)), v)
```
or similarly using integration by parts

```python
   A = inner(grad(u), grad(v))
```
Note the similarity with FEniCS, and the similarity between code and mathematical problem.

To solve the Poisson equation, we need to first assemble a right hand side, for example a random function

```python
    import numpy as np
    from shenfun import Function
    
    f = Function(V)
    f[:] = np.random.random(f.shape)
    f_hat = inner(f, v)
    f_hat = A.solve(f_hat)
```
Complete examples for the Poisson equation with various boundary conditions are given in the *demo* folder.

Installation
------------
*shenfun* is installed by cloning or forking the repository and then with regular python distutils

    python setup.py install --prefix="path used for installation. Must be on the PYTHONPATH"
    
or in-place using

    python setup.py build_ext --inplace

*shenfun* depends on [mpiFFT4py](https://github.com/spectralDNS/mpiFFT4py) and [mpi4py-fft](https://bitbucket.org/mpi4py/mpi4py-fft). Other than that, it requires [*cython*](http://cython.org), which is used to optimize a few routines and [*pyFFTW*](https://github.com/pyFFTW/pyFFTW) for serial fast Fourier transforms. However, since *pyFFTW* is very slow at incorporating new pull requests, you currently need to use the fork by [David Wells](https://github.com/drwells/pyFFTW/tree/r2r-try-two) for fast discrete cosine transforms.

Probably the easiest installation is achieved though Anaconda, where also the correct dependencies will be pulled in. From the top directory build it with

    conda build -c conda-forge -c spectralDNS conf/conda
    conda install -c conda-forge -c spectralDNS shenfun --use-local

You may also use precompiled binaries in the [*spectralDNS*](https://anaconda.org/spectralDNS) channel on Anaconda cloud. Use for exampel

    conda create --name shenfun -c conda-forge -c spectralDNS shenfun
    source activate shenfun

which installs both spectralDNS, mpiFFT4py and all required dependencies, most of which are pulled in from the conda-forge channel. There are binaries compiled for both OSX and linux, for either Python version 2.7 or 3.6. To specify the Python version as 3.6 instead of default (used above) you can for exampel do

    conda create --name shenfun_py3 -c conda-forge -c spectralDNS python=3.6 shenfun
    source activate shenfun_py3

Background
----------

Shenfun is named as a tribute to Prof. Jie Shen, as it contains many tools for working with his modified Chebyshev and Legendre bases, as described here:
  * Jie Shen, SIAM Journal on Scientific Computing, 15 (6), 1489-1505 (1994) (JS1)
  * Jie Shen, SIAM Journal on Scientific Computing, 16 (1), 74-87, (1995) (JS2)

Shenfun has implemented classes for the bases described in these papers, and within each class there are methods for fast transforms, inner products and for computing matrices arising from bilinear forms in the spectral Galerkin method. The following bases are defined in submodules `shenfun.chebyshev`, `shenfun.legendre` and `shenfun.fourier`

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

Matrices that arise with Shen's bases and the spectral Galerkin method are often very sparse. As such, `shenfun` defines it's own sparse matrix class `SparseMatrix` and the subclassed `SpectralMatrix` in `shenfun.matrixbase.py`. The matrix baseclass `SparseMatrix` is subclassing a regular Python dictionary, and its keys and values are, respectively, the offsets and the diagonals. For example, we may declare a tridiagonal matrix of shape N x N as

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

which can also be achieved with the `inner` method, using the `TestFunction` class

```python
    >>> from shenfun import inner, TestFunction
    >>> v = TestFunction(SD)
    >>> fs = inner(v, fj)
```

A `mass` matrix can be assembled with

```python
    >>> u = TrialFunction(SD)
    >>> mass = inner(v, u)
    >>> mass
    {-2: array([-1.57079633]),
      0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265,  3.14159265, 3.14159265]),
      2: array([-1.57079633])}
```

You may notice that `mass` takes advantage of the fact that two diagonals are constant.

The `inner` method may be used to compute any bilinear form. For example the stiffness matrix `K`

```python
    >>> K = inner(v, div(grad(u)))
```
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


