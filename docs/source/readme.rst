shenfun
=======

Description
-----------

Shenfun is a toolbox for automating the spectral Galerkin method. The
user interface to shenfun is very similar to FEniCS (fenicsproject.org),
but works only for tensor product grids and the spectral Galerking
method. The code is parallelized with MPI through the `mpi4py-fft`_
package.

Spectral Galerkin
-----------------

The spectral Galerkin method uses the method of weighted residuals, and
solves PDEs by first creating variational forms from inner products,

.. math::

    (u, v)_w = \int_{\Omega} u(\boldsymbol{x}) \overline{v}(\boldsymbol{x}) w(\boldsymbol{x}) d\boldsymbol{x} 

where :math:`\Omega` is the computational domain, :math:`u` is a trial 
function, :math:`v` a test function (overline indicates a complex conjugate),
and :math:`w` is a weight function. The bold :math:`\boldsymbol{x}` represents 
:math:`(x,y,z)` for a 3D inner product, but shenfun may be used for any number 
of dimensions.

Consider the Poisson equation

.. math::

    -\nabla^2 u(\boldsymbol{x}) = f(\boldsymbol{x}) \quad \text{for } \, \boldsymbol{x} \in \Omega 

We obtain a variational form by multiplying with :math:`\overline{v} w` and 
integrating over the domain

.. math::

    (-\nabla^2 u, v)_w = (f, v)_w   
 
With shenfun a user chooses the appropriate bases for each dimension of the
problem, and may then combine these bases into tensor product spaces. For
example, to create a basis for a triply periodic domain, a user may proceed
as follows

>>> from shenfun import fourier, TensorProductSpace
>>> from mpi4py import MPI
>>> comm = MPI.COMM_WORLD
>>> N = (14, 15, 16)
>>> B0 = fourier.C2CBasis(N[0])
>>> B1 = fourier.C2CBasis(N[1])
>>> B2 = fourier.R2CBasis(N(2))
>>> V = TensorProductSpace(comm, (B0, B1, B2))

where C2CBasis is a Fourier basis for complex-to-complex transforms, whereas
R2CBasis is actually the same Fourier basis, but it is used on real input data,
and as such it performs real-to-complex transforms. The tensor product space
V will be distributed with the pencil method and it can here use a maximum of
14*9 CPUs (14 being the first dimension, and 9 since the last dimension is
transformed from 16 real data to 9 complex, using the Hermitian symmetry of
real transforms, i.e., the shape of a transformed array in the V space will be
(14, 15, 9)).

Getting started
---------------

Shenfun consists of classes and functions that aim at making it easy to implement
PDE's in tensor product domains. The most important everyday tools are

	* :class:`TensorProductSpace`
	* :class:`TrialFunction`
	* :class:`TestFunction`
	* :class:`Function`
	* :func:`div`
	* :func:`grad`
	* :func:`project`

Installation
------------

Shenfun is installed by cloning or forking the repository and then with
regular python distutils

::

    python setup.py install --prefix="path used for installation. Must be on the PYTHONPATH"

or in-place using

::

    python setup.py build_ext --inplace

Shenfun depends on `mpiFFT4py`_ and `mpi4py-fft`_. Other than that, it
requires `cython`_, which is used to optimize a few routines and
`pyFFTW`_ for serial fast Fourier transforms. However, since *pyFFTW*
is very slow at incorporating new pull requests, you currently need to
use the fork by `David Wells`_ for fast discrete cosine transforms.

Probably the easiest installation is achieved though Anaconda, where
also the correct dependencies will be pulled in. From the top directory
build it with

::

    conda build -c conda-forge -c spectralDNS conf/conda
    conda install -c conda-forge -c spectralDNS shenfun --use-local

You may also use precompiled binaries in the `spectralDNS`_ channel on
Anaconda cloud. Use for exampel

::

    conda create --name shenfun -c conda-forge -c spectralDNS shenfun
    source activate shenfun

which installs both shenfun, mpiFFT4py and all required dependencies,
most of which are pulled in from the conda-forge channel. There are
binaries compiled for both OSX and linux, for either Python version 2.7
or 3.6. To specify the Python version as 3.6 instead of default (used
above) you can for exampel do

::

    conda create --name shenfun_py3 -c conda-forge -c spectralDNS python=3.6 shenfun
    source activate shenfun_py3

Background
----------

Shenfun is named as a tribute to Prof. Jie Shen, as it contains many
tools for working with his modified Chebyshev and Legendre bases, as
described here:

    * Jie Shen, SIAM Journal on Scientific Computing, 15 (6), 1489-1505 (1994) (JS1)
    * Jie Shen, SIAM Journal on Scientific Computing, 16 (1), 74-87, (1995) (JS2)

Shenfun has implemented classes for the bases described in these papers,
and within each class there are methods for fast transforms, inner
products and for computing matrices arising from bilinear forms in the
spectral Galerkin method.

.. _demo: https://github.com/spectralDNS/shenfun/tree/master/demo
.. _mpiFFT4py: https://github.com/spectralDNS/mpiFFT4py
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _cython: http://cython.org
.. _pyFFTW: https://github.com/pyFFTW/pyFFTW
.. _David Wells: https://github.com/drwells/pyFFTW/tree/r2r-try-two
.. _spectralDNS: https://anaconda.org/spectralDNS
.. _Demo for the nonlinear Klein-Gordon equation: https://rawgit.com/spectralDNS/shenfun/master/docs/src/KleinGordon/kleingordon_bootstrap.html
.. _Demo for the Kuramato-Sivashinsky equation: https://rawgit.com/spectralDNS/shenfun/master/docs/src/KuramatoSivashinsky/kuramatosivashinsky_bootstrap.html
.. _Demo for Poisson equation in 1D with inhomogeneous Dirichlet boundary conditions: https://rawgit.com/spectralDNS/shenfun/master/docs/src/Poisson/poisson_bootstrap.html
.. _Demo for Poisson equation in 3D with Dirichlet in one and periodicity in remaining two dimensions: https://rawgit.com/spectralDNS/shenfun/master/docs/src/Poisson3D/poisson3d_bootstrap.html
.. _Shenfun paper: https://rawgit.com/spectralDNS/shenfun/master/docs/shenfun_bootstrap.html

