"""
This is the **shenfun** package

What is **shenfun**?
================================

``Shenfun`` is a high performance computing platform for solving partial
differential equations (PDEs) by the spectral Galerkin method. The user
interface to shenfun is very similar to
`FEniCS <https://fenicsproject.org>`_, but applications are limited to
multidimensional tensor product grids. The code is parallelized with MPI
through the `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_
package.

``Shenfun`` enables fast development of efficient and accurate PDE solvers
(spectral order and accuracy), in the comfortable high-level Python language.
The spectral accuracy is ensured from using high-order *global* orthogonal
basis functions (Fourier, Legendre, Chebyshev, Laguerre, Hermite and Jacobi),
as opposed to finite element codes like `FEniCS <https://fenicsproject.org>`_
that are using low-order *local* basis functions. Efficiency is ensured
through vectorization (`Numpy <https://www.numpy.org/>`_), parallelization
(`mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_) and by moving critical
routines to `Cython <https://cython.org/>`_ or
`Numba <https://numba.pydata.org>`_.

``Shenfun`` has support for solving scalar and vector equations in curvilinear
coordinates, like polar or spherical coordinates.

"""
#pylint: disable=wildcard-import,no-name-in-module

__version__ = '4.2.1'
__author__ = 'Mikael Mortensen'

import numpy as np
from mpi4py import MPI
from .config import config, dumpconfig
from . import chebyshev
from . import chebyshevu
from . import legendre
from . import laguerre
from . import hermite
from . import fourier
from . import jacobi
from . import ultraspherical
from . import matrixbase
from . import la
from .coordinates import Coordinates
from .fourier import energy_fourier
from .io import *
from .matrixbase import *
from .spectralbase import inner_product, MixedFunctionSpace, BoundaryConditions
from .forms import *
from .tensorproductspace import *
from .utilities import *
from .utilities.lagrangian_particles import *
from .utilities.integrators import *
comm = MPI.COMM_WORLD
