"""
This is the **shenfun** package

What is **shenfun**?
================================

Shenfun is a toolbox for automating the spectral Galerkin method.  The user
interface to shenfun is very similar to FEniCS (fenicsproject.org), but works
only for tensor product grids and the spectral Galerkin method. The code is
parallelized with MPI through the
[*mpi4py-fft*](https://bitbucket.org/mpi4py/mpi4py-fft) package.

"""
#pylint: disable=wildcard-import,no-name-in-module

__version__ = '1.2.0'
__author__ = 'Mikael Mortensen'

import numpy as np
from . import chebyshev
from . import legendre
from . import fourier
from . import matrixbase
from .fourier import energy_fourier
from .io import *
from .matrixbase import *
from .forms import *
from .tensorproductspace import *
from .utilities import *
from .utilities.lagrangian_particles import *
from .utilities.integrators import *
from .optimization import Cheb, la, Matvec, convolve, evaluate
