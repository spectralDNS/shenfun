"""
Main module for import from shenfun
"""
#pylint: disable=wildcard-import

import numpy as np
from . import chebyshev
from . import legendre
from . import fourier
from . import matrixbase
from .fourier import energy_fourier
from .matrixbase import *
from .forms.project import *
from .forms.inner import *
from .forms.operators import *
from .forms.arguments import *
from .tensorproductspace import *
from .utilities import *
from .utilities.integrators import *
from .utilities.h5py_writer import *
from .utilities.nc_writer import *
from .utilities.generate_xdmf import *
from .optimization import Cheb, la, Matvec, convolve, evaluate

