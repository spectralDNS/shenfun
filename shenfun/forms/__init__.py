#pylint: disable=missing-docstring
from .project import *
from .inner import *
from .operators import *
from .arguments import *

def extract_bc_matrices(mats):
    """Extract boundary matrices from list of ``mats``

    Parameters
    ----------
    mats : list of list of :class:`.TPMatrix`es

    Returns
    -------
    list
        list of boundary matrices.

    Note
    ----
    The ``mats`` list is modified in place since boundary matrices are
    extracted.
    """
    bc_mats = []
    for a in mats:
        for b in a:
            if b.is_bc_matrix():
                bc_mats.append(b)
                a.remove(b)
    return bc_mats
