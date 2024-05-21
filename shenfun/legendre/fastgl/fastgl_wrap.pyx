# distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3str
from . cimport fastgl_wrap

import numpy as np

def getGLPair(int N, int k):
    """Return point and weight k for N-point Gauss-Legendre rule

    Parameters
    ----------
    N : int
        The total number of quadrature points
    k : int
        The k'th point in the N-point quadrature rule
    """
    f = fastgl_wrap.GLPair(N, k)
    return tuple((f.x(), f.weight))

cdef GLPairs(int N, double [::1] x, double [::1] w):
    cdef:
        int i

    for i in range(N):
        f = fastgl_wrap.GLPair(N, N-i)
        x[i] = f.x()
        w[i] = f.weight

def leggauss(int N):
    """Gauss-Legendre quadrature

    Computes the points and weights for N-point Gauss-Legendre quadrature.

    Parameters
    ----------
    N : int
        The total number of quadrature points/weights

    Note
    ----
    These points and weights are accurate for more than millions of points.
    The code is based on the paper

        "I. Bogaert, 'Iteration-Free Computation of Gauss-Legendre Quadrature
        Nodes and Weights', SIAM Journal on Scientific Computing, 36, 3,
        A1008-A1026, 2014"

    """
    x = np.zeros(N)
    w = np.zeros(N)
    GLPairs(N, x, w)
    return x, w
