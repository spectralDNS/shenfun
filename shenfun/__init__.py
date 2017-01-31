from numpy import arange, float
from . import chebyshev
from . import legendre

def inner_product(test, trial, N):
    assert trial[0].__module__ == test[0].__module__
    k = arange(N).astype(float)
    if isinstance(test[0], chebyshev.ChebyshevBase):
        return chebyshev.ChebyshevMatrices[(test, trial)](k)

    elif isinstance(test[0], legendre.LegendreBase):
        return legendre.LegendreMatrices[(test, trial)](k)
