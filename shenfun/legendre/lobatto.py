"""
Module contains some useful methods for Legendre quadrature.
"""
import numpy as np
from numpy.polynomial import legendre as leg

def legendre_lobatto_nodes_and_weights(N, tol=1e-16):
    """Return points and weights for Legendre-Lobatto quadrature

    Parameters
    ----------
        N : int
            Number of quadrature points
    """
    M = N//2
    j = np.arange(1, M)
    x = np.zeros(N)
    x[0] = -1.
    x[-1] = 1.
    x[1:M] = -np.cos((j+0.25)*np.pi/N - 3./(8.*N*np.pi*(j+0.25)))
    Ln = leg.Legendre.basis(N-1)
    Ld = Ln.deriv(1)
    Ld2 = Ln.deriv(2)
    converged = False
    dx = np.zeros_like(x[1:M])
    y = x[1:M]
    count = 0
    prev = 1
    while not converged and count < 10:
        dx[:] = Ld(y)/Ld2(y)
        y -= dx
        error = np.linalg.norm(dx)
        converged = (error < tol) or (abs(error - prev) < tol)
        count += 1
        prev = error
        #print(count, error)

    MM = M if N % 2 == 0 else M+1
    x[MM:-1] = -x[1:M][::-1]
    w = 2./(N*(N-1)*Ln(x)**2)
    return x, w


