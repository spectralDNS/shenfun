import numba as nb
import numpy as np

@nb.jit(nopython=True, fastmath=True, cache=True)
def legendre_shendirichlet_scalar_product(xj, wj, input_array, output_array, is_scaled=True):
    N = xj.shape[0]
    phi_i = np.zeros_like(xj)
    Lnm = np.ones_like(xj)
    Ln = xj.copy()
    Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
    for i in range(N-2):
        s = 0.0
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        ss = np.sqrt(4*i+6)
        for j in range(N):
            phi_i[j] = Lnm[j]-Lnp[j]
            if is_scaled:
                phi_i[j] /= ss
            s += phi_i[j]*wj[j]*input_array[j]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]
        output_array[i] = s

@nb.jit(nopython=True, fastmath=True, cache=True)
def legendre_shenneumann_scalar_product(xj, wj, input_array, output_array):
    N = xj.shape[0]
    phi_i = np.zeros_like(xj)
    Lnm = np.ones_like(xj)
    Ln = xj.copy()
    Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
    for i in range(N-2):
        s = 0.0
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        for j in range(N):
            phi_i[j] = Lnm[j]-Lnp[j]*(i*(i+1)/(i+2)/(i+3))
            s += phi_i[j]*wj[j]*input_array[j]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]
        output_array[i] = s

@nb.jit(nopython=True, fastmath=True, cache=True)
def legendre_shendirichlet_evaluate_expansion_all(xj, input_array, output_array, is_scaled=True):
    N = xj.shape[0]
    phi_i = np.zeros_like(xj)
    Lnm = np.ones_like(xj)
    Ln = xj.copy()
    Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
    output_array[:] = 0
    for i in range(N-2):
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        ss = np.sqrt(4*i+6)
        for j in range(N):
            phi_i[j] = Lnm[j]-Lnp[j]
            if is_scaled:
                phi_i[j] /= ss

            output_array[j] += phi_i[j]*input_array[i]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]

@nb.jit(nopython=True, fastmath=True, cache=True)
def legendre_shenneumann_evaluate_expansion_all(xj, input_array, output_array):
    N = xj.shape[0]
    phi_i = np.zeros_like(xj)
    Lnm = np.ones_like(xj)
    Ln = xj.copy()
    Lnp = ((2+1)*xj*Ln - 1*Lnm)/2
    output_array[:] = 0
    for i in range(N-2):
        s2 = (i+2)/(i+3)
        s1 = (2*(i+2)+1)/(i+3)
        for j in range(N):
            phi_i[j] = Lnm[j]-Lnp[j]*(i*(i+1)/(i+2)/(i+3))
            output_array[j] += phi_i[j]*input_array[i]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*xj[j]*Ln[j] - s2*Lnm[j]
