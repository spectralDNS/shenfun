import sympy as sp
import numpy as np

class Coordinates(object):
    """Class for handling curvilinear coordinates

    Parameters
    ----------
    psi : tuple
        The new coordinates
    rv : tuple
        The position vector in terms of the new coordinates
    """
    def __init__(self, psi, rv):
        self._psi = psi
        self._rv = rv
        self._hi = None
        self._b = None
        self._bt = None
        self._g = None
        self._gt = None
        self._ct = None

    @property
    def b(self):
        return self.get_covariant_basis()

    @property
    def hi(self):
        return self.get_scaling_factors()

    @property
    def coordinates(self):
        return (self.psi, self.rv)

    @property
    def psi(self):
        return self._psi

    @property
    def rv(self):
        return self._rv

    @property
    def is_orthogonal(self):
        return sp.Matrix(self.get_covariant_metric_tensor()).is_diagonal()

    @property
    def is_cartesian(self):
        return sp.Matrix(self.get_covariant_metric_tensor()).is_Identity

    def get_g(self):
        """Return determinant of covariant metric tensor"""
        return sp.Matrix(self.get_covariant_metric_tensor()).det()

    def get_sqrt_g(self):
        """Return square root of determinant of covariant metric tensor"""
        g = self.get_g()
        return sp.simplify(sp.srepr(sp.sqrt(g)).replace('Abs', ''))

    def get_cartesian_basis(self):
        """Return Cartesian basis vectors"""
        return np.eye(len(self.rv), dtype=object)

    def get_scaling_factors(self):
        """Return scaling factors"""
        if self._hi is not None:
            return self._hi
        hi = np.zeros_like(self.psi)
        for i, s in enumerate(np.sum(self.b**2, axis=1)):
            hi[i] = sp.simplify(sp.srepr(sp.simplify(sp.sqrt(s))).replace('Abs', ''))
        self._hi = hi
        return hi

    def get_covariant_basis(self):
        """Return covariant basisvectors"""
        if self._b is not None:
            return self._b
        b = np.zeros((len(self.psi), len(self.rv)), dtype=object)
        for i, ti in enumerate(self.psi):
            for j, rj in enumerate(self.rv):
                b[i, j] = sp.simplify(rj.diff(ti, 1))
        self._b = b
        return b

    def get_contravariant_basis(self):
        """Return contravariant basisvectors"""
        if self._bt is not None:
            return self._bt
        bt = np.zeros_like(self.b)
        g = self.get_contravariant_metric_tensor()
        b = self.b
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                bt[i] += g[i, j]*b[j]

        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                bt[i, j] = sp.simplify(bt[i, j])
        self._bt = bt
        return bt

    def get_covariant_metric_tensor(self):
        """Return covariant metric tensor"""
        if self._g is not None:
            return self._g
        g = np.zeros((len(self.psi), len(self.psi)), dtype=object)
        b = self.b
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                g[i, j] = sp.simplify(np.dot(b[i], b[j]))
        self._g = g
        return g

    def get_contravariant_metric_tensor(self):
        """Return contravariant metric tensor"""
        if self._gt is not None:
            return self._gt
        g = self.get_covariant_metric_tensor()
        gt = np.array(sp.Matrix(g).inv())
        self._gt = gt
        return gt

    def get_christoffel_second(self):
        """Return Christoffel symbol of second kind"""
        if self._ct is not None:
            return self._ct
        b = self.get_covariant_basis()
        bt = self.get_contravariant_basis()
        ct = np.zeros((len(self.psi),)*3, object)
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                for k in range(len(self.psi)):
                    ct[k, i, j] = sp.simplify(np.dot(np.array([bij.diff(self.psi[j], 1) for bij in b[i]]), bt[k]))
        self._ct = ct
        return ct
