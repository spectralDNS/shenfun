import numbers
import sympy as sp
import numpy as np
from shenfun.config import config

class Coordinates:
    """Class for handling curvilinear coordinates

    Parameters
    ----------
    psi : tuple or sp.Symbol
        The new coordinates
    rv : tuple
        The position vector in terms of the new coordinates
    assumptions : Sympy assumptions
        One or more `Sympy assumptions <https://docs.sympy.org/latest/modules/assumptions/index.html>`_
    replace : sequence of two-tuples
        Use Sympy's replace with these two-tuples
    measure : Python function to replace Sympy's count_ops.
        For example, to discourage the use of powers in an expression use::

        def discourage_powers(expr):
            POW = sp.Symbol('POW')
            count = sp.count_ops(expr, visual=True)
            count = count.replace(POW, 100)
            count = count.replace(sp.Symbol, type(sp.S.One))
            return count
    """
    def __init__(self, psi, rv, assumptions=True, replace=(), measure=sp.count_ops):
        self._psi = (psi,) if isinstance(psi, sp.Symbol) else psi
        self._rv = rv
        self._assumptions = assumptions
        self._replace = replace
        self._measure = measure
        self._hi = None
        self._b = None
        self._bt = None
        self._e = None
        self._g = None
        self._gt = None
        self._gn = None
        self._ct = None
        self._det_g = {True: None, False: None}
        self._sqrt_det_g = {True: None, False: None}

    @property
    def b(self):
        return self.get_covariant_basis()

    @property
    def e(self):
        return self.get_normal_basis()

    @property
    def hi(self):
        return self.get_scaling_factors()

    @property
    def sg(self):
        if self.is_cartesian:
            return 1
        return self.get_sqrt_det_g(True)

    @property
    def coordinates(self):
        return (self.psi, self.rv, self._assumptions, self._replace)

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
        if len(self.psi) != len(self.rv):
            return False
        return sp.Matrix(self.get_covariant_metric_tensor()).is_Identity

    def get_det_g(self, covariant=True):
        """Return determinant of covariant metric tensor"""
        if self._det_g[covariant] is not None:
            return self._det_g[covariant]
        if covariant:
            g = sp.Matrix(self.get_covariant_metric_tensor()).det()
        else:
            g = sp.Matrix(self.get_contravariant_metric_tensor()).det()
        g = g.factor()
        g = self.refine(g)
        g = sp.simplify(g, measure=self._measure)
        self._det_g[covariant] = g
        return g

    def get_sqrt_det_g(self, covariant=True):
        """Return square root of determinant of covariant metric tensor"""
        if self._sqrt_det_g[covariant] is not None:
            return self._sqrt_det_g[covariant]
        g = self.get_det_g(covariant)
        sg = sp.simplify(sp.sqrt(g), measure=self._measure)
        sg = self.refine(sg)
        if isinstance(sg, numbers.Number):
            if isinstance(sg, numbers.Real):
                sg = float(sg)
            elif isinstance(sg, numbers.Complex):
                sg = complex(sg)
            else:
                raise NotImplementedError

        self._sqrt_det_g[covariant] = sg
        return sg

    def get_cartesian_basis(self):
        """Return Cartesian basis vectors"""
        return np.eye(len(self.rv), dtype=object)

    def get_scaling_factors(self):
        """Return scaling factors"""
        if self._hi is not None:
            return self._hi
        hi = np.zeros_like(self.psi)

        for i, s in enumerate(np.sum(self.b**2, axis=1)):
            hi[i] = sp.sqrt(self.refine(sp.simplify(s, measure=self._measure)))
            hi[i] = self.refine(hi[i])

        self._hi = hi
        return hi

    def get_normal_basis(self):
        if self._e is not None:
            return self._e
        b = self.b
        e = np.zeros_like(b)
        for i, bi in enumerate(b):
            l = sp.sqrt(sp.simplify(np.dot(bi, bi)))
            l = self.refine(l)
            e[i] = bi / l
        self._e = e
        return e

    def get_covariant_basis(self):
        """Return covariant basisvectors"""
        if self._b is not None:
            return self._b
        b = np.zeros((len(self.psi), len(self.rv)), dtype=object)
        for i, ti in enumerate(self.psi):
            for j, rj in enumerate(self.rv):
                b[i, j] = rj.diff(ti, 1)
                b[i, j] = sp.simplify(b[i, j], measure=self._measure)
                b[i, j] = self.refine(b[i, j])

        #if len(self.psi) == 2 and len(self.rv) == 3:
        #    b[-1] = np.cross(b[0], b[1])
        #    bl = self.refine(sp.sqrt(sp.simplify(np.dot(b[-1], b[-1]))))
        #    b[-1] = b[-1] / bl
        #    for j in range(len(self.rv)):
        #        b[-1, j] = sp.simplify(b[-1, j], measure=self._measure)
        #        b[-1, j] = self.refine(b[-1, j])
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
                bt[i, j] = sp.simplify(bt[i, j], measure=self._measure)
        self._bt = bt
        return bt

    def get_normal_metric_tensor(self):
        """Return normal metric tensor"""
        if self._gn is not None:
            return self._gn
        gn = np.zeros((len(self.psi), len(self.psi)), dtype=object)
        e = self.e
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                gn[i, j] = sp.simplify(np.dot(e[i], e[j]).expand(), measure=self._measure)
                gn[i, j] = self.refine(gn[i, j])

        self._gn = gn
        return gn

    def get_covariant_metric_tensor(self):
        """Return covariant metric tensor"""
        if self._g is not None:
            return self._g
        g = np.zeros((len(self.psi), len(self.psi)), dtype=object)
        b = self.b
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                g[i, j] = sp.simplify(np.dot(b[i], b[j]).expand(), measure=self._measure)
                g[i, j] = self.refine(g[i, j])

        self._g = g
        return g

    def get_contravariant_metric_tensor(self):
        """Return contravariant metric tensor"""
        if self._gt is not None:
            return self._gt
        g = self.get_covariant_metric_tensor()
        gt = sp.Matrix(g).inv()
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                gt[i, j] = sp.simplify(gt[i, j], measure=self._measure)
        gt = np.array(gt)
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
                    ct[k, i, j] = sp.simplify(np.dot(np.array([bij.diff(self.psi[j], 1) for bij in b[i]]), bt[k]), measure=self._measure)
        self._ct = ct
        return ct

    def get_metric_tensor(self, kind='normal'):
        if kind == 'covariant':
            gij = self.get_covariant_metric_tensor()
        elif kind == 'contravariant':
            gij = self.get_contravariant_metric_tensor()
        elif kind == 'normal':
            gij = self.get_normal_metric_tensor()
        else:
            raise NotImplementedError
        return gij

    def get_basis(self, kind='normal'):
        if kind == 'covariant':
            return self.get_covariant_basis()
        assert kind == 'normal'
        return self.get_normal_basis()

    def refine(self, sc):
        sc = sp.refine(sc, self._assumptions)
        for a, b in self._replace:
            sc = sc.replace(a, b)
        return sc

    def subs(self, s0, s1):
        b = self.get_covariant_basis()
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                b[i, j] = b[i, j].subs(s0, s1)

        g = self.get_covariant_metric_tensor()
        gt = self.get_contravariant_metric_tensor()
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                g[i, j] = g[i, j].subs(s0, s1)
                gt[i, j] = gt[i, j].subs(s0, s1)

        sg = self.get_sqrt_det_g().subs(s0, s1)
        self._sqrt_det_g[True] = sg

        hi = self.get_scaling_factors()
        for i in range(len(hi)):
            hi[i] = hi[i].subs(s0, s1)

        self._psi = tuple([p.subs(s0, s1) for p in self._psi])
        self._rv = tuple([r.subs(s0, s1) for r in self._rv])

    def latex_basis_vectors(self, symbol_names=None, replace=None, kind=None):
        if kind is None:
            kind = config['basisvectors']
        if kind == 'covariant':
            b = self.get_covariant_basis()
        elif kind == 'contravariant':
            b = self.get_contravariant_basis()
        else:
            b = self.get_normal_basis()
        psi = self.psi
        symbols = {p: str(p) for p in psi}
        if symbol_names is not None:
            symbols = symbol_names

        k = {0: '\\mathbf{i}', 1: '\\mathbf{j}', 2: '\\mathbf{k}'}
        m = ' '
        bl = 'e' if kind == 'normal' else 'b'
        for i, p in enumerate(psi):
            if kind in ('covariant', 'normal'):
                m += '\\mathbf{%s}_{%s} ='%(bl, symbols[p])
            else:
                m += '\\mathbf{%s}^{%s} ='%(bl, symbols[p])
            for j in range(b.shape[1]):
                if b[i, j] == 1:
                    m += (k[j]+'+')
                elif b[i, j] != 0:
                    if replace is not None:
                        for repl in replace:
                            assert len(repl) == 2
                            b[i, j] = b[i, j].replace(*repl)
                    sl = sp.latex(b[i, j], symbol_names=symbols)
                    if sl.startswith('-') and not isinstance(b[i, j], sp.Add):
                        m = m.rstrip('+')
                    if isinstance(b[i, j], sp.Add):
                        sl = '\\left(%s\\right)'%(sl)
                    m += (sl+'\\,'+k[j]+'+')

            m = m.rstrip('+')
            m += ' \\\\ '
        m += ' '
        return r'%s'%(m)
