"""
Module for implementing helper functions.
"""
from numbers import Number
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
from collections import defaultdict
import math
import numpy as np
import sympy as sp
from scipy.fftpack import dct
from shenfun.optimization import optimizer

__all__ = ['dx', 'clenshaw_curtis1D', 'CachedArrayDict', 'surf3D', 'wrap_periodic',
           'outer', 'apply_mask', 'integrate_sympy', 'mayavi_show']


def dx(u):
    r"""Compute integral of u over domain

    .. math::

        \int_{\Omega} u dx

    Parameters
    ----------

        u : Array
            The Array to integrate

    """
    T = u.function_space()
    uc = u.copy()
    dim = len(u.shape)
    if dim == 1:
        w = T.points_and_weights(weighted=False)[1]
        return np.sum(uc*w).item()

    for ax in range(dim):
        uc = uc.redistribute(axis=ax)
        w = T.bases[ax].points_and_weights(weighted=False)[1]
        sl = [np.newaxis]*len(uc.shape)
        sl[ax] = slice(None)
        uu = np.sum(uc*w[tuple(sl)], axis=ax)
        sl = [slice(None)]*len(uc.shape)
        sl[ax] = np.newaxis
        uc[:] = uu[tuple(sl)]
    return uc.flat[0]

def clenshaw_curtis1D(u, quad="GC"):  # pragma: no cover
    """Clenshaw-Curtis integration in 1D"""
    assert u.ndim == 1
    N = u.shape[0]
    if quad == 'GL':
        w = np.arange(0, N, 1, dtype=float)
        w[2:] = 2./(1-w[2:]**2)
        w[0] = 1
        w[1::2] = 0
        ak = dct(u, 1)
        ak /= (N-1)
        return np.sqrt(np.sum(ak*w))

    assert quad == 'GC'
    d = np.zeros(N)
    k = 2*(1 + np.arange((N-1)//2))
    d[::2] = (2./N)/np.hstack((1., 1.-k*k))
    w = dct(d, type=3)
    return np.sqrt(np.sum(u*w))

class CachedArrayDict(MutableMapping):
    """Dictionary for caching Numpy arrays (work arrays)

    Example
    -------

    >>> import numpy as np
    >>> from shenfun.utilities import CachedArrayDict
    >>> work = CachedArrayDict()
    >>> a = np.ones((3, 4), dtype=int)
    >>> w = work[(a, 0, True)] # create work array with shape as a
    >>> print(w.shape)
    (3, 4)
    >>> print(w)
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]
    >>> w2 = work[(a, 1, True)] # Get different(note 1!) array of same shape/dtype
    """
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        newkey, fill = self.__keytransform__(key)
        try:
            value = self._data[newkey]
        except KeyError:
            shape, dtype, _ = newkey
            value = np.empty(shape, dtype=np.dtype(dtype, align=True))
            self._data[newkey] = value
        if fill:
            value.fill(0)
        return value

    @staticmethod
    def __keytransform__(key):
        assert len(key) == 3
        return (key[0].shape, key[0].dtype, key[1]), key[2]

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        self._data[self.__keytransform__(key)[0]] = value

    def __delitem__(self, key):
        del self._data[self.__keytransform__(key)[0]]

    def __iter__(self):
        return iter(self._data)

    def values(self):
        raise TypeError('Cached work arrays not iterable')

def outer(a, b, c):
    r"""Return outer product $c_{i,j} = a_i b_j$

    Parameters
    ----------
    a : Array of shape (N, ...)
    b : Array of shape (N, ...)
    c : Array of shape (N*N, ...)

    The outer product is taken over the first index of a and b,
    for all remaining indices.
    """
    av = a.v
    bv = b.v
    cv = c.v
    symmetric = a is b
    if av.shape[0] == 2:
        outer2D(av, bv, cv, symmetric)
    elif av.shape[0] == 3:
        outer3D(av, bv, cv, symmetric)
    return c

@optimizer
def outer2D(a, b, c, symmetric):
    c[0] = a[0]*b[0]
    c[1] = a[0]*b[1]
    if symmetric:
        c[2] = c[1]
    else:
        c[2] = a[1]*b[0]
    c[3] = a[1]*b[1]

@optimizer
def outer3D(a, b, c, symmetric):
    c[0] = a[0]*b[0]
    c[1] = a[0]*b[1]
    c[2] = a[0]*b[2]
    if symmetric:
        c[3] = c[1]
        c[6] = c[2]
        c[7] = c[5]
    else:
        c[3] = a[1]*b[0]
        c[6] = a[2]*b[0]
        c[7] = a[2]*b[1]
    c[4] = a[1]*b[1]
    c[5] = a[1]*b[2]
    c[8] = a[2]*b[2]

@optimizer
def apply_mask(u_hat, mask):
    if mask is not None:
        u_hat *= mask
    return u_hat

def integrate_sympy(f, d):
    """Exact definite integral using sympy

    Try to convert expression `f` to a polynomial before integrating.

    See sympy issue https://github.com/sympy/sympy/pull/18613 to why this is
    needed. Poly().integrate() is much faster than sympy.integrate() when applicable.

    Parameters
    ----------
    f : sympy expression
    d : 3-tuple
        First item the symbol, next two the lower and upper integration limits
    """
    try:
        p = sp.Poly(f, d[0]).integrate()
        return p(d[2]) - p(d[1])
    except sp.PolynomialError:
        #return sp.Integral(f, d).evalf()
        return sp.integrate(f, d).evalf()

def split(measures):

    def _split(ms):
        ms = sp.sympify(ms)
        if isinstance(ms, Number):
            d = {'coeff': ms}
        elif len(ms.free_symbols) == 0:
            d = {'coeff': ms.evalf()}
        else:
            d = sp.separatevars(ms.factor(), dict=True)
        if d is None:
            raise RuntimeError('Could not split', ms)
        d = defaultdict(lambda: 1, {str(k): v for k, v in d.items()})
        dc = d['coeff']
        d['coeff'] = int(dc) if isinstance(dc, (sp.Integer, int)) else float(dc)
        return d

    ms = sp.sympify(measures)
    if len(ms.free_symbols) <= 1:
        return [_split(ms)]

    ms = ms.expand()
    result = []
    if isinstance(ms, sp.Add) or (isinstance(ms, sp.Mul) and np.all([isinstance(s, sp.Add) for s in ms.args])):
        for arg in ms.args:
            result.append(_split(arg))
    else:
        result.append(_split(ms))
    return result

def oldsplit(measures):

    def _split(mss, result):
        for ms in mss:
            ms = sp.sympify(ms)
            if isinstance(ms, sp.Mul):
                # Multiplication of two or more terms
                result = _split(ms.args, result)
                continue

            # Something else with only one symbol
            sym = ms.free_symbols
            assert len(sym) <= 1
            if len(sym) == 1:
                sym = sym.pop()
                result[str(sym)] *= ms
            else:
                ms = int(ms) if isinstance(ms, sp.Integer) else float(ms)
                result['coeff'] *= ms
        return result

    ms = sp.sympify(measures).expand()
    result = []
    if isinstance(ms, sp.Add):
        for arg in ms.args:
            result.append(_split([arg], defaultdict(lambda: 1)))
    else:
        result.append(_split([ms], defaultdict(lambda: 1)))
    return result

def mayavi_show():
    """
    Return show function that updates the mayavi figure in the background.
    """
    from pyface.api import GUI
    from mayavi import mlab
    return mlab.show(GUI().stop_event_loop)

def wrap_periodic(xs, axes=()):
    """Return arrays wrapped around periodically

    xs : array or sequence of arrays
    axes : sequence of integers, optional
        Extend arrays in xs by one in direction given by wrap
    """
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    ys = []
    for x in xs:
        if 0 in axes:
            x = np.vstack([x, x[0]])
        if 1 in axes:
            x = np.hstack([x, x[:, 0][:, None]])
        if 2 in axes:
            x = np.concatenate([x, x[:, :, 0][:, :, None]], axis=2)
        ys.append(x)
    if len(ys) == 1:
        return ys[0]
    return ys

def surf3D(u, backend='plotly', wrapaxes=(), slices=None, fig=None, kind='normal', **kw):
    """Plot surface embedded in 3D

    Parameters
    ----------
    u : Function or Array
    backend : str, optional
        plotly or mayavi
    wrapaxes : sequence of integers, optional
        For domains that wrap around, extend mesh/u by one in given direction
    slices : None or sequence of slices, optional
        If only part of u should be plotted
    fig : Figure instance, optional
    kind : str, optional
        normal or uniform
    kw : Keyword arguments, optional
        Used by plotly Surface. Possibly colorscale.
    """
    from shenfun.forms.arguments import Function
    T = u.function_space()
    uniform = True if kind == 'uniform' else False
    x, y, z = T.local_cartesian_mesh(uniform=uniform)
    if isinstance(u, Function):
        u = u.backward(kind=kind)

    x, y, z, u = wrap_periodic([x, y, z, u], wrapaxes)

    if slices:
        x = x[slices]
        y = y[slices]
        z = z[slices]
        u = u[slices]

    if u.dtype.char in 'FDG':
        u = abs(u)**2

    if backend == 'plotly':
        import plotly.graph_objects as go
        s = go.Surface(x=x, y=y, z=z, surfacecolor=u, **kw)
        fig = go.Figure(s)
        d = {'visible': False, 'showgrid': False, 'zeroline': False}
        fig.update_layout(scene={'xaxis': d, 'yaxis': d, 'zaxis': d})

    elif backend == 'mayavi':
        from mayavi import mlab
        fig = mlab.figure(bgcolor=(1, 1, 1))
        mlab.mesh(x, y, z, scalars=u, colormap='jet')

    return fig
