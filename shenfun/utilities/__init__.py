"""
Module for implementing helper functions.
"""
from numbers import Number
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
from collections import defaultdict
import numpy as np
import sympy as sp
from scipy.fftpack import dct
from scipy.integrate import quad
from shenfun.optimization import runtimeoptimizer, cython
from shenfun.config import config
from .findbasis import get_bc_basis, get_stencil_matrix, n

__all__ = ['dx', 'clenshaw_curtis1D', 'CachedArrayDict', 'surf3D',
           'wrap_periodic', 'outer', 'dot', 'apply_mask', 'integrate_sympy',
           'mayavi_show', 'quiver3D', 'get_bc_basis', 'get_stencil_matrix',
           'scalar_product', 'n', 'cross', 'reset_profile', 'Lambda']

Lambda = getattr(cython, 'Lambda', None)

def dx(u, weighted=False):
    r"""Compute integral of u over domain

    .. math::

        \int_{\Omega} u dx

    Parameters
    ----------

    u : Array
        The Array to integrate

    Note
    ----
    This function assumes a standard reference domain. If the domain
    is not standard, then modify the result accordingly.

    """
    T = u.function_space()
    dim = len(u.shape)
    if dim == 1:
        w = T.points_and_weights(weighted=weighted)[1]
        return np.sum(u*w).item()

    uc = u.copy()
    for ax in range(dim):
        uc = uc.redistribute(axis=ax)
        w = T.bases[ax].points_and_weights(weighted=weighted)[1]
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

def reset_profile(prof):
    """Reset profiler for kernprof

    Parameters
    ----------
    prof : The profiler
    """
    prof.code_map = {}
    prof.last_time = {}
    prof.enable_count = 0
    for func in prof.functions:
        prof.add_function(func)

def cross(c, a, b):
    """Cross product c = a x b

    Parameters
    ----------
    c : Array
    a : Array
    b : Array

    Returns
    -------
    c : Array
    """
    if a.ndim == 3:
        cross2D(c, a, b)
    elif a.ndim == 4:
        cross3D(c, a, b)
    else:
        crossND(c, a, b)
    return c

@runtimeoptimizer
def cross2D(c, a, b):
    c[:] = a[0]*b[1]-a[1]*b[0]

@runtimeoptimizer
def cross3D(c, a, b):
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]

@runtimeoptimizer
def crossND(c, a, b):
    cross3D(c, a, b)

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

@runtimeoptimizer
def outer2D(a, b, c, symmetric):
    c[0] = a[0]*b[0]
    c[1] = a[0]*b[1]
    if symmetric:
        c[2] = c[1]
    else:
        c[2] = a[1]*b[0]
    c[3] = a[1]*b[1]

@runtimeoptimizer
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

def dot(u, v, output_array=None, forward_output=True):
    """Return dot product of u and v

    Parameters
    ----------
    u : Function or Array
    v : Function or Array
    output_array : Function or Array
        Must be of correct shape
    forward_output : bool, optional
        Return the result as a Function if True, and an Array if
        False.

    Note
    ----
    This function uses the 3/2-rule for dealiasing all spaces if
    the input arrays are spectral Functions. The returned Function
    is projected to an orthogonal space, without boundary conditions.

    """
    from shenfun import Function, Array
    assert isinstance(u, (Function, Array))
    assert isinstance(v, (Function, Array))
    Vu = u.function_space()
    Vv = v.function_space()
    assert Vu.dimensions == Vv.dimensions
    D = Vu.dimensions
    Vup = Vu
    Vvp = Vv

    # Get the correct output_array
    # Logic is to first follow the output_array if given,
    # use input_arrays if not
    uv = output_array
    if isinstance(output_array, Function):
        forward_output = True
        Top = output_array.function_space()
        uv = Array(Top)

    elif isinstance(output_array, Array):
        forward_output = False
        Top = output_array.function_space()
        uv = output_array

    else:
        output_rank = Vu.tensor_rank+Vv.tensor_rank-2
        if output_rank == 0:
            To = Vu[0].get_orthogonal()
        elif output_rank == 1:
            if Vu.tensor_rank == 1:
                To = Vu.get_orthogonal()
            else:
                To = Vu[0].get_orthogonal()
        elif output_rank == 2:
            To = Vu.get_orthogonal()

        Top = To.get_dealiased()
        uv = Array(Top)
    uv.fill(0)

    if Top.is_padded:
        Vup = Vu.get_dealiased()
        Vvp = Vv.get_dealiased()

    if isinstance(u, Array):
        ua = u

    else:
        ua = Array(Vup)
        ua = Vup.backward(u, ua)

    if isinstance(v, Array):
        va = v
    else:
        va = Array(Vvp)
        va = Vvp.backward(v, va)

    if Vu.coors.is_cartesian:

        if Vu.tensor_rank == 1 and Vv.tensor_rank == 1:
            uv = np.sum(ua.v*va.v, axis=0, out=uv)

        elif Vu.tensor_rank == 2 and Vv.tensor_rank == 1:
            for i in range(D):
                ui = ua[i]
                for j in range(D):
                    uv[i] += ui[j]*va[j]

        elif Vu.tensor_rank == 1 and Vv.tensor_rank == 2:
            for i in range(D):
                ui = ua[i]
                vi = va[i]
                for j in range(D):
                    uv[j] += ui*vi[j]

        elif Vu.tensor_rank == 2 and Vv.tensor_rank == 2:
            for i in range(D):
                ui = ua[i]
                wi = uv[i]
                for j in range(D):
                    vj = va[j]
                    for k in range(D):
                        wi[k] += ui[j]*vj[k]

        else:
            raise NotImplementedError

    else:
        gij = Vu.coors.get_metric_tensor(config['basisvectors'])
        mesh = Vup.local_mesh(True)
        def measure(g):
            sym0 = g.free_symbols
            m = []
            for sym in sym0:
                j = 'xyzrs'.index(str(sym))
                m.append(mesh[j])
            return sp.lambdify(tuple(sym0), g)(*m)

        if Vu.tensor_rank == 1 and Vv.tensor_rank == 1:
            if Vu.coors.is_orthogonal:
                for i in range(D):
                    uv += ua[i]*va[i]*measure(gij[i, i])
            else:
                for i in range(D):
                    ui = ua[i]
                    for j in range(D):
                        g = gij[i, j]
                        if not g == 0:
                            uv += ui*va[j]*measure(g)

        elif Vu.tensor_rank == 2 and Vv.tensor_rank == 1:
            for i in range(D):
                ui = ua[i]
                for j in range(D):
                    for k in range(D):
                        g = gij[j, k]
                        if not g == 0:
                            uv[i] += ui[j]*va[k]*measure(g)

        elif Vu.tensor_rank == 1 and Vv.tensor_rank == 2:
            for i in range(D):
                ui = ua[i]
                for j in range(D):
                    vj = va[j]
                    g = gij[i, j]
                    if not g == 0:
                        ms = measure(g)
                        for k in range(D):
                            uv[k] += ui*vj[k]*ms

        elif Vu.tensor_rank == 2 and Vv.tensor_rank == 2:
            for i in range(D):
                ui = ua[i]
                wi = uv[i]
                for j in range(D):
                    for k in range(D):
                        g = gij[j, k]
                        if not g == 0:
                            vk = va[k]
                            ms = measure(g)
                            for l in range(D):
                                wi[l] += ui[j]*vk[l]*ms

        else:
            raise NotImplementedError

    if forward_output is True:

        if isinstance(output_array, Function):
            output_array = uv.forward(output_array)
            return output_array

        uv_hat = Function(To)
        uv_hat = uv.forward(uv_hat)
        return uv_hat

    return uv


@runtimeoptimizer
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

def rsplit(measures):
    d = []
    def _split(ms):
        ms = sp.sympify(ms)
        if isinstance(ms, Number):
            d = {'coeff': ms}
        elif len(ms.free_symbols) == 0:
            d = {'coeff': ms.evalf()}
        else:
            d = sp.separatevars(ms.factor(), dict=True)
        if d is None:
            return d
        d = defaultdict(lambda: 1, {str(k): v for k, v in d.items()})
        dc = d['coeff']
        d['coeff'] = int(dc) if isinstance(dc, (sp.Integer, int)) else float(dc)
        return d

    ms = sp.sympify(measures)
    di = _split(ms)
    if di is not None:
        d += [di]
    else:
        ms = ms.expand()
        for arg in ms.args:
            d += split(arg)
    return d

def split(measures, expand=False):

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
        dc = sp.sympify(d['coeff'])
        if dc.is_integer:
            d['coeff'] = int(dc)
        elif dc.is_real:
            d['coeff'] = float(dc)
        elif dc.is_complex:
            d['coeff'] = complex(dc)
        else:
            raise RuntimeError
        return d

    ms = sp.sympify(measures)
    if not expand:
        try:
            di = _split(ms)
            if di is not None:
                return [di]
        except RuntimeError:
            pass

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

    Parameters
    ----------
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

def surf3D(u, mesh=None, backend='plotly', wrapaxes=(), slices=None, fig=None, **kw):
    """Plot surface embedded in 3D

    Parameters
    ----------
    u : Function or Array
    mesh : list, str or None, optional
        - list of Cartesian meshes [x, y, z]
        - 'quadrature' - use quadrature mesh
        - 'uniform' - use uniform mesh
    backend : str, optional
        plotly or mayavi
    wrapaxes : sequence of integers, optional
        For domains that wrap around, extend mesh/u by one in given direction
    slices : None or sequence of slices, optional
        If only part of u should be plotted
    fig : Figure instance, optional
    kw : Keyword arguments, optional
        Used by plotly Surface. Possibly colorscale.
    """
    from shenfun.forms.arguments import Function, Array

    if mesh is None:
        mesh = 'quadrature'

    if isinstance(mesh, str):
        assert isinstance(u, (Function, Array)), "u must be Function/Array if mesh is not given"
        T = u.function_space()
        x, y, z = T.local_cartesian_mesh(kind=mesh)
    else:
        x, y, z = mesh

    if isinstance(u, Function):
        u = u.backward(mesh=mesh)

    x, y, z, u = wrap_periodic([x, y, z, u], wrapaxes)

    if slices:
        x = x[slices]
        y = y[slices]
        z = z[slices]
        u = u[slices]

    if u.dtype.char in 'FDG':
        u = abs(u)

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

def quiver3D(u, mesh=None, wrapaxes=(), slices=None, fig=None, kind='quadrature', **kw):
    """

    Parameters
    ----------
    u : Array
    mesh : list of Cartesian meshes [x, y, z], optional
    backend : str, optional
        plotly or mayavi
    wrapaxes : sequence of integers, optional
        For domains that wrap around, extend mesh/u by one in given direction
    fig : None, optional
        If None create a new figure
    slices : None or sequence of slices, optional
        If only part of u should be plotted
    kind : str, optional
        - 'quadrature' - Use quadrature mesh
        - 'uniform' - Use uniform mesh
    kw : Keyword arguments, optional
        Arguments to mlab.quiver3d, for example 'scale_factor',
        'color' or ''mode

    """
    from shenfun.forms.arguments import Function, Array
    from mayavi import mlab

    par = {
        'scale_factor': 0.1,
        'color': (0, 0, 0),
        'mode': '2darrow',
    }
    par.update(kw)
    if mesh is None:
        assert isinstance(u, (Function, Array)), "u must be Function/Array if mesh is not given"
        T = u.function_space()
        x, y, z = T.local_cartesian_mesh(kind=kind)
    else:
        x, y, z = mesh

    if isinstance(u, Function):
        u = u.backward(mesh=kind)

    if u.dtype.char in 'FDG':
        u = abs(u)

    x, y, z = wrap_periodic([x, y, z], wrapaxes)
    u = wrap_periodic([u], np.array(wrapaxes)+1)
    if slices:
        x = x[slices]
        y = y[slices]
        z = z[slices]
        u = u[slices]
    if fig is None:
        mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
    mlab.quiver3d(x, y, z, u[0], u[1], u[2], **par)

def scalar_product(v, f, output_array=None, assemble='exact'):
    r"""Return scalar product

    .. math::

        (v, f)_w

    Parameters
    ----------
    v : :class:`.TestFunction`
    f : Sympy function
    output_array : :class:`.Function`
    assemble : str, optional

        - 'exact'
        - 'adaptive'

    Note
    ----
    The computed scalar product is not compensated for a non-standard domain size.

    >>> from shenfun import inner, FunctionSpace, TestFunction
    >>> import sympy as sp
    >>> C = FunctionSpace(4, 'C')
    >>> v = TestFunction(C)
    >>> x = sp.Symbol('x', real=True)
    >>> f = x**2
    >>> inner(v, f, assemble='exact')
    Function([1.57079633, 0.        , 0.78539816, 0.        ])

    """
    assert assemble in ('exact', 'adaptive')
    T = v.function_space()
    if output_array is None:
        from shenfun import Function
        output_array = Function(T)

    if T.is_composite_space:
        for vi, xi in zip(v, output_array):
            xi = scalar_product(vi, f, xi)
        return output_array

    x = sp.Symbol('x', real=True)
    cheb = T.family() == 'chebyshev'

    if not isinstance(f, Number):
        s = f.free_symbols
        assert len(s) == 1
        x = s.pop()
        f = T.map_expression_true_domain(f, x=x)
        if cheb:
            f = f.subs(x, sp.cos(x))

    if cheb:
        S = T.stencil_matrix().diags('csr')
        for i in range(T.slice().start, T.slice().stop):
            M = S.getrow(i)
            integrand = sp.S(0)
            for ind, d in zip(M.indices, M.data):
                integrand += d*sp.cos(ind*x)
            integrand = f*integrand
            if assemble == 'exact':
                output_array[i] = sp.integrate(integrand, (x, (0, sp.pi)))
            elif assemble == 'adaptive':
                if isinstance(integrand, Number):
                    output_array[i] = integrand*np.pi
                else:
                    output_array[i] = quad(sp.lambdify(x, integrand), 0, np.pi)[0]
    else:
        domain = T.reference_domain()
        f *= T.weight()
        for i in range(T.slice().start, T.slice().stop):
            integrand = f*sp.conjugate(T.basis_function(i, x=x))
            if assemble == 'exact':
                output_array[i] = sp.integrate(integrand, (x, (domain[0], domain[1])))
            elif assemble == 'adaptive':
                if len(integrand.free_symbols) == 0:
                    if cheb:
                        output_array[i] = integrand*np.pi
                    else:
                        output_array[i] = integrand*float(domain[1]-domain[0])
                else:
                    output_array[i] = quad(sp.lambdify(x, integrand), float(domain[0]), float(domain[1]))[0]
    if T.domain_factor() != 1:
        output_array /= float(T.domain_factor())
    return output_array
