from numbers import Number, Integral
import copy
from scipy.special import sph_harm, erf, airy
import numpy as np
import sympy as sp
from shenfun.optimization.cython import evaluate
from mpi4py_fft import DistArray

__all__ = ('Expr', 'BasisFunction', 'TestFunction', 'TrialFunction', 'Function',
           'Array', 'FunctionSpace', 'Basis')

# Define some special functions required for spherical harmonics
cot = lambda x: 1/np.tan(x)
Ynm = lambda n, m, x, y : sph_harm(m, n, y, x)
airyai = lambda x: airy(x)[0]
printwarning = True

def Basis(*args, **kwargs): #pragma: no cover
    global printwarning
    import warnings
    if printwarning:
        warnings.warn("Basis() is deprecated; use FunctionSpace().", FutureWarning)
        printwarning = False
    return FunctionSpace(*args, **kwargs)

def FunctionSpace(N, family='Fourier', bc=None, dtype='d', quad=None,
                  domain=None, scaled=None, padding_factor=1, basis=None,
                  dealias_direct=False, coordinates=None, **kw):
    """Return function space

    Parameters
    ----------

    N : int
        Number of quadrature points
    family : str, optional
        Choose one of

        - ``Chebyshev`` or ``C``,
        - ``Legendre`` or ``L``,
        - ``Fourier`` or ``F``,
        - ``Laguerre`` or ``La``,
        - ``Hermite`` or ``H``
        - ``Jacobi`` or ``J``

    bc : tuple or dict, optional
        Choose one of

        * 2-tuple of numbers (a, b) - Dirichlet boundary condition with
          :math:`v(-1)=a` and :math:`v(1)=b`.

        * (None, a) - Dirichlet on right boundary, nothing on left.

        * 4-tuple (a, b, c, d) - Biharmonic with the two non-zero Dirichlet
          conditions first :math:`v(-1)=a` and :math:`v(1)=b` and then
          the two Neumann.

        * dict with keys 'left' and 'right', for left and right boundaries,
          and a list of 2-tuples to specify the condition. This is the most
          general form, and all boundary conditions may be specified like this.
          Specify Dirichlet on both ends with

              {'left': [('D', a)], 'right': [('D', b)]}

          Specify mixed Neumann and Dirichlet as

              {'left': [('N', a)], 'right': [('D', b)]}

          For both conditions on the right do

              {'right': [('N', a), ('D', b)]}

          Note that not all combinations are possible for biharmonic
          problems. One combination that is possible is a fixed free beam
          with

              {'left': [('D', a), ('N', b)], 'right': [('N2', c), ('N3', d)]}

          where 'N2' and 'N3' represent second and third derivatives.

    dtype : str or np.dtype, optional
        The datatype of physical space (input to forward transforms)
    quad : str, optional
        Type of quadrature

        * For family=Chebyshev:

          - GL - Chebyshev-Gauss-Lobatto
          - GC - Chebyshev-Gauss

        * For family=Legendre:

          - LG - Legendre-Gauss
          - GL - Legendre-Gauss-Lobatto
        * For family=Laguerre:

          - LG - Laguerre-Gauss
        * For family=Hermite:

          - HG - Hermite-Gauss
        * For family=Jacobi:

          - JG - Jacobi-Gauss
    domain : two-tuple of floats, optional
        The computational domain
    scaled : bool
        Whether to use scaled basis (only Legendre)
    basis : str
        Name of basis to use, if there are more than one possible basis for a given
        boundary condition. For example, there are two Dirichlet bases for the
        Chebyshev family: 'Heinricht' and 'ShenDirichlet'
    padding_factor : float, optional
        For padding backward transform (for dealiasing)
    dealias_direct : bool, optional
        Use 2/3-rule dealiasing
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem.
        The new coordinate variable in the new coordinate system is the first item.
        Second item is a tuple for the Cartesian position vector as function of the
        new variable in the first tuple. Example::

            theta = sp.Symbols('x', real=True, positive=True)
            rv = (sp.cos(theta), sp.sin(theta))

        where theta and rv are the first and second items in the 2-tuple.

    Examples
    --------
    >>> from shenfun import FunctionSpace
    >>> F0 = FunctionSpace(16, 'F')
    >>> C1 = FunctionSpace(32, 'C', quad='GC')

    """
    par = {'padding_factor': padding_factor,
           'dealias_direct': dealias_direct,
           'dtype': dtype,
           'coordinates': coordinates}
    par.update(kw)
    if domain is not None:
        par['domain'] = domain

    def _process_bcs(bc, domain):

        bcs = {'left': [], 'right': []}
        df = 1
        if domain is not None:
            df = 2./(domain[1]-domain[0])
        if isinstance(bc, tuple):
            # Short form for Dirichlet or biharmonic with 2 or 4 numbers
            assert len(bc) in (2, 4)
            assert np.all([isinstance(i, (sp.Expr, Number)) or i is None for i in bc])
            if len(bc) == 2:
                if bc[0] is not None:
                    bcs['left'] = [('D', bc[0])]
                if bc[1] is not None:
                    bcs['right'] = [('D', bc[1])]

            if len(bc) == 4:
                bcs['left'] = [('D', bc[0]), ('N', bc[2])]
                bcs['right'] = [('D', bc[1]), ('N', bc[3])]
            key = ['L'+bci[0] for bci in bcs['left']] + ['R'+bci[0] for bci in bcs['right']]

        elif isinstance(bc, dict):
            bcs = {k.lower(): list(v) if isinstance(v[0], (tuple, list)) else [v] for k, v in bc.items()}
            bc = []
            key = []
            if 'left' in bcs:
                bcs['left'].sort()
                for bci in bcs['left']:
                    if bci[0] == 'N':
                        bc.append(bci[1]/df)
                    elif bci[0] == 'N2':
                        bc.append(bci[1]/df**2)
                    elif bci[0] == 'N3':
                        bc.append(bci[1]/df**3)
                    else:
                        bc.append(bci[1])
                key += ['L'+bci[0] for bci in bcs['left']]
            if 'right' in bcs:
                bcs['right'].sort()
                for bci in bcs['right']:
                    if bci[0] == 'N':
                        bc.append(bci[1]/df)
                    elif bci[0] == 'N2':
                        bc.append(bci[1]/df**2)
                    elif bci[0] == 'N3':
                        bc.append(bci[1]/df**3)
                    else:
                        bc.append(bci[1])
                key += ['R'+bci[0] for bci in bcs['right']]

        return key, tuple(bc)

    if family.lower() in ('fourier', 'f'):
        from shenfun import fourier
        if np.dtype(dtype).char in 'FDG':
            B = fourier.bases.C2C
        else:
            B = fourier.bases.R2C
        del par['dtype']
        return B(N, **par)

    elif family.lower() in ('chebyshev', 'c'):
        from shenfun import chebyshev
        if quad is not None:
            assert quad in ('GC', 'GL', 'GU')
            par['quad'] = quad

        if scaled is not None:
            assert isinstance(scaled, bool)
            par['scaled'] = scaled

        # Boundary conditions abbreviated in dictionary keys as
        #   left->L, right->R, Dirichlet->D, Neumann->N
        # So LDRD means left Dirichlet, right Dirichlet
        bases = {
            '': chebyshev.bases.Orthogonal,
            'LDRD': chebyshev.bases.ShenDirichlet,
            'LNRN': chebyshev.bases.ShenNeumann,
            'LDRN': chebyshev.bases.DirichletNeumann,
            'LNRD': chebyshev.bases.NeumannDirichlet,
            'RD': chebyshev.bases.UpperDirichlet,
            'RDRN': chebyshev.bases.UpperDirichletNeumann,
            'LDLNRDRN': chebyshev.bases.ShenBiharmonic
        }

        if isinstance(bc, (tuple, dict)):
            key, par['bc'] = _process_bcs(bc, domain)

        elif bc is None:
            key = ''

        else:
            raise NotImplementedError

        if basis is not None:
            assert isinstance(basis, str)
            B = getattr(chebyshev.bases, basis)
        else:
            B = bases[''.join(key)]

        return B(N, **par)

    elif family.lower() in ('legendre', 'l'):
        from shenfun import legendre

        bases = {
            '': legendre.bases.Orthogonal,
            'LDRD': legendre.bases.ShenDirichlet,
            'LNRN': legendre.bases.ShenNeumann,
            'LDRN': legendre.bases.DirichletNeumann,
            'LNRD': legendre.bases.NeumannDirichlet,
            'LD': legendre.bases.LowerDirichlet,
            'RD': legendre.bases.UpperDirichlet,
            'RDRN': legendre.bases.UpperDirichletNeumann,
            'LNRDRN': legendre.bases.ShenBiPolar0,
            'LDLNRDRN': legendre.bases.ShenBiharmonic,
            'LDLNRN2RN3': legendre.bases.BeamFixedFree
        }

        if quad is not None:
            assert quad in ('LG', 'GL')
            par['quad'] = quad

        if scaled is not None:
            assert isinstance(scaled, bool)
            par['scaled'] = scaled

        if isinstance(bc, (tuple, dict)):
            key, par['bc'] = _process_bcs(bc, domain)

        elif bc is None:
            key = ''

        else:
            raise NotImplementedError

        if basis is not None:
            assert isinstance(basis, str)
            B = getattr(legendre.bases, basis)
        else:
            B = bases[''.join(key)]

        return B(N, **par)

    elif family.lower() in ('laguerre', 'la'):
        from shenfun import laguerre
        if quad is not None:
            assert quad in ('LG', 'GR')
            par['quad'] = quad

        if bc is None:
            B = laguerre.bases.Orthogonal

        elif isinstance(bc, tuple):
            assert len(bc) == 2
            par['bc'] = bc
            B = laguerre.bases.ShenDirichlet

        elif isinstance(bc, str):
            if bc.lower() == 'dirichlet':
                B = laguerre.bases.ShenDirichlet

        else:
            raise NotImplementedError

        return B(N, **par)

    elif family.lower() in ('hermite', 'h'):
        from shenfun import hermite
        if quad is not None:
            assert quad in ('HG',)
            par['quad'] = quad

        B = hermite.bases.Orthogonal

        if isinstance(bc, tuple):
            assert len(bc) == 2
            par['bc'] = bc

        elif isinstance(bc, str):
            assert bc.lower() == 'dirichlet'

        else:
            assert bc is None

        return B(N, **par)

    elif family.lower() in ('jacobi', 'j'):
        from shenfun import jacobi
        if quad is not None:
            assert quad in ('JG',)
            par['quad'] = quad

        if bc is None:
            B = jacobi.bases.Orthogonal

        elif isinstance(bc, tuple):
            assert len(bc) in (2, 4)
            par['bc'] = bc
            if len(bc) == 2:
                B = jacobi.bases.ShenDirichlet
            else:
                assert np.all([abs(bci)<1e-12 for bci in bc])
                B = jacobi.bases.ShenBiharmonic

        elif isinstance(bc, str):
            if bc.lower() == 'dirichlet':
                B = jacobi.bases.ShenDirichlet
            elif bc.lower() == 'biharmonic':
                B = jacobi.bases.ShenBiharmonic
            elif bc.lower() == '6th order':
                B = jacobi.bases.ShenOrder6
            else:
                raise NotImplementedError

        return B(N, **par)

    else:
        raise NotImplementedError


class Expr:
    r"""
    Class for spectral Galerkin forms

    An Expression that is linear in :class:`.TestFunction`,
    :class:`.TrialFunction` or :class:`.Function`. Exprs are used as input
    to :func:`.inner` or :func:`.project`.

    Parameters
    ----------
    basis : :class:`.BasisFunction`
        :class:`.TestFunction`, :class:`.TrialFunction` or :class:`.Function`
    terms : list of list of lists of length dimension
        Describes the differential operations performed on the basis function
        in the `Expr`

        The triply nested `terms` lists are such that

        - the outermost list represents a tensor component. There is one item for
          each tensor component. If the Expr is a scalar with rank = 0, then
          len(terms) = 1. For vectors it equals the number of dimensions and
          for second order tensors it equals ndim**2

        - the second nested list represents the different terms in the form, that
          may be more than one. For example, the scalar valued `div(grad(u))` has
          three terms in 3D:

        .. math::

           \partial^2u/\partial x^2 + \partial^2u/\partial y^2 + \partial^2u/\partial z^2

        - the last inner list represents the differential operations for each term
          and each tensor component, stored for each as a list of length = dim

        The Expr `div(grad(u))`, where u is a scalar, is as such represented
        as a nested list of shapes (1, 3, 3), 1 meaning it's a scalar, the first 3
        because the Expr consists of the sum of three terms, and the last 3
        because it is 3D. The entire representation is::

            [[[2, 0, 0],
              [0, 2, 0],
              [0, 0, 2]]]

        where the first [2, 0, 0] term has two derivatives in first direction
        and none in the others, the second [0, 2, 0] has two derivatives in
        second direction, and the last [0, 0, 2] has two derivatives in the
        last direction and none in the others.

    scales :  list of lists
        Representing a scalar multiply of each inner product. Note that
        the scalar can also be a function of coordinates (using sympy).
        There is one scale for each term in each tensor component, and as
        such it is a list of lists.

    indices : list of lists
        Index into :class:`.CompositeSpace`. Only used when the basis of the
        form is composite. There is one index for each term in each tensor component,
        and as such it is a list of lists.

    Examples
    --------
    >>> from shenfun import *
    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD
    >>> C0 = FunctionSpace(16, 'F', dtype='D')
    >>> C1 = FunctionSpace(16, 'F', dtype='D')
    >>> R0 = FunctionSpace(16, 'F', dtype='d')
    >>> T = TensorProductSpace(comm, (C0, C1, R0))
    >>> v = TestFunction(T)
    >>> e = div(grad(v))
    >>> e.terms()
    [[[2, 0, 0], [0, 2, 0], [0, 0, 2]]]
    >>> e2 = grad(v)
    >>> e2.terms()
    [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]]

    Note that `e2` in the example has shape (3, 1, 3). The first 3 because it
    is a vector, the 1 because each vector item contains one term, and the
    final 3 since it is a 3-dimensional tensor product space.
    """

    def __init__(self, basis, terms=None, scales=None, indices=None):
        self._basis = basis
        self._terms = terms
        self._scales = scales
        self._indices = indices
        ndim = self.function_space().dimensions
        if terms is None:
            self._terms = np.zeros((self.function_space().num_components(), 1, ndim),
                                   dtype=int).tolist()
        if scales is None:
            self._scales = np.ones((self.function_space().num_components(), 1), dtype=object).tolist()

        if indices is None:
            self._indices = (basis.offset()+np.arange(self.function_space().num_components())[:, np.newaxis]).tolist()

        #assert np.prod(self._scales.shape) == self.num_terms()*self.num_components()

    def basis(self):
        """Return basis of Expr"""
        return self._basis

    @property
    def base(self):
        """Return base BasisFunction used in Expr"""
        return self._basis if self._basis.base is None else self._basis.base

    def function_space(self):
        """Return function space of basis in Expr"""
        return self._basis.function_space()

    def terms(self):
        """Return terms of Expr"""
        return self._terms

    def scales(self):
        """Return scales of Expr"""
        return self._scales

    @property
    def argument(self):
        """Return argument of Expr's basis"""
        return self._basis.argument

    def expr_rank(self):
        """Return rank of Expr"""
        if self.dimensions == 1:
            assert len(self._terms) < 3
            return len(self._terms)-1

        if len(self._terms) == 1:
            return 0
        if len(self._terms) == len(self._terms[0][0]):
            return 1
        if len(self._terms) == len(self._terms[0][0])**2:
            return 2

    @property
    def tensor_rank(self):
        """Return rank of Expr's :class:`BasisFunction`"""
        return self._basis.tensor_rank

    def indices(self):
        """Return indices of Expr"""
        return self._indices

    def num_components(self):
        """Return number of components in Expr"""
        return len(self._terms)

    def num_terms(self):
        """Return number of terms in Expr"""
        return [len(terms) for terms in self._terms]

    @property
    def dimensions(self):
        """Return ndim of Expr"""
        return len(self._terms[0][0])

    def index(self):
        if self.num_components() == 1:
            return self._basis.offset()
        return None

    def tolatex(self, symbol_names=None, funcname='u', replace=None):
        s = ""
        x = 'xyzrst'
        symbols = {k: k for k in x}
        if symbol_names is not None:
            symbols = {str(k): val for k, val in symbol_names.items()}

        for i, vec in enumerate(self.terms()):
            if self.num_terms()[i] == 0 and self.num_components() > 1:
                s += '0 \\mathbf{b}_{%s} \\\\ +'%(symbols[x[i]])
                continue

            if self.num_components() > 1:
                s += '\\left( '

            for j, term in enumerate(vec):
                sc = self.scales()[i][j]
                k = self.indices()[i][j]
                if not sc == 1:
                    if replace is not None:
                        for repl in replace:
                            assert len(repl) == 2
                            sc = sc.replace(*repl)
                        #sc = sp.simplify(sc)
                    scp = sp.latex(sc, symbol_names=symbol_names)

                    if isinstance(sc, sp.Add):
                        scp = "\\left( %s \\right)"%(scp)
                    elif scp.startswith('-'):
                        s = s.rstrip('+')
                    s += scp
                t = np.array(term)
                cmp = funcname
                #if self.num_components() > 1:

                if self.tensor_rank > 0:
                    cmp = funcname + '^{%s}'%(symbols[x[k]])
                if np.sum(t) == 0:
                    s += cmp
                else:
                    p = '^'+str(np.sum(t)) if np.sum(t) > 1 else ' '
                    s += "\\frac{\\partial%s %s}{"%(p, cmp)
                    for j, ti in enumerate(t):
                        if ti > 0:
                            tt = '^'+str(ti) if ti > 1 else ' '
                            s += "\\partial %s%s "%(symbols[x[j]], tt)
                    s += "}"
                s += '+'
            if self.num_components() > 1:
                s = s.rstrip('+')
                s += "\\right) \\mathbf{b}_{%s} \\\\"%(symbols[x[i]])
                s += '+'

        return r"""%s"""%(s.rstrip('+'))

    def tosympy(self, basis=None, psi=sp.symbols('x,y,z,r,s', real=True)):
        """Return self evaluated with a sympy basis

        Parameters
        ----------
        basis : sympy Expr or string
            if sympy Expr, then use this directly
            if str, then use that as name for a generic sympy Function
        psi : tuple of sympy Symbols

        Examples
        --------
        >>> from shenfun import FunctionSpace, TensorProductSpace
        >>> import sympy as sp
        >>> theta, r = psi = sp.symbols('x,y', real=True, positive=True)
        >>> rv = (r*sp.cos(theta), r*sp.sin(theta))
        >>> F0 = FunctionSpace(8, 'F', dtype='d')
        >>> T0 = FunctionSpace(8, 'C')
        >>> T = TensorProductSpace(comm, (F0, T0), coordinates=(psi, rv))
        >>> u = TrialFunction(T)
        >>> ue = r*sp.cos(theta)
        >>> du = div(grad(u))
        >>> du.tosympy()
        Derivative(u(x, y), (y, 2)) + Derivative(u(x, y), y)/y + Derivative(u(x, y), (x, 2))/y**2
        >>> du.tosympy(basis=ue, psi=psi)
        0

        """

        ndim = self.dimensions
        if basis is None:
            basis = 'u'

        if self.expr_rank() > 0:
            b = self.function_space().coors.get_covariant_basis()

        if isinstance(basis, str):
            # Create sympy Function
            st = basis
            if self.tensor_rank == 1:
                basis = []
                for i in range(self.dimensions):
                    basis.append(sp.Function(st+'xyzrs'[i])(*psi[:self.dimensions]))
            elif self.tensor_rank == 2:
                raise NotImplementedError
            else:
                basis = [sp.Function(st)(*psi[:self.dimensions])]
        else:
            if isinstance(basis, sp.Expr):
                basis = [basis]
            assert len(basis) == ndim**self.tensor_rank

        u = []
        for i, vec in enumerate(self.terms()):
            s = sp.S(0)
            for j, term in enumerate(vec):
                sc = self.scales()[i][j]
                k = self.indices()[i][j]
                b0 = basis[k]
                tt = tuple([psi[n] for n, l in enumerate(term) for m in range(l)])
                bi = 1
                if np.sum(term) > 0:
                    ss = sc*bi*sp.diff(b0, *tt)
                    s += ss

                else:
                    ss = sc*b0*bi
                    s += ss
            u.append(s)

        if len(u) == 1:
            return u[0]
        return u

    def eval(self, x, output_array=None):
        """Return expression evaluated on x

        Parameters
        ----------
        x : float or array of floats
            Array must be of shape (D, N), for  N points in D dimensions

        """
        from shenfun import CompositeSpace
        from shenfun.fourier.bases import R2C

        if len(x.shape) == 1: # 1D case
            x = x[None, :]

        V = self.function_space()
        basis = self.basis()

        if output_array is None:
            output_array = np.zeros(x.shape[1], dtype=V.forward.input_array.dtype)
        else:
            output_array[:] = 0

        work = np.zeros_like(output_array)

        assert V.dimensions == len(x)

        for vec, (base, ind) in enumerate(zip(self.terms(), self.indices())):
            for base_j, b0 in enumerate(base):
                M = []
                test_sp = V
                if isinstance(V, CompositeSpace):
                    test_sp = V.flatten()[ind[base_j]]
                r2c = -1
                last_conj_index = -1
                sl = -1
                for axis, k in enumerate(b0):
                    xx = test_sp[axis].map_reference_domain(np.squeeze(x[axis]))
                    P = test_sp[axis].evaluate_basis_derivative_all(xx, k=k)
                    if not test_sp[axis].domain_factor() == 1:
                        P *= test_sp[axis].domain_factor()**(k)
                    if len(x) > 1:
                        M.append(P[..., V.local_slice()[axis]])

                    if isinstance(test_sp[axis], R2C) and len(x) > 1:
                        r2c = axis
                        m = test_sp[axis].N//2+1
                        if test_sp[axis].N % 2 == 0:
                            last_conj_index = m-1
                        else:
                            last_conj_index = m
                        sl = V.local_slice()[axis].start

                bv = basis if basis.tensor_rank == 0 else basis[ind[base_j]]
                work.fill(0)
                if len(x) == 1:
                    work = np.dot(P, bv)

                elif len(x) == 2:
                    work = evaluate.evaluate_2D(work, bv, M, r2c, last_conj_index, sl)

                elif len(x) == 3:
                    work = evaluate.evaluate_3D(work, bv, M, r2c, last_conj_index, sl)

                sc = self.scales()[vec][base_j]
                if not hasattr(sc, 'free_symbols'):
                    sc = float(sc)
                else:
                    sym0 = tuple(sc.free_symbols)
                    m = []
                    for sym in sym0:
                        j = 'xyzrs'.index(str(sym))
                        m.append(x[j])
                    sc = sp.lambdify(sym0, sc)(*m)
                output_array += sc*work

        return output_array

    def __getitem__(self, i):
        if i >= self.dimensions:
            raise IndexError

        basis = self._basis
        if basis.function_space().is_composite_space > 0:
            basis = self._basis[i]
        else:
            basis = self._basis
        if self.expr_rank() == 1:
            return Expr(basis,
                        [copy.deepcopy(self._terms[i])],
                        [copy.deepcopy(self._scales[i])],
                        [copy.deepcopy(self._indices[i])])

        elif self.expr_rank() == 2:
            ndim = self.dimensions
            return Expr(basis,
                        copy.deepcopy(self._terms[i*ndim:(i+1)*ndim]),
                        copy.deepcopy(self._scales[i*ndim:(i+1)*ndim]),
                        copy.deepcopy(self._indices[i*ndim:(i+1)*ndim]))
        else:
            raise NotImplementedError

    def __mul__(self, a):
        sc = np.array(self.scales())
        if self.expr_rank() == 0:
            sc = sc*sp.sympify(a)

        else:
            if isinstance(a, tuple):
                assert len(a) == self.num_components()
                for i in range(self.num_components()):
                    sc[i] = sc[i]*sp.sympify(a[i])

            else:
                sc = sc*sp.sympify(a)

        for i in range(sc.shape[0]):
            for j in range(sc.shape[1]):
                sc[i, j] = self.function_space().coors.refine(sc[i, j])
                sc[i, j] = sp.simplify(sc[i, j], measure=self.function_space().coors._measure)

        return Expr(self._basis, copy.deepcopy(self._terms), sc.tolist(), copy.deepcopy(self._indices))

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        sc = np.array(self.scales())
        if self.expr_rank() == 0:
            sc = sc*sp.sympify(a)

        else:
            if isinstance(a, tuple):
                assert len(a) == self.dimensions
                for i in range(self.dimensions):
                    sc[i] = sc[i]*sp.sympify(a[i])

            else:
                sc = sc*sp.sympify(a)

        coors = self.function_space().coors
        for i in range(sc.shape[0]):
            for j in range(sc.shape[1]):
                sc[i, j] = coors.refine(sc[i, j])
                sc[i, j] = sp.simplify(sc[i, j], measure=coors._measure)

        self._scales = sc.tolist()

        return self

    def __add__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        #assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        if id(self._basis) == id(a._basis):
            basis = self._basis
        else:
            assert id(self._basis.base) == id(a._basis.base)
            basis = self._basis.base

        # Concatenate terms
        terms, scales, indices = [], [], []
        for i in range(self.num_components()):
            terms.append(self._terms[i] + a.terms()[i])
            scales.append(self._scales[i] + a.scales()[i])
            indices.append(self._indices[i] + a.indices()[i])

        return Expr(basis, terms, scales, indices)

    def __iadd__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        assert self.argument == a.argument
        self._basis = self.base
        for i in range(self.num_components()):
            self._terms[i] += a.terms()[i]
            self._scales[i] += a.scales()[i]
            self._indices[i] += a.indices()[i]
        return self

    def __sub__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        #assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        if id(self._basis) == id(a._basis):
            basis = self._basis
        else:
            assert id(self._basis.base) == id(a._basis.base)
            basis = self._basis.base

        # Concatenate terms
        terms, scales, indices = [], [], []
        for i in range(self.num_components()):
            terms.append(self._terms[i] + a.terms()[i])
            scales.append(self._scales[i] + (-np.array(a.scales()[i])).tolist())
            indices.append(self._indices[i] + a.indices()[i])

        return Expr(basis, terms, scales, indices)

    def __isub__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        if not isinstance(a, Expr):
            a = Expr(a)
        assert self.num_components() == a.num_components()
        assert self.function_space() == a.function_space()
        assert self.argument == a.argument
        if id(self._basis) == id(a._basis):
            basis = self._basis
        else:
            assert id(self._basis.base) == id(a._basis.base)
            basis = self._basis.base
        self._basis = basis
        for i in range(self.num_components()):
            self._terms[i] += a.terms()[i]
            self._scales[i] += (-np.array(a.scales()[i])).tolist()
            self._indices[i] += a.indices()[i]

        return self

    def __neg__(self):
        return Expr(self.basis(), copy.deepcopy(self.terms()), (-np.array(self.scales())).tolist(),
                    copy.deepcopy(self.indices()))

    def simplify(self):
        """Join terms, that are otherwise equal, using scale"""
        if np.all(np.array(self.num_terms()) == 1):
            return

        tms = []
        inds = []
        scs = []
        for (terms, indices, scales) in zip(self.terms(), self.indices(), self.scales()):
            tms.append([])
            inds.append([])
            scs.append([])
            for i, (term, ind, sc) in enumerate(zip(terms, indices, scales)):
                match = []
                if len(tms[-1]) > 0:
                    match = np.where((np.array(term) == np.array(tms[-1])).prod(axis=1) == 1)[0]
                    matchi = np.where( ind == np.array(inds[-1]))[0]
                    match = np.intersect1d(match, matchi)
                if len(match) > 0:
                    k = match[0]
                    assert inds[-1][k] == ind
                    scs[-1][k] += sc
                    scs[-1][k] = sp.simplify(scs[-1][k], measure=self.function_space().coors._measure)
                    scs[-1][k] = self.function_space().coors.refine(scs[-1][k])
                    if scs[-1][k] == 0: # Remove if scale is zero
                        scs[-1].pop(k)
                        tms[-1].pop(k)
                        inds[-1].pop(k)
                    continue
                sc = sp.simplify(sc, measure=self.function_space().coors._measure)
                sc = self.function_space().coors.refine(sc)
                if not sc == 0:
                    tms[-1].append(term)
                    inds[-1].append(ind)
                    scs[-1].append(sc)
        self._terms = tms
        self._indices = inds
        self._scales = scs
        return

    def __eq__(self, a):
        if self.base != a.base:
            return False
        if self.num_components() != a.num_components():
            return False
        if not np.all(self.num_terms() == a.num_terms()):
            return False
        for s0, q0 in zip(self.scales(), a.scales()):
            for si, qi in zip(s0, q0):
                if si != qi:
                    return False
        for s0, q0 in zip(self.indices(), a.indices()):
            for si, qi in zip(s0, q0):
                if si != qi:
                    return False
        for t0, p0 in zip(self.terms(), a.terms()):
            for ti, pi in zip(t0, p0):
                if not np.all(ti == pi):
                    return False
        return True

    def subs(self, a, b):
        """Replace `a` with `b` in scales

        Parameters
        ----------
        a : Sympy Symbol
        b : Number
        """
        scs = self.scales()
        for i, sci in enumerate(scs):
            for j, scj in enumerate(sci):
                scs[i][j] = scj.subs(a, b)


class BasisFunction:
    """Base class for arguments to shenfun's Exprs

    Parameters
    ----------
    space : :class:`.TensorProductSpace`, :class:`.CompositeSpace` or
        :class:`.SpectralBase`
    index : int
        Local component of basis in :class:`.CompositeSpace`
    basespace : The base :class:`.CompositeSpace` if space is a
        subspace.
    offset : int
        The number of scalar spaces (i.e., :class:`.TensorProductSpace`es)
        ahead of this space
    base : The base :class:`BasisFunction`
    """

    def __init__(self, space, index=0, basespace=None, offset=0, base=None):
        self._space = space
        self._index = index
        self._basespace = basespace
        self._offset = offset
        self._base = base

    @property
    def tensor_rank(self):
        """Return tensor rank of basis"""
        return self.function_space().tensor_rank

    def expr_rank(self):
        """Return rank of basis"""
        return self.tensor_rank

    def function_space(self):
        """Return function space of BasisFunction"""
        return self._space

    @property
    def basespace(self):
        """Return base space"""
        return self._basespace if self._basespace is not None else self._space

    @property
    def base(self):
        """Return base """
        return self._base if self._base is not None else self

    @property
    def argument(self):
        """Return argument of basis"""
        raise NotImplementedError

    def num_components(self):
        """Return number of components in basis"""
        return self.function_space().num_components()

    @property
    def dimensions(self):
        """Return dimensions of function space"""
        return self.function_space().dimensions

    def index(self):
        """Return index into base space"""
        return self._offset + self._index

    def offset(self):
        """Return offset of this basis

        The offset is the number of scalar :class:`.TensorProductSpace`es ahead
        of this space in a :class:`.CompositeSpace`.
        """
        return self._offset

    def __getitem__(self, i):
        assert self.function_space().is_composite_space
        basespace = self.basespace
        base = self.base
        space = self._space[i]
        offset = self._offset
        for k in range(i):
            offset += self._space[k].num_components()
        t0 = self.__class__(space, 0, basespace, offset, base)
        return t0

    def __mul__(self, a):
        b = Expr(self)
        return b*a

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        raise RuntimeError

    def __add__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        b = Expr(self)
        return b+a

    def __iadd__(self, a):
        raise RuntimeError

    def __sub__(self, a):
        assert isinstance(a, (Expr, BasisFunction))
        b = Expr(self)
        return b-a

    def __isub__(self, a):
        raise RuntimeError


class TestFunction(BasisFunction):
    """Test function - BasisFunction with argument = 0

    Parameters
    ----------
    space: :class:`TensorProductSpace` or :class:`CompositeSpace`
    index: int, optional
        Component of basis in :class:`.CompositeSpace`
    basespace : The base :class:`.CompositeSpace` if space is a
        subspace.
    offset : int
        The number of scalar spaces (i.e., :class:`.TensorProductSpace`es)
        ahead of this space
    base : The base :class:`TestFunction`
    """

    def __init__(self, space, index=0, basespace=None, offset=0, base=None):
        BasisFunction.__init__(self, space, index, basespace, offset, base)

    @property
    def argument(self):
        return 0

class TrialFunction(BasisFunction):
    """Trial function - BasisFunction with argument = 1

    Parameters
    ----------
    space: :class:`TensorProductSpace` or :class:`CompositeSpace`
    index: int, optional
        Component of basis in :class:`.CompositeSpace`
    basespace : The base :class:`.CompositeSpace` if space is a
        subspace.
    offset : int
        The number of scalar spaces (i.e., :class:`.TensorProductSpace`es)
        ahead of this space
    base : The base :class:`TrialFunction`
    """
    def __init__(self, space, index=0, basespace=None, offset=0, base=None):
        BasisFunction.__init__(self, space, index, basespace, offset, base)

    @property
    def argument(self):
        return 1

class ShenfunBaseArray(DistArray):

    def __new__(cls, space, val=0, buffer=None, reltol=1e-12, abstol=1e-15):

        if hasattr(space, 'points_and_weights'): # 1D case
            if space.N == 0:
                space = space.get_adaptive(fun=buffer, reltol=reltol, abstol=abstol)

            if cls.__name__ == 'Function':
                dtype = space.forward.output_array.dtype
                shape = space.forward.output_array.shape
            elif cls.__name__ == 'Array':
                dtype = space.forward.input_array.dtype
                shape = space.forward.input_array.shape

            if not space.num_components() == 1:
                shape = (space.num_components(),) + shape

            # if a list of sympy expressions
            if isinstance(buffer, (list, tuple)):
                assert len(buffer) == len(space.flatten())
                sympy_buffer = buffer
                buffer = Array(space)
                dtype = space.forward.input_array.dtype
                for i, buf0 in enumerate(sympy_buffer):
                    if isinstance(buf0, Number):
                        buffer.v[i] = buf0
                    elif hasattr(buf0, 'free_symbols'):
                        x = buf0.free_symbols.pop()
                        buffer.v[i] = sp.lambdify(x, buf0)(space.mesh()).astype(dtype)

                if cls.__name__ == 'Function':
                    buf = Function(space)
                    buf = buffer.forward(buf)
                    buffer = buf

            elif hasattr(buffer, 'free_symbols'):
                # Evaluate sympy function on entire mesh
                x = buffer.free_symbols.pop()
                buffer = sp.lambdify(x, buffer)
                buf = buffer(space.mesh()).astype(space.forward.input_array.dtype)
                buffer = Array(space)
                buffer.v[:] = buf
                if cls.__name__ == 'Function':
                    buf = Function(space)
                    buf = buffer.forward(buf)
                    buffer = buf
            val0 = val if isinstance(val, Number) else None
            obj = DistArray.__new__(cls, shape, buffer=buffer, dtype=dtype,
                                    val=val0, rank=space.is_composite_space)
            obj._space = space
            obj._offset = 0
            if buffer is None and isinstance(val, (list, tuple)):
                assert len(val) == len(obj)
                for i, v in enumerate(val):
                    obj.v[i] = v

            return obj

        if min(space.global_shape()) == 0:
            space = space.get_adaptive(fun=buffer, reltol=reltol, abstol=abstol)

        if cls.__name__ == 'Function':
            forward_output = True
            p0 = space.forward.output_pencil
            dtype = space.forward.output_array.dtype
        elif cls.__name__ == 'Array':
            forward_output = False
            p0 = space.backward.output_pencil
            dtype = space.forward.input_array.dtype

        # if a list of sympy expressions
        if isinstance(buffer, (list, tuple)):
            assert len(buffer) == len(space.flatten())
            sympy_buffer = buffer
            buffer = Array(space)
            dtype = space.forward.input_array.dtype
            if cls.__name__ == 'Function':
                buff = Function(space)
            mesh = space.local_mesh(True)
            for i, buf0 in enumerate(sympy_buffer):
                if isinstance(buf0, Number):
                    buffer.v[i] = buf0
                elif hasattr(buf0, 'free_symbols'):
                    sym0 = tuple(buf0.free_symbols)
                    m = []
                    for sym in sym0:
                        j = 'xyzrs'.index(str(sym))
                        m.append(mesh[j])
                    buffer.v[i] = sp.lambdify(sym0, buf0, modules=['numpy', {'airyai': airyai, 'cot': cot, 'Ynm': Ynm, 'erf': erf}])(*m).astype(dtype)
                else:
                    raise NotImplementedError

            if cls.__name__ == 'Function':
                buff = Function(space)
                buff = buffer.forward(buff)
                buffer = buff

        # if just one sympy expression
        if hasattr(buffer, 'free_symbols'):
            sym0 = tuple(buffer.free_symbols)
            mesh = space.local_mesh(True)
            m = []
            for sym in sym0:
                j = 'xyzrs'.index(str(sym))
                m.append(mesh[j])
            buf = sp.lambdify(sym0, buffer, modules=['numpy', {'airyai': airyai, 'cot': cot, 'Ynm': Ynm, 'erf': erf}])(*m).astype(space.forward.input_array.dtype)
            buffer = Array(space)
            buffer.v[:] = buf
            if cls.__name__ == 'Function':
                buf = Function(space)
                buf = buffer.forward(buf)
                buffer = buf

        global_shape = space.global_shape(forward_output)
        val0 = val if isinstance(val, Number) else None
        obj = DistArray.__new__(cls, global_shape,
                                subcomm=p0.subcomm, val=val0, dtype=dtype,
                                buffer=buffer, alignment=p0.axis,
                                rank=space.is_composite_space)
        obj._space = space
        obj._offset = 0
        obj._rank = space.is_composite_space
        if isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                obj.v[i] = v
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._space = getattr(obj, '_space', None)
        self._p0 = getattr(obj, '_p0', None)
        self._rank = getattr(obj, '_rank', None)
        self._offset = getattr(obj, '_offset', None)

    def function_space(self):
        """Return function space of array ``self``"""
        return self._space

    def index(self):
        """Return index for scalar into mixed base space"""
        if self.base is None:
            return None

        if self.function_space().num_components() > 1:
            return None

        data_self = self.ctypes.data
        data_base = self.base.ctypes.data
        itemsize = self.itemsize
        return (data_self - data_base) // (itemsize*np.prod(self.shape))

    @property
    def argument(self):
        """Return argument of basis"""
        return 2

    @property
    def global_shape(self):
        """Return global shape of ``self``"""
        return self.function_space().global_shape(self.forward_output)

    @property
    def forward_output(self):
        """Return whether ``self`` is the result of a forward transform"""
        raise NotImplementedError

    def __getitem__(self, i):
        if self.ndim == 1:
            return np.ndarray.__getitem__(self, i)

        if self._space.is_composite_space and isinstance(i, Integral):
            # Return view into composite Function
            space = self._space[i]
            offset = 0
            for j in range(i):
                offset += self._space[j].num_components()
            ns = space.num_components()
            s = slice(offset, offset+ns) if ns > 1 else offset
            v0 = np.ndarray.__getitem__(self, s)
            v0._space = space
            v0._offset = offset + self.offset()
            v0._rank = space.is_composite_space
            return v0

        return np.ndarray.__getitem__(self.v, i)

    def dim(self):
        return self.function_space().dim()

    def dims(self):
        return self.function_space().dims()

    @property
    def tensor_rank(self):
        return self.function_space().tensor_rank


class Function(ShenfunBaseArray, BasisFunction):
    r"""
    Spectral Galerkin function for given :class:`.TensorProductSpace` or :func:`.Basis`

    The Function is the product of all 1D basis expansions, that for each
    dimension is defined like

    .. math::

        u(x) = \sum_{k \in \mathcal{K}} \hat{u}_k \psi_k(x),

    where :math:`\psi_k(x)` are the trial functions and
    :math:`\{\hat{u}_k\}_{k\in\mathcal{K}}` are the expansion coefficients.
    Here an index set :math:`\mathcal{K}=0, 1, \ldots, N` is used
    to simplify notation.

    For an M+1-dimensional TensorProductSpace with coordinates
    :math:`x_0, x_1, \ldots, x_M` we get

    .. math::

        u(x_{0}, x_{1}, \ldots, x_{M}) = \sum_{k_0 \in \mathcal{K}_0}\sum_{k_1 \in \mathcal{K}_1} \ldots \sum_{k_M \in \mathcal{K}_M} \hat{u}_{k_0, k_1, \ldots k_M} \psi_{k_0}(x_0) \psi_{k_1}(x_1) \ldots \psi_{k_M}(x_M),

    where :math:`\mathcal{K}_j` is the index set for the wavenumber mesh
    along axis :math:`j`.

    Note that for a Cartesian mesh in 3D it would be natural to use coordinates
    :math:`(x, y, z) = (x_0, x_1, x_2)` and the expansion would be the
    simpler and somewhat more intuitive

    .. math::

        u(x, y, z) = \sum_{l \in \mathcal{K}_0}\sum_{m \in \mathcal{K}_1} \sum_{n \in \mathcal{K}_2} \hat{u}_{l, m, n} \psi_{l}(x) \psi_{m}(y) \psi_{n}(z).

    The Function's values (the Numpy array) represent the :math:`\hat{u}` array.
    The trial functions :math:`\psi` may differ in the different directions.
    They are chosen when creating the TensorProductSpace.

    Parameters
    ----------
    space : :class:`.TensorProductSpace` or :class:`.FunctionSpace`
    val : int or float
        Value used to initialize array
    buffer : Numpy array, :class:`.Function` or sympy `Expr`
        If array it must be of correct shape.
        A sympy expression is evaluated on the quadrature mesh and
        forward transformed to create the buffer array.


    .. note:: For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_

    Example
    -------
    >>> from shenfun import comm, FunctionSpace, TensorProductSpace, Function
    >>> K0 = FunctionSpace(8, 'F', dtype='D')
    >>> K1 = FunctionSpace(8, 'F', dtype='d')
    >>> T = TensorProductSpace(comm, [K0, K1])
    >>> u = Function(T)
    >>> K2 = FunctionSpace(8, 'C', bc=(0, 0))
    >>> T2 = TensorProductSpace(comm, [K0, K1, K2])
    >>> v = Function(T2)

    You get the array the space is planned for. This is probably what you
    expect:

    >>> u = Function(K0)
    >>> print(u.shape)
    (8,)

    But make the TensorProductSpace change K0 and K1 inplace:

    >>> T = TensorProductSpace(comm, [K0, K1], modify_spaces_inplace=True)
    >>> u = Function(K0)
    >>> print(u.shape)
    (8, 5)

    If you want to preserve K0 as a 1D function space, then use
    `modify_spaces_inplace=False`, which is the default behaviour.

    Note
    ----
    The array returned will have the same shape as the arrays
    `space` is planned for. So if you want a Function on a 1D
    FunctionSpace, then make sure that FunctionSpace is not planned
    for a TensorProductSpace.

    """
    # pylint: disable=too-few-public-methods,too-many-arguments

    def __init__(self, space, val=0, buffer=None, reltol=1e-12, abstol=1e-15):
        BasisFunction.__init__(self, self._space, offset=0)

    def set_boundary_dofs(self):
        space = self.function_space()
        if space.is_composite_space:
            for i, s in enumerate(space.flatten()):
                bases = s.bases if hasattr(s, 'bases') else [s]
                for base in bases:
                    if base.has_nonhomogeneous_bcs:
                        base.bc.set_boundary_dofs(self.v[i], True)

        else:
            bases = space.bases if hasattr(space, 'bases') else [space]
            for base in bases:
                if base.bc:
                    base.bc.set_boundary_dofs(self, True)
        return self

    @property
    def forward_output(self):
        return True

    def __call__(self, x, output_array=None):
        return self.eval(x, output_array=output_array)

    def eval(self, x, output_array=None):
        """Evaluate Function at points

        Parameters
        ----------
        points : float or array of floats
        coefficients : array
            Expansion coefficients
        output_array : array, optional
            Return array, function values at points

        Examples
        --------
        >>> import sympy as sp
        >>> K0 = FunctionSpace(9, 'F', dtype='D')
        >>> K1 = FunctionSpace(8, 'F', dtype='d')
        >>> T = TensorProductSpace(MPI.COMM_WORLD, [K0, K1], axes=(0, 1))
        >>> X = T.local_mesh()
        >>> x, y = sp.symbols("x,y")
        >>> ue = sp.sin(2*x) + sp.cos(3*y)
        >>> ul = sp.lambdify((x, y), ue, 'numpy')
        >>> ua = Array(T, buffer=ul(*X))
        >>> points = np.random.random((2, 4))
        >>> u = ua.forward()
        >>> u0 = u.eval(points).real
        >>> assert np.allclose(u0, ul(*points))
        """
        return self.function_space().eval(x, self, output_array)

    def backward(self, output_array=None, kind='normal'):
        """Return Function evaluated on some quadrature mesh

        Parameters
        ----------
        output_array : array, optional
            Return array, Function evaluated on mesh
        kind : str or function space
            There are three kinds of backward transforms

            - 'normal' - evaluate Function on its quadrature mesh
            - 'uniform' - evaluate Function on uniform mesh
            - function space - evaluate Function on the quadrature mesh of the
              given function space. This could be a :func:`.FunctionSpace`
              if 1D, or a :class:`.TensorProductSpace` if multidimensional.

        """
        space = self.function_space()
        if hasattr(kind, 'mesh'):
            if output_array is None:
                output_array = Array(kind)
            output_array = space.backward(self, output_array, kind=kind)
            return output_array
        assert isinstance(kind, str)
        if output_array is None:
            output_array = Array(space)
        if kind.lower() == 'uniform':
            output_array = space.backward(self, output_array, kind='uniform')
        elif kind.lower() == 'normal':
            output_array = space.backward(self, output_array)
        return output_array

    def to_ortho(self, output_array=None):
        """Project Function to orthogonal basis"""
        space = self.function_space()
        if output_array is None:
            output_array = Function(space.get_orthogonal())

        # In case of mixed space make a loop
        if space.num_components() > 1:
            for x, subfunction in zip(output_array, self):
                x = subfunction.to_ortho(x)
            return output_array

        if space.dimensions > 1:
            naxes = space.get_nonperiodic_axes()
            axis = naxes[0]
            base = space.bases[axis]
            if not base.is_orthogonal:
                output_array = base.to_ortho(self, output_array)
            else:
                output_array[:] = self

            if len(naxes) > 1:
                input_array = np.zeros_like(output_array.v)
                for axis in naxes[1:]:
                    base = space.bases[axis]
                    input_array[:] = output_array
                    if not base.is_orthogonal:
                        output_array = base.to_ortho(input_array, output_array)
        else:
            output_array = space.to_ortho(self, output_array)
        return output_array

    def mask_nyquist(self, mask=None):
        """Set self to have zeros in Nyquist coefficients"""
        self.function_space().mask_nyquist(self, mask=mask)

    def assign(self, u_hat):
        """Assign self to u_hat of possibly different size

        Parameters
        ----------
        u_hat : Function
            Function of possibly different shape than self. Must have
            the same function_space
        """
        from shenfun import VectorSpace
        if self.ndim == 1:
            assert u_hat.__class__ == self.__class__
            if self.shape[0] < u_hat.shape[0]:
                self.function_space()._padding_backward(self, u_hat)
            elif self.shape[0] == u_hat.shape[0]:
                u_hat[:] = self
            elif self.shape[0] > u_hat.shape[0]:
                self.function_space()._truncation_forward(self, u_hat)
            return u_hat

        space = self.function_space()
        newspace = u_hat.function_space()

        if isinstance(space, VectorSpace):
            for i, self_i in enumerate(self):
                u_hat[i] = self_i.assign(u_hat[i])
            return u_hat

        same_bases = True
        for base0, base1 in zip(space.bases, newspace.bases):
            if not base0.__class__ == base1.__class__:
                same_bases = False
                break
        assert same_bases, "Can only assign on spaces with the same underlying bases"

        N = []
        for newbase in newspace.bases:
            N.append(newbase.N)

        u_hat = self.refine(N, output_array=u_hat)
        return u_hat

    def refine(self, N, output_array=None):
        """Return self with new number of quadrature points

        Parameters
        ----------
        N : number or sequence of numbers
            The new number of quadrature points

        Note
        ----
        If N is smaller than for self, then a truncated array
        is returned. If N is greater than before, then the
        returned array is padded with zeros.

        """
        from shenfun.fourier.bases import R2C
        from shenfun import VectorSpace

        if self.ndim == 1:
            assert isinstance(N, Number)
            space = self.function_space()
            if output_array is None:
                refined_basis = space.get_refined(N)
                output_array = Function(refined_basis)
            output_array = self.assign(output_array)
            return output_array

        if isinstance(N, Number):
            N = (N,)*len(self)
        elif isinstance(N, (tuple, list, np.ndarray)):
            assert len(N) == len(self.function_space())

        space = self.function_space()

        if isinstance(space, VectorSpace):
            if output_array is None:
                output_array = [None]*len(self)
            for i, array in enumerate(self):
                output_array[i] = array.refine(N, output_array=output_array[i])
            if isinstance(output_array, list):
                T = output_array[0].function_space()
                VT = VectorSpace(T)
                output_array = np.array(output_array)
                output_array = Function(VT, buffer=output_array)
            return output_array

        axes = [bx for ax in space.axes for bx in ax]
        base = space.bases[axes[0]]
        global_shape = list(self.global_shape) # Global shape in spectral space
        factor = N[axes[0]]/self.function_space().bases[axes[0]].N
        if isinstance(base, R2C):
            global_shape[axes[0]] = int((2*global_shape[axes[0]]-2)*factor)//2+1
        else:
            global_shape[axes[0]] = int(global_shape[axes[0]]*factor)
        c1 = DistArray(global_shape,
                       subcomm=self.pencil.subcomm,
                       dtype=self.dtype,
                       alignment=self.alignment)
        if self.global_shape[axes[0]] <= global_shape[axes[0]]:
            base._padding_backward(self, c1)
        else:
            base._truncation_forward(self, c1)
        for ax in axes[1:]:
            c0 = c1.redistribute(ax)
            factor = N[ax]/self.function_space().bases[ax].N

            # Get a new padded array
            base = space.bases[ax]
            if isinstance(base, R2C):
                global_shape[ax] = int(base.N*factor)//2+1
            else:
                global_shape[ax] = int(global_shape[ax]*factor)
            c1 = DistArray(global_shape,
                           subcomm=c0.pencil.subcomm,
                           dtype=c0.dtype,
                           alignment=ax)

            # Copy from c0 to d0
            if self.global_shape[ax] <= global_shape[ax]:
                base._padding_backward(c0, c1)
            else:
                base._truncation_forward(c0, c1)

        # Reverse transfer to get the same distribution as u_hat
        for ax in reversed(axes[:-1]):
            c1 = c1.redistribute(ax)

        if output_array is None:
            refined_space = space.get_refined(N)
            output_array = Function(refined_space, buffer=c1)
        else:
            output_array[:] = c1
        return output_array


class Array(ShenfunBaseArray):
    r"""
    Numpy array for :class:`.TensorProductSpace`

    The Array is the result of a :class:`.Function` evaluated on its quadrature
    mesh.

    The Function is the product of all 1D basis expansions, that for each
    dimension is defined like

    .. math::

        u(x) = \sum_{k \in \mathcal{K}} \hat{u}_k \psi_k(x),

    where :math:`\psi_k(x)` are the trial functions and
    :math:`\{\hat{u}_k\}_{k\in\mathcal{K}}` are the expansion coefficients.
    Here an index set :math:`\mathcal{K}=0, 1, \ldots, N` is used to
    simplify notation.

    For an M+1-dimensional TensorProductSpace with coordinates
    :math:`x_0, x_1, \ldots, x_M` we get

    .. math::

        u(x_{0}, x_{1}, \ldots, x_{M}) = \sum_{k_0 \in \mathcal{K}_0}\sum_{k_1 \in \mathcal{K}_1} \ldots \sum_{k_M \in \mathcal{K}_M} \hat{u}_{k_0, k_1, \ldots k_M} \psi_{k_0}(x_0) \psi_{k_1}(x_1) \ldots \psi_{k_M}(x_M),

    where :math:`\mathcal{K}_j` is the index set for the wavenumber mesh
    along axis :math:`j`.

    Note that for a Cartesian mesh in 3D it would be natural to use coordinates
    :math:`(x, y, z) = (x_0, x_1, x_2)` and the expansion would be the
    simpler and somewhat more intuitive

    .. math::

        u(x, y, z) = \sum_{l \in \mathcal{K}_0}\sum_{m \in \mathcal{K}_1} \sum_{n \in \mathcal{K}_2} \hat{u}_{l, m, n} \psi_{l}(x) \psi_{m}(y) \psi_{n}(z).

    The Array's values (the Numpy array) represent the left hand side,
    evaluated on the Cartesian quadrature mesh. With this we mean the
    :math:`u(x_i, y_j, z_k)` array, where :math:`\{x_i\}_{i=0}^{N_0}`,
    :math:`\{y_j\}_{j=0}^{N_1}` and :math:`\{z_k\}_{k=0}^{N_2}` represent
    the mesh along the three directions. The quadrature mesh is then

    .. math::

        (x_i, y_j, z_k) \quad \forall \, (i, j, k) \in [0, 1, \ldots, N_0] \times [0, 1, \ldots, N_1] \times [0, 1, \ldots, N_2]

    The entire spectral Galerkin function can be obtained using the
    :class:`.Function` class.

    Parameters
    ----------

    space : :class:`.TensorProductSpace` or :class:`.SpectralBase`
    val : int or float
        Value used to initialize array
    buffer : Numpy array, :class:`.Function` or sympy `Expr`
        If array it must be of correct shape.
        A sympy expression is evaluated on the quadrature mesh and
        the result is used as buffer.

    .. note:: For more information, see `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_

    Examples
    --------
    >>> from shenfun import comm, FunctionSpace, TensorProductSpace, Array
    >>> K0 = FunctionSpace(8, 'F', dtype='D')
    >>> K1 = FunctionSpace(8, 'F', dtype='d')
    >>> FFT = TensorProductSpace(comm, [K0, K1])
    >>> u = Array(FFT)
    """

    @property
    def forward_output(self):
        return False

    def forward(self, output_array=None):
        """Return Function used to evaluate Array"""
        space = self.function_space()
        if output_array is None:
            output_array = Function(space)
        output_array = space.forward(self, output_array)
        return output_array

    def offset(self):
        """Return offset of this basis

        The offset is the number of scalar :class:`.TensorProductSpace`es ahead
        of this Arrays space in a :class:`.CompositeSpace`.
        """
        return self._offset

