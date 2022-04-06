import functools
import sympy as sp
import numpy as np
from shenfun.matrixbase import SpectralMatrix, extract_diagonal_matrix
from shenfun.utilities import split
from . import bases

J  = bases.Orthogonal
CB = bases.CompositeBase
CD = bases.CompactDirichlet
CN = bases.CompactNeumann
UD = bases.UpperDirichlet
LD = bases.LowerDirichlet
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4
BCG = bases.BCGeneric


class BJJmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(J_j, J_k)_w,

    :math:`J_k \in` :class:`.jacobi.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], J)
        assert isinstance(trial[0], J)
        from .recursions import h, n
        N = test[0].N
        k = np.arange(N, dtype=int)
        a = test[0].alpha
        b = test[0].beta
        hh = h(a, b, n, 0)
        d = {0: sp.lambdify(n, hh)(k)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BGBCGmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k` is a subclass of
    :class:`.jacobi.bases.CompositeBase`, the
    trial :math:`\psi_j \in` :class:`.jacobi.bases.BCGeneric`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], BCG)
        B = BJJmat((test[0].get_orthogonal(domain=(-1, 1)), 0), (trial[0].get_orthogonal(domain=(-1, 1)), 0))
        K = test[0].stencil_matrix()
        K.shape = (test[0].dim(), test[0].N)
        S = extract_diagonal_matrix(trial[0].stencil_matrix().T).diags('csr')
        q = sp.degree(measure)
        if measure != 1:
            from shenfun.jacobi.recursions import pmat, a
            assert sp.sympify(measure).is_polynomial()
            A = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = pmat(a, qi, test[0].alpha, test[0].beta, test[0].N, test[0].N)
                A = A + sc*Ax.diags('csr')
            A = K.diags('csr') * A.T * B.diags('csr') * S
        else:
            A = K.diags('csr') * B.diags('csr') * S

        M = B.shape[1]
        K.shape = (test[0].N, test[0].N)
        d = extract_diagonal_matrix(A, lowerband=M+q, upperband=M)
        SpectralMatrix.__init__(self, dict(d), test, trial, scale=scale, measure=measure)

class BGGmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, x^q \phi_k)_w,

    where the test and trial functions :math:`\phi_k` and :math:`\psi_j` are
    any subclasses of :class:`.jacobi.bases.CompositeBase` and :math:`q \ge 0`
    is an integer. Test and trial spaces have dimensions of M and N, respectively.

    Note
    ----
    Creating mass matrices this way is efficient in terms of memory since the
    mass matrix of the orthogonal basis is diagonal.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], CB)
        B = BJJmat((test[0].get_orthogonal(domain=(-1, 1)), 0), (trial[0].get_orthogonal(domain=(-1, 1)), 0))
        K = test[0].stencil_matrix()
        K.shape = (test[0].dim(), test[0].N)
        S = trial[0].stencil_matrix()
        S.shape = (trial[0].dim(), trial[0].N)
        q = sp.degree(measure)
        if measure != 1:
            from shenfun.jacobi.recursions import pmat, a
            assert sp.sympify(measure).is_polynomial()
            A = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = pmat(a, qi, test[0].alpha, test[0].beta, test[0].N, test[0].N)
                A = A + sc*Ax.diags('csr')
            A = K.diags('csr') * B.diags('csr') * A * S.diags('csr').T

        else:
            A = K.diags('csr') * B.diags('csr') * S.diags('csr').T

        K.shape = (test[0].N, test[0].N)
        S.shape = (trial[0].N, trial[0].N)
        ub = test[0].N-test[0].dim()+q
        lb = trial[0].N-trial[0].dim()+q
        d = extract_diagonal_matrix(A, lowerband=lb, upperband=ub)
        SpectralMatrix.__init__(self, dict(d), test, trial, scale=scale, measure=measure)

class PXGmat(SpectralMatrix):
    r"""Matrix :math:`D=(d_{ij}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        d_{ij}=(\partial^{k-l} \psi_j, x^q \phi_i)_w,

    where the test function :math:`\phi_i` is in one of :class:`.jacobi.bases.Phi1`,
    :class:`.jacobi.bases.Phi2`, :class:`.jacobi.bases.Phi3`, :class:`.jacobi.bases.Phi4`,
    the trial :math:`\psi_j` any class in :class:`.jacobi.bases`,
    The three parameters k, q and l are integers, and test and trial spaces
    have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], (P1, P2, P3, P4))
        assert test[0].quad == 'GC'
        from shenfun.jacobi.recursions import Lmat
        q = sp.degree(measure)
        k = (test[0].N-test[0].dim())//2
        l = k-trial[1]
        if q > 0 and test[0].domain != test[0].reference_domain():
            D = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = Lmat(k, qi, l, test[0].dim(), trial[0].N, test[0].alpha, test[0].beta)
                D = D + sc*Ax
        else:
            D = Lmat(k, q, l, test[0].dim(), trial[0].N, test[0].alpha, test[0].beta)

        if trial[0].is_orthogonal:
            D = extract_diagonal_matrix(D, lowerband=q-k+l, upperband=q+k+l)
        else:
            K = trial[0].stencil_matrix()
            K.shape = (trial[0].dim(), trial[0].N)
            keys = np.sort(np.array(list(K.keys())))
            lb, ub = -keys[0], keys[-1]
            D = extract_diagonal_matrix(D*K.diags('csr').T, lowerband=q-k+l+ub, upperband=q+k+l+lb)
            K.shape = (trial[0].N, trial[0].N)
        SpectralMatrix.__init__(self, dict(D), test, trial, scale=scale, measure=measure)

class PXBCGmat(SpectralMatrix):
    r"""Matrix :math:`D=(d_{ij}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        d_{ij}=(\partial^{k-l}\psi_j, x^q \phi_i)_w,

    where the test function :math:`\phi_i` is in one of :class:`.jacobi.bases.Phi1`,
    :class:`.jacobi.bases.Phi2`, :class:`.jacobi.bases.Phi3`, :class:`.jacobi.bases.Phi4`,
    trial :math:`\psi_j \in` :class:`.jacobi.bases.BCGeneric`.
    The three parameters k, q, l are integers and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], (P1, P2, P3, P4))
        assert isinstance(trial[0], BCG)
        from shenfun.jacobi.recursions import Lmat
        M = test[0].dim()
        N = trial[0].dim_ortho
        q = sp.degree(measure)
        k = (test[0].N-test[0].dim())//2
        l = k-trial[1]
        if q > 0 and test[0].domain != test[0].reference_domain():
            D = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = Lmat(k, qi, l, M, N, test[0].alpha, test[0].beta)
                D = D + sc*Ax
        else:
            D = Lmat(k, q, l, M, N, test[0].alpha, test[0].beta)

        K = trial[0].stencil_matrix()
        D = extract_diagonal_matrix(D*extract_diagonal_matrix(K).diags('csr').T, lowerband=N+q, upperband=N)
        SpectralMatrix.__init__(self, dict(D), test, trial, scale=scale, measure=measure)


class _Jacmatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)


class _JacMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        if key[0][1]+key[1][1] == 0 and sp.sympify(measure).is_polynomial():
            if key[1][0] == BCG:
                c = functools.partial(BGBCGmat, measure=measure)
            else:
                c = functools.partial(BGGmat, measure=measure)
        else:
            c = functools.partial(_Jacmatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        if key[0][0] in (P1, P2, P3, P4):
            if key[1][0] == BCG:
                k = ('PX', 1)
            else:
                k = ('PX', 0)
            if key[1][1] > int(key[0][0].short_name()[1]) or key[1][0] in (P1, P2, P3, P4):
                # If the number of derivatives is larger than 1 for P1, 2 for P2 etc,
                # then we need to use quadrature. But it should not be larger if you
                # have designed the scheme appropriately, so perhaps we should throw
                # a warning
                k = key
            matrix = functools.partial(dict.__getitem__(self, k),
                                       measure=measure)
        elif len(key) == 3:
            matrix = functools.partial(dict.__getitem__(self, key),
                                       measure=key[2])
        else:
            matrix = dict.__getitem__(self, key)
        #assert key[0][1] == 0, 'Test cannot be differentiated (weighted space)'
        return matrix

mat = _JacMatDict({
    ((J,  0), (J,  0)): BJJmat,
    ('PX', 0): PXGmat,
    ('PX', 1): PXBCGmat,
})
