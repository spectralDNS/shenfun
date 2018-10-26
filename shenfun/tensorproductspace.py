"""
Module for implementation of the TensorProductSpace class and related methods.
"""
from numbers import Number
import warnings
import sympy
import numpy as np
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import chebyshev, legendre
from shenfun.forms.arguments import Function, Array
from shenfun.optimization import evaluate
from mpi4py_fft.mpifft import Transform, PFFT
from mpi4py_fft.pencil import Subcomm, Pencil

__all__ = ('TensorProductSpace', 'VectorTensorProductSpace',
           'MixedTensorProductSpace', 'Convolve')


class TensorProductSpace(PFFT):
    """Class for multidimensional tensorproductspaces.

    The tensorproductspaces are created as Cartesian products from a set of 1D
    bases. The 1D bases are subclassed instances of the :class:`.SpectralBase`
    class.

    Parameters
    ----------
        comm : MPI communicator
        bases : list
            List of 1D bases
        axes : tuple of ints, optional
            A tuple containing the order of which to perform transforms.
            Last item is transformed first. Defaults to range(len(bases))
        dtype : data-type, optional
            Type of input data in real physical space. If not provided it
            will be inferred from the bases.
        slab : bool, optional
            Use 1D slab decomposition.
        collapse_fourier : bool, optional
            Collapse axes for Fourier bases if possible
        kw : dict, optional
            Dictionary that can be used to plan transforms. Input to method
            `plan` for the bases.

    """
    def __init__(self, comm, bases, axes=None, dtype=None, slab=False,
                 collapse_fourier=False, **kw):
        # Note do not call __init__ of super
        self.comm = comm
        self.bases = bases
        shape = self.shape()
        assert shape
        assert min(shape) > 0

        if axes is not None:
            axes = list(axes) if np.ndim(axes) else [axes]
            for i, axis in enumerate(axes):
                if axis < 0:
                    axes[i] = axis + len(shape)
        else:
            axes = list(range(len(shape)))
        assert min(axes) >= 0
        assert max(axes) < len(shape)
        assert 0 < len(axes) <= len(shape)
        assert sorted(axes) == sorted(set(axes))

        if dtype is None:
            dtype = np.complex if isinstance(bases[axes[-1]], C2CBasis) else np.float
        else:
            if isinstance(bases[axes[-1]], C2CBasis):
                assert np.dtype(dtype).char in 'FDG'
            elif isinstance(bases[axes[-1]], R2CBasis):
                assert np.dtype(dtype).char in 'fdg'

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'

        if isinstance(comm, Subcomm):
            assert slab is False
            assert len(comm) == len(shape)
            assert comm[axes[-1]].Get_size() == 1
            self.subcomm = comm
        else:
            if slab:
                dims = [1] * len(shape)
                dims[axes[0]] = comm.Get_size()
            else:
                dims = [0] * len(shape)
                dims[axes[-1]] = 1
            self.subcomm = Subcomm(comm, dims)

        self.axes = tuple((axis,) for axis in axes)
        self.xfftn = []
        self.transfer = []
        self.pencil = [None, None]

        # At this points axes is a tuple of tuples of length one.
        # Try to collapse some Fourier transforms into one.
        if np.any([abs(base.padding_factor - 1.0) > 1e-6 for base in bases]):
            collapse_fourier = False
        if collapse_fourier:
            F = lambda ax: bases[ax].family() == 'fourier' and self.subcomm[ax].Get_size() == 1
            axis = self.axes[-1][-1]
            groups = [list(self.axes[-1])]
            F0 = F(axis)
            for ax in reversed(self.axes[:-1]):
                axis = ax[-1]
                if F0 and F(axis):
                    groups[0].insert(0, axis)
                else:
                    groups.insert(0, list(ax))
                F0 = F(axis)
            self.axes = groups
        self.axes = tuple(map(tuple, self.axes))

        # Configure all transforms
        axes = self.axes[-1]
        pencil = Pencil(self.subcomm, shape, axes[-1])
        self.xfftn.append(self.bases[axes[-1]])
        self.xfftn[-1].plan(pencil.subshape, axes, dtype, kw)
        self.pencil[0] = pencilA = pencil
        if not shape[axes[-1]] == self.xfftn[-1].forward.output_array.shape[axes[-1]]:
            dtype = self.xfftn[-1].forward.output_array.dtype
            shape[axes[-1]] = self.xfftn[-1].forward.output_array.shape[axes[-1]]
            pencilA = Pencil(self.subcomm, shape, axes[-1])

        for i, axes in enumerate(reversed(self.axes[:-1])):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = self.bases[axes[-1]]
            xfftn.plan(pencilB.subshape, axes, dtype, kw)
            self.xfftn.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB
            if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
                pencilA = Pencil(pencilB.subcomm, shape, axes[-1])

        self.pencil[1] = pencilA

        self.forward = Transform(
            [o.forward for o in self.xfftn],
            [o.forward for o in self.transfer],
            self.pencil)
        self.backward = Transform(
            [o.backward for o in self.xfftn[::-1]],
            [o.backward for o in self.transfer[::-1]],
            self.pencil[::-1])
        self.scalar_product = Transform(
            [o.scalar_product for o in self.xfftn],
            [o.forward for o in self.transfer],
            self.pencil)

        for i, base in enumerate(bases):
            if base.boundary_condition() == 'Dirichlet':
                base.bc.set_tensor_bcs(self)
            base.axis = i
            base._ndim_tensor = len(bases)


    def convolve(self, a_hat, b_hat, ab_hat):
        """Convolution of a_hat and b_hat

        Parameters
        ----------
            a_hat : array
                Input array of shape and type as output array from
                self.forward, or instance of :class:`.Function`
            b_hat : array
                Input array of shape and type as output array from
                self.forward, or instance of :class:`.Function`
            ab_hat : array
                Return array of same type and shape as a_hat and b_hat

        Note
        ----
        The return array ab_hat is truncated to the shape of a_hat and b_hat.

        Also note that self should have bases with padding for this method to give
        a convolution without aliasing. The padding is specified when creating
        instances of bases for the :class:`.TensorProductSpace`.

        FIXME Efficiency due to allocation
        """
        a = self.backward.output_array.copy()
        b = self.backward.output_array.copy()
        a = self.backward(a_hat, a)
        b = self.backward(b_hat, b)
        ab_hat = self.forward(a*b, ab_hat)
        return ab_hat

    def eval(self, points, coefficients, output_array=None, method=0):
        """Evaluate Function at points, given expansion coefficients

        Parameters
        ----------
            points : float or array of floats
                Array must be of shape (D, N), for  N points in D dimensions
            coefficients : array
                Expansion coefficients, or instance of :class:`.Function`
            output_array : array, optional
                Return array, function values at points
            method : int, optional
                Chooses implementation. The default 0 is a low-memory cython
                version. Using method = 1 leads to a faster cython
                implementation that, on the downside, uses more memory.
                The final, method = 2, is a python implementation.
        """
        if output_array is None:
            output_array = np.zeros(points.shape[1], dtype=self.forward.input_array.dtype)
        else:
            output_array[:] = 0
        if method == 0:
            return self._eval_lm_cython(points, coefficients, output_array)
        elif method == 1:
            return self._eval_cython(points, coefficients, output_array)
        else:
            return self._eval_python(points, coefficients, output_array)

    def _eval_python(self, points, coefficients, output_array):
        """Evaluate Function at points, given expansion coefficients

        Parameters
        ----------
            points : float or array of floats
            coefficients : array
                Expansion coefficients
            output_array : array
                Return array, function values at points
        """
        P = []
        last_conj_index = -1
        sl = -1
        out = []
        previous_axes = []
        flataxes = []
        for ax in self.axes:
            for a in ax:
                flataxes.append(a)
        for axis in flataxes:
            base = self.bases[axis]
            x = base.map_reference_domain(points[axis])
            D = base.evaluate_basis_all(x=x)
            P = D[..., self.local_slice()[axis]]
            if isinstance(base, R2CBasis):
                M = base.N//2+1
                if base.N % 2 == 0:
                    last_conj_index = M-1
                else:
                    last_conj_index = M
                sl = self.local_slice()[axis].start
                st = self.local_slice()[axis].stop
                if sl == 0:
                    sl = 1
                st = min(last_conj_index, st)
                sp = [slice(None), slice(sl, st)]

            if len(out) == 0:
                out = np.tensordot(P, coefficients, (1, axis))
                if isinstance(base, R2CBasis):
                    ss = [slice(None)]*len(self)
                    ss[axis] = slice(sl, st)
                    out += np.conj(np.tensordot(P[tuple(sp)], coefficients[tuple(ss)], (1, axis)))

            else:
                k = np.count_nonzero([m < axis for m in previous_axes])
                if len(self) == 2:
                    if not isinstance(base, R2CBasis):
                        out2 = np.sum(P*out, axis=-1)
                    else:
                        sp = tuple(sp)
                        out2 = np.sum(P.real*out.real - P.imag*out.imag, axis=-1)
                        out2 += np.sum(np.conj(P[sp].real*out[sp].real - P[sp].imag*out[sp].imag), axis=-1)

                elif len(self) == 3:
                    if len(out.shape) == 3:
                        sx = [slice(None)]*3
                        kk = 1 if axis-k == 1 else 2
                        sx[kk] = np.newaxis
                    else:
                        sx = [slice(None), slice(None)]
                    if not isinstance(base, R2CBasis):
                        out2 = np.sum(P[tuple(sx)]*out, axis=1+axis-k)
                    else:
                        out2 = np.sum(P[tuple(sx)].real*out.real - P[tuple(sx)].imag*out.imag, axis=1+axis-k)
                        sx[1+axis-k] = slice(sl, st)
                        sx = tuple(sx)
                        out2 += np.sum(np.conj(P[sx].real*out[sx].real - P[sx].imag*out[sx].imag), axis=1+axis-k)
                out = out2
            previous_axes.append(axis)
        output_array[:] = out
        output_array = self.comm.allreduce(output_array)
        return output_array

    def _eval_lm_cython(self, points, coefficients, output_array):
        """Evaluate Function at points, given expansion coefficients

        Parameters
        ----------
            points : float or array of floats
            coefficients : array
                Expansion coefficients
            output_array : array
                Return array, function values at points
        """
        r2c = -1
        last_conj_index = -1
        sl = -1
        x = []
        w = []
        for base in self:
            axis = base.axis
            if isinstance(base, R2CBasis):
                r2c = axis
                M = base.N//2+1
                if base.N % 2 == 0:
                    last_conj_index = M-1
                else:
                    last_conj_index = M
                sl = self.local_slice()[axis].start
            x.append(base.map_reference_domain(points[axis]))
            w.append(base.wavenumbers(bcast=False)[self.local_slice()[axis]].astype(np.float))

        if len(self) == 2:
            output_array = evaluate.evaluate_lm_2D(list(self.bases), output_array, coefficients, x[0], x[1], w[0], w[1], r2c, last_conj_index, sl)

        elif len(self) == 3:
            output_array = evaluate.evaluate_lm_3D(list(self.bases), output_array, coefficients, x[0], x[1], x[2], w[0], w[1], w[2], r2c, last_conj_index, sl)

        output_array = np.atleast_1d(output_array)
        output_array = self.comm.allreduce(output_array)
        return output_array

    def _eval_cython(self, points, coefficients, output_array):
        """Evaluate Function at points, given expansion coefficients

        Parameters
        ----------
            points : float or array of floats
            coefficients : array
                Expansion coefficients
            output_array : array
                Return array, function values at points
        """
        P = []
        r2c = -1
        last_conj_index = -1
        sl = -1
        for base in self:
            axis = base.axis
            x = base.map_reference_domain(points[axis])
            D = base.evaluate_basis_all(x=x)
            P.append(D[..., self.local_slice()[axis]])
            if isinstance(base, R2CBasis):
                r2c = axis
                M = base.N//2+1
                if base.N % 2 == 0:
                    last_conj_index = M-1
                else:
                    last_conj_index = M
                sl = self.local_slice()[axis].start
        if len(self) == 2:
            output_array = evaluate.evaluate_2D(output_array, coefficients, P, r2c, last_conj_index, sl)

        elif len(self) == 3:
            output_array = evaluate.evaluate_3D(output_array, coefficients, P, r2c, last_conj_index, sl)

        output_array = np.atleast_1d(output_array)
        output_array = self.comm.allreduce(output_array)
        return output_array

    def wavenumbers(self, scaled=False, eliminate_highest_freq=False):
        """Return list of wavenumbers of TensorProductSpace

        Parameters
        ----------
            scaled : bool, optional
                Scale wavenumbers with size of box
            eliminate_highest_freq : bool, optional
                Set Nyquist frequency to zero for evenly shaped axes of Fourier
                bases.

        """
        K = []
        for base in self:
            K.append(base.wavenumbers(scaled=scaled,
                                      eliminate_highest_freq=eliminate_highest_freq))
        return K

    def local_wavenumbers(self, broadcast=False, scaled=False,
                          eliminate_highest_freq=False):
        """Return list of local wavenumbers of TensorProductSpace

        Parameters
        ----------
            broadcast : bool, optional
                Broadcast returned wavenumber arrays to actual
                dimensions of TensorProductSpace
            scaled : bool, optional
                Scale wavenumbers with size of box
            eliminate_highest_freq : bool, optional
                Set Nyquist frequency to zero for evenly shaped axes of Fourier
                bases

        """
        k = self.wavenumbers(scaled=scaled, eliminate_highest_freq=eliminate_highest_freq)
        lk = []
        for axis, (n, s) in enumerate(zip(k, self.local_slice(True))):
            ss = [slice(None)]*len(k)
            ss[axis] = s
            lk.append(n[tuple(ss)])
        if broadcast is True:
            return [np.broadcast_to(m, self.local_shape(True)) for m in lk]
        return lk

    def mesh(self):
        """Return list of 1D physical meshes for each dimension of
        TensorProductSpace
        """
        X = []
        for base in self:
            X.append(base.mesh())
        return X

    def local_mesh(self, broadcast=False):
        """Return list of local 1D physical meshes for each dimension of
        TensorProductSpace

        Parameters
        ----------
            broadcast : bool, optional
                Broadcast each 1D mesh to real shape of
                :class:`.TensorProductSpace`
        """
        m = self.mesh()
        lm = []
        for axis, (n, s) in enumerate(zip(m, self.local_slice(False))):
            ss = [slice(None)]*len(m)
            ss[axis] = s
            lm.append(n[tuple(ss)])
        if broadcast is True:
            return [np.broadcast_to(m, self.local_shape(False)) for m in lm]
        return lm

    def shape(self, forward_output=False):
        """Return shape of arrays for TensorProductSpace

        Parameters
        ----------
            forward_output : bool, optional
                If True then return shape of an array that is the result of a
                forward transform. If False then return shape of physical
                space, i.e., the input to a forward transform.
        """
        if not forward_output:
            return [int(np.round(base.N*base.padding_factor)) for base in self]
        return [base.shape() for base in self]

    def __iter__(self):
        return iter(self.bases)

    @staticmethod
    def rank():
        """Return rank of TensorProductSpace"""
        return 1

    def __len__(self):
        """Return dimension of TensorProductSpace"""
        return len(self.bases)

    def num_components(self):
        """Return number of spaces in TensorProductSpace"""
        return 1

    def __getitem__(self, i):
        """Return instance of base i

        Parameters
        ----------
            i : int
        """
        return self.bases[i]


class MixedTensorProductSpace(object):
    """Class for composite tensorproductspaces.

    Parameters
    ----------
        spaces : list
            List of TensorProductSpaces
    """

    def __init__(self, spaces):
        self.spaces = spaces
        self.forward = VectorTransform([space.forward for space in spaces])
        self.backward = VectorTransform([space.backward for space in spaces])
        self.scalar_product = VectorTransform([space.scalar_product for space in spaces])

    def eval(self, points, coefficients, output_array=None, method=0):
        """Evaluate Function at points, given expansion coefficients

        Parameters
        ----------
            points : float or array of floats
            coefficients : array
                Expansion coefficients
            output_array : array, optional
                Return array, function values at points
            method : int, optional
                Chooses implementation. The default 0 is a low-memory cython
                version. Using method = 1 leads to a faster cython
                implementation that, on the downside, uses more memory.
                The final, method = 2, is a python implementation used only
                for verification.
        """

        if output_array is None:
            output_array = np.zeros((len(self.spaces), points.shape[-1]), dtype=self.forward.input_array.dtype)
        for i, space in enumerate(self.spaces):
            output_array[i] = space.eval(points, coefficients[i], output_array[i], method)
        return output_array

    def convolve(self, a_hat, b_hat, ab_hat):
        """Convolution of a_hat and b_hat

        Parameters
        ----------
            a_hat : array
                Input array of shape and type as output array from
                self.forward, or instance of :class:`.Function`
            b_hat : array
                Input array of shape and type as output array from
                self.forward, or instance of :class:`.Function`
            ab_hat : array
                Return array of same type and shape as a_hat and b_hat

        Note
        ----
        Note that self should have bases with padding for this method to give
        a convolution without aliasing. The padding is specified when creating
        instances of bases for the TensorProductSpace.

        FIXME Efficiency due to allocation
        """
        N = list(self.backward.output_array.shape)
        a = np.zeros([self.ndim()]+N, dtype=self.backward.output_array.dtype)
        b = np.zeros([self.ndim()]+N, dtype=self.backward.output_array.dtype)
        a = self.backward(a_hat, a)
        b = self.backward(b_hat, b)
        ab_hat = self.forward(a*b, ab_hat)
        return ab_hat

    def ndim(self):
        """Return dimension of scalar space"""
        return self.spaces[0].ndim()

    @staticmethod
    def rank():
        """Return rank of space"""
        return 2

    def shape(self, forward_output=False):
        """Return shape of arrays for MixedTensorProductSpace

        Parameters
        ----------
            forward_output : bool, optional
                If True then return shape of an array that is the result of a
                forward transform. If False then return shape of physical
                space, i.e., the input to a forward transform.
        """
        s = self.spaces[0].shape(forward_output)
        return [self.num_components()] + s

    def local_slice(self, forward_output=False):
        """The local view into the global data

        Parameters
        ----------
        forward_output : bool, optional
            Return local slices of output array (spectral space) if True, else
            return local slices of input array (physical space)

        """
        s = self.spaces[0].local_slice(forward_output)
        return [slice(None)] + s

    def num_components(self):
        """Return number of spaces in mixed space"""
        return len(self.spaces)

    def __getitem__(self, i):
        return self.spaces[i]

    def __getattr__(self, name):
        obj = object.__getattribute__(self, 'spaces')
        return getattr(obj[0], name)

    def __len__(self):
        return self.ndim()


class VectorTensorProductSpace(MixedTensorProductSpace):
    """A special MixedTensorProductSpace where the number of spaces must equal
    the geometrical dimension of the problem.

    For example, a TensorProductSpace created by a Cartesian product of 2 1D
    bases, will have vectors of length 2. A TensorProductSpace created from 3
    1D bases will have vectors of length 3.

    Parameters
    ----------
        space : TensorProductSpace
            Space to create vector from
    """

    def __init__(self, space):
        if isinstance(space, list):
            warnings.warn("Use only the TensorProductSpace as argument", DeprecationWarning)
            spaces = space
        else:
            spaces = [space]*space.ndim()
        MixedTensorProductSpace.__init__(self, spaces)

    def num_components(self):
        """Return number of spaces in mixed space"""
        assert len(self.spaces) == self.ndim()
        return self.ndim()

    @staticmethod
    def rank():
        """Return rank of space"""
        return 2


class VectorTransform(object):

    __slots__ = ('_transforms',)

    def __init__(self, transforms):
        self._transforms = transforms

    def __getattr__(self, name):
        obj = object.__getattribute__(self, '_transforms')
        if name == '_transforms':
            return obj
        return getattr(obj[0], name)

    def __call__(self, input_array, output_array, **kw):
        for i, transform in enumerate(self._transforms):
            output_array[i] = transform(input_array[i], output_array[i], **kw)
        return output_array


class Convolve(object):
    r"""Class for convolving without truncation.

    The convolution of :math:`\hat{a}` and :math:`\hat{b}` is computed by first
    transforming backwards with padding::

        a = Tp.backward(a_hat)
        b = Tp.backward(b_hat)

    and then transforming the product ``a*b`` forward without truncation::

        ab_hat = T.forward(a*b)

    where Tp is a :class:`.TensorProductSpace` for regular padding, and
    T is a TensorProductSpace with no padding, but using the shape
    of the padded a and b arrays.

    For convolve with truncation forward, use just the convolve method
    of the Tp space instead.

    Parameters
    ----------
        padding_space : TensorProductSpace
            Space with regular padding backward and truncation forward.
    """

    def __init__(self, padding_space):
        self.padding_space = padding_space
        shape = padding_space.shape()
        bases = []
        for i, base in enumerate(padding_space.bases):
            newbase = base.__class__(shape[i], padding_factor=1.0)
            bases.append(newbase)
        axes = []
        for axis in padding_space.axes:
            axes.append(axis[0])
        newspace = TensorProductSpace(padding_space.comm, bases, axes=axes)
        self.newspace = newspace

    def __call__(self, a_hat, b_hat, ab_hat=None):
        """Compute convolution of a_hat and b_hat without truncation

        Parameters
        ----------
            a_hat : :class:`.Function`
            b_hat : :class:`.Function`
            ab_hat : :class:`.Function`
        """
        Tp = self.padding_space
        T = self.newspace
        if ab_hat is None:
            ab_hat = Function(T)

        a = Array(Tp)
        b = Array(Tp)
        a = Tp.backward(a_hat, a)
        b = Tp.backward(b_hat, b)
        ab_hat = T.forward(a*b, ab_hat)
        return ab_hat


class BoundaryValues(object):
    """Class for setting nonhomogeneous boundary conditions for a 1D Dirichlet
    base inside a multidimensional TensorProductSpace.

    Parameters
    ----------
        T : TensorProductSpace
        bc : tuple of numbers
            Tuple with physical boundary values at edges of 1D domain
    """
    # pylint: disable=protected-access, redefined-outer-name, dangerous-default-value, unsubscriptable-object

    def __init__(self, T, bc=(0, 0)):
        self.T = T
        self.bc = bc            # Containing Data, sympy.Exprs or np.ndarray
        self.bcs = [0, 0]       # Processed bc
        self.bcs_final = [0, 0] # Data. May differ from bcs only for TensorProductSpaces
        self.sl0 = 0
        self.sl1 = 1
        self.slm1 = -1
        self.slm2 = -2
        self.axis = 0
        self.update_bcs(bc=bc)

    def update_bcs(self, sympy_params=None, bc=None):
        if sympy_params:
            assert isinstance(sympy_params, dict)
            for i in range(2):
                if isinstance(self.bc[i], sympy.Expr):
                    self.bcs[i] = self.bc[i].evalf(subs=sympy_params)
            self.bcs_final[:] = self.bcs

        if bc is not None:
            assert isinstance(bc, (list, tuple))
            assert len(bc) == 2
            self.bc = list(bc)
            for i in range(2):
                if isinstance(bc[i], (Number, sympy.Expr, np.ndarray)):
                    self.bcs[i] = bc[i]
                else:
                    raise NotImplementedError

            self.bcs_final[:] = self.bcs

    def set_tensor_bcs(self, T):
        self.T = T
        if isinstance(T, (chebyshev.bases.ShenDirichletBasis,
                          legendre.bases.ShenDirichletBasis)):
            # In this case we may be looking at multidimensional data with just one of the bases.
            # Mainly for testing that solvers and other routines work along any dimension.
            self.set_slices(T)
            self.axis = T.axis

        elif any(isinstance(base, (chebyshev.bases.ShenDirichletBasis,
                                   legendre.bases.ShenDirichletBasis))
                 for base in T.bases):
            # Setting the Dirichlet boundary condition in a TensorProductSpace
            # is more involved than for a single dimension, and the routine will
            # depend on the order of the bases. If the Dirichlet space is the last
            # one, then the boundary condition is applied directly. If there is
            # one Fourier space to the right, then one Fourier transform needs to
            # be performed on the bc data first. For two Fourier spaces to the right,
            # two transforms need to be executed.
            axis = None
            dirichlet_base = None
            number_of_bases_after_dirichlet = 0
            bases = []
            for axes in reversed(T.axes):
                #for ax in axes:
                base = T.bases[axes[-1]]
                if base.boundary_condition() == 'Dirichlet':
                    axis = self.axis = base.axis
                    dirichlet_base = base
                    bases.append('D')
                else:
                    if axis is None:
                        number_of_bases_after_dirichlet += 1
                    bases.append('F')

            self.set_slices(dirichlet_base)

            if self.has_nonhomogeneous_bcs() is False:
                self.bcs[0] = self.bcs_final[0] = 0
                self.bcs[1] = self.bcs_final[1] = 0
                return

            # Set boundary values
            # These are values set at the end of a transform in Dirichlet space,
            # but before any Fourier transforms
            # Shape is like real space, since Dirichlet does not alter shape
            b = Array(T)
            s = T.local_slice(False)[axis]

            if isinstance(self.bc[0], sympy.Expr):
                bc0 = self.bc[0]
                bc1 = self.bc[1]
                X = T.local_mesh(True)
                x, y, z = sympy.symbols("x,y,z")
                sym0 = [sym for sym in (x, y, z) if sym in bc0.free_symbols]
                sym1 = [sym for sym in (x, y, z) if sym in bc1.free_symbols]
                lbc0 = sympy.lambdify(sym0, bc0, 'numpy')
                lbc1 = sympy.lambdify(sym1, bc1, 'numpy')
                Y0 = []
                Y1 = []
                for i, ax in enumerate((x, y, z)):
                    if ax in bc0.free_symbols:
                        Y0.append(X[i][dirichlet_base.sl(0)])
                    if ax in bc1.free_symbols:
                        Y1.append(X[i][dirichlet_base.sl(0)])

                f_bc0 = lbc0(*Y0)
                f_bc1 = lbc1(*Y1)
                if s.stop == dirichlet_base.N:
                    b[self.slm2] = f_bc0
                    b[self.slm1] = f_bc1

            elif isinstance(self.bc[0], Number):
                if s.stop == dirichlet_base.N:
                    b[self.slm2] = self.bc[0]
                    b[self.slm1] = self.bc[1]

            elif isinstance(self.bc[0], np.ndarray):
                if s.stop == dirichlet_base.N:
                    b[self.slm2] = self.bc[0][self.sl0]
                    b[self.slm1] = self.bc[0][self.slm1]

            else:
                raise NotImplementedError

            if number_of_bases_after_dirichlet == 0:
                # Dirichlet base is the first to be transformed
                b_hat = b

            elif number_of_bases_after_dirichlet == 1:
                T.forward._xfftn[0].input_array[...] = b

                T.forward._xfftn[0]()
                arrayA = T.forward._xfftn[0].output_array
                arrayB = T.forward._xfftn[1].input_array
                T.forward._transfer[0](arrayA, arrayB)
                b_hat = arrayB.copy()

            elif number_of_bases_after_dirichlet == 2:
                T.forward._xfftn[0].input_array[...] = b

                T.forward._xfftn[0]()
                arrayA = T.forward._xfftn[0].output_array
                arrayB = T.forward._xfftn[1].input_array
                T.forward._transfer[0](arrayA, arrayB)

                T.forward._xfftn[1]()
                arrayA = T.forward._xfftn[1].output_array
                arrayB = T.forward._xfftn[2].input_array
                T.forward._transfer[1](arrayA, arrayB)
                b_hat = arrayB.copy()

            # Now b_hat contains the correct slices in slm1 and slm2
            self.bcs[0] = b_hat[self.slm2].copy()
            self.bcs[1] = b_hat[self.slm1].copy()

            # Final
            T.forward._xfftn[0].input_array[...] = b
            for i in range(len(T.forward._transfer)):
                if bases[i] == 'F':
                    T.forward._xfftn[i]()
                else:
                    T.forward._xfftn[i].output_array[...] = T.forward._xfftn[i].input_array

                arrayA = T.forward._xfftn[i].output_array
                arrayB = T.forward._xfftn[i+1].input_array
                T.forward._transfer[i](arrayA, arrayB)

            if bases[-1] == 'F':
                T.forward._xfftn[-1]()
            else:
                T.forward._xfftn[-1].output_array[...] = T.forward._xfftn[-1].input_array

            b_hat = T.forward._xfftn[-1].output_array
            self.bcs_final[0] = b_hat[self.slm2].copy()
            self.bcs_final[1] = b_hat[self.slm1].copy()

    def set_slices(self, T):
        self.sl0 = T.sl(0)
        self.sl1 = T.sl(1)
        self.slm1 = T.sl(-1)
        self.slm2 = T.sl(-2)

    def apply_before(self, u, final=False, scales=(0.5, 0.5)):
        if final is True:
            u[self.sl0] += scales[0]*(self.bcs_final[0] + self.bcs_final[1])
            u[self.sl1] += scales[1]*(self.bcs_final[0] - self.bcs_final[1])

        else:
            u[self.sl0] += scales[0]*(self.bcs[0] + self.bcs[1])
            u[self.sl1] += scales[1]*(self.bcs[0] - self.bcs[1])

    def apply_after(self, u, final=False):
        if final is True:
            u[self.slm2] = self.bcs_final[0]
            u[self.slm1] = self.bcs_final[1]

        else:
            u[self.slm2] = self.bcs[0]
            u[self.slm1] = self.bcs[1]

    def has_nonhomogeneous_bcs(self):
        if self.bc[0] == 0 and self.bc[1] == 0:
            return False
        return True

def some_basic_tests():
    import pyfftw
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    N = 8
    K0 = C2CBasis(N)
    K1 = C2CBasis(N)
    K2 = C2CBasis(N)
    K3 = R2CBasis(N)
    T = TensorProductSpace(comm, (K0, K1, K2, K3))

    # Create data on rank 0 for testing
    if comm.Get_rank() == 0:
        f_g = np.random.random(T.shape())
        f_g_hat = pyfftw.interfaces.numpy_fft.rfftn(f_g, axes=(0, 1, 2, 3))
    else:
        f_g = np.zeros(T.shape())
        f_g_hat = np.zeros(T.shape(), dtype=np.complex)

    # Distribute test data to all ranks
    comm.Bcast(f_g, root=0)
    comm.Bcast(f_g_hat, root=0)

    # Create a function in real space to hold the test data
    fj = Array(T)
    fj[:] = f_g[T.local_slice(False)]

    # Perform forward transformation
    f_hat = T.forward(fj)

    assert np.allclose(f_g_hat[T.local_slice(True)], f_hat*N**4)

    # Perform backward transformation
    fj2 = Array(T)
    fj2 = T.backward(f_hat)

    assert np.allclose(fj, fj2)

    f_hat = T.scalar_product(fj)

    # Padding
    # Needs new instances of bases because arrays have new sizes
    Kp0 = C2CBasis(N, padding_factor=1.5)
    Kp1 = C2CBasis(N, padding_factor=1.5)
    Kp2 = C2CBasis(N, padding_factor=1.5)
    Kp3 = R2CBasis(N, padding_factor=1.5)
    Tp = TensorProductSpace(comm, (Kp0, Kp1, Kp2, Kp3))

    f_g_pad = Tp.backward(f_hat)
    f_hat2 = Tp.forward(f_g_pad)

    assert np.allclose(f_hat2, f_hat)


if __name__ == '__main__':
    some_basic_tests()
