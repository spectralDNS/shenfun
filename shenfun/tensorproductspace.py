import numpy as np
from shenfun.fourier.bases import FourierBase, R2CBasis, C2CBasis
import shenfun
import pyfftw
import six
from mpi4py_fft.mpifft import Transform
from mpi4py_fft.pencil import Subcomm, Pencil

class TensorProductSpace(object):

    def __init__(self, comm, bases, axes=None, dtype=None, **kw):
        self.bases = bases
        shape = self.shape()
        assert len(shape) > 0
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
            dtype = np.complex if isinstance(bases[-1], C2CBasis) else np.float
        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG'

        slab = False
        if isinstance(comm, Subcomm):
            assert len(comm) == len(shape)
            assert comm[axes[-1]].Get_size() == 1
            self.subcomm = comm
        else:
            if slab:
                dims = [1] * len(shape)
                dims[0] = 0
            else:
                dims = [0] * len(shape)
                dims[axes[-1]] = 1
            self.subcomm = Subcomm(comm, dims)

        collapse = False # kw.pop('collapse', True)
        if collapse:
            groups = [[]]
            for axis in reversed(axes):
                if self.subcomm[axis].Get_size() == 1:
                    groups[0].insert(0, axis)
                else:
                    groups.insert(0, [axis])
            self.axes = tuple(map(tuple, groups))
        else:
            self.axes = tuple((axis,) for axis in axes)

        self.xfftn = []
        self.transfer = []
        self.pencil = [None, None]

        axes = self.axes[-1]
        pencil = Pencil(self.subcomm, shape, axes[-1])
        self.bases[-1].plan(pencil.subshape, axes, dtype, kw)
        self.pencil[0] = pencilA = pencil
        if not shape[axes[-1]] == self.bases[-1].forward.output_array.shape[axes[-1]]:
            dtype = self.bases[-1].forward.output_array.dtype
            shape[axes[-1]] = self.bases[-1].forward.output_array.shape[axes[-1]]
            pencilA = Pencil(self.subcomm, shape, axes[-1])

        for i, axes in enumerate(reversed(self.axes[:-1])):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = self.bases[-(i+2)]
            xfftn.plan(pencilB.subshape, axes, dtype, kw)
            self.transfer.append(transAB)
            pencilA = pencilB
            if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
                pencilA = Pencil(pencilB.subcomm, shape, axes[-1])

        self.pencil[1] = pencilA

        self.forward = Transform(
            [o.forward for o in self.bases[::-1]],
            [o.forward for o in self.transfer],
            self.pencil)
        self.backward = Transform(
            [o.backward for o in self.bases],
            [o.backward for o in self.transfer[::-1]],
            self.pencil[::-1])
        self.scalar_product = Transform(
            [o.scalar_product for o in self.bases[::-1]],
            [o.forward for o in self.transfer],
            self.pencil)

    def destroy(self):
        self.subcomm.destroy()
        for trans in self.transfer:
            trans.destroy()

    def wavenumbers(self):
        K = []
        N = self.shape()
        for axis, base in enumerate(self):
            K.append(base.wavenumbers(N, axis))
        return K

    def local_wavenumbers(self, broadcast=False):
        k = self.wavenumbers()
        lk = []
        for axis, (n, s) in enumerate(zip(k, self.local_slice(True))):
            ss = [slice(None)]*len(k)
            ss[axis] = s
            lk.append(n[ss])
        if broadcast is True:
            return [np.broadcast_to(m, self.local_shape(True)) for m in lk]
        return lk

    def mesh(self):
        X = []
        N = self.shape()
        for axis, base in enumerate(self):
            X.append(base.mesh(N, axis))
        return X

    def local_mesh(self, broadcast=False):
        m = self.mesh()
        lm = []
        for axis, (n, s) in enumerate(zip(m, self.local_slice(False))):
            ss = [slice(None)]*len(m)
            ss[axis] = s
            lm.append(n[ss])
        if broadcast is True:
            return [np.broadcast_to(m, self.local_shape(False)) for m in lm]
        return lm

    def shape(self):
        return [int(np.round(base.N*base.padding_factor)) for base in self]

    def spectral_shape(self):
        return [base.spectral_shape() for base in self]

    def __iter__(self):
        return iter(self.bases)

    def test_function(self):
        return (self, np.zeros((1, len(self)), dtype=np.int))

    def local_shape(self, spectral=True):
        if not spectral:
            return self.forward.input_pencil.subshape
        else:
            return self.backward.input_pencil.subshape

    def local_slice(self, spectral=True):
        """The local view into the global data"""

        if not spectral is True:
            ip = self.forward.input_pencil
            s = [slice(start, start+shape) for start, shape in zip(ip.substart,
                                                                   ip.subshape)]
        else:
            ip = self.backward.input_pencil
            s = [slice(start, start+shape) for start, shape in zip(ip.substart,
                                                                   ip.subshape)]
        return s

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, i):
        return self.bases[i]


class Function(np.ndarray):
    """Numpy array for TensorProductSpace

    Parameters
    ----------

    space : Instance of TensorProductSpace
    input_array: boolean.
        If True then create Function of shape/type for input to PFFT.forward,
        otherwise create Function of shape/type for output from PFFT.forward
    val : int or float
        Value used to initialize array

    For more information, see numpy.ndarray

    Examples
    --------
    from mpi4py_fft import MPI
    from shenfun.tensorproductspace import TensorProductSpace, Function
    from shenfun.fourier.bases import R2CBasis, C2CBasis

    K0 = C2CBasis(8)
    K1 = R2CBasis(8)
    FFT = TensorProductSpace(MPI.COMM_WORLD, [K0, K1])
    u = Function(FFT)
    uhat = Function(FFT, False)

    """

    # pylint: disable=too-few-public-methods,too-many-arguments

    def __new__(cls, space, input_array=True, val=0):

        shape = space.forward.input_array.shape
        dtype = space.forward.input_array.dtype
        if spectral is True:
            shape = space.forward.output_array.shape
            dtype = space.forward.output_array.dtype

        obj = np.ndarray.__new__(cls,
                                 shape,
                                 dtype=dtype)
        obj.fill(val)
        return obj

def inner_product(test, trial):
    assert test[1].shape[0] == trial[1].shape[0]

    base = test[0]
    A = []

    for base_test, base_trial in zip(test[1], trial[1]):

        if len(base_test.shape) == 1:
            base_test = base_test[np.newaxis, :]
        if len(base_trial.shape) == 1:
            base_trial = base_trial[np.newaxis, :]

        for b0 in base_test:
            for b1 in base_trial:
                A.append([])
                assert len(b0) == len(b1)
                for i, (a, b) in enumerate(zip(b0, b1)):
                    AA = shenfun.inner_product((base[i], a), (base[i], b))
                    AA.axis = i
                    A[-1].append(AA)

    # Strip off diagonal matrices, put contribution in scale array
    B = []
    for matrices in A:
        scale = np.ones(1).reshape((1,)*len(base))
        nonperiodic = None
        for axis, mat in enumerate(matrices):
            if isinstance(base[axis], shenfun.fourier.FourierBase):
                a = mat[0]
                if np.ndim(a):
                    ss = [np.newaxis]*len(base)
                    ss[axis] = slice(None)
                    a = mat[0][ss]
                scale = scale*a
            else:
                nonperiodic = mat
                nonperiodic.axis = axis
        if nonperiodic is None:
            # All diagonal matrices
            B.append(scale)

        else:
            nonperiodic.scale = scale
            B.append(nonperiodic)

    # Get local matrices
    if np.all([isinstance(b, np.ndarray) for b in B]):
        # All Fourier
        BB = []
        for b in B:
            s = b.shape
            ss = [slice(None)]*len(base)
            ls = base.local_slice()
            for axis, shape in enumerate(s):
                if shape > 1:
                    ss[axis] = ls[axis]
            BB.append((b[ss]).copy())

        diagonal_array = BB[0]
        for ci in BB[1:]:
            diagonal_array = diagonal_array + ci

        diagonal_array = np.where(diagonal_array==0, 1, diagonal_array)

        return {'diagonal': diagonal_array}

    else:
        c = B[0]
        C = {c.__class__.__name__: c}
        for b in B[1:]:
            name = b.__class__.__name__
            if name in C:
                C[name].scale = C[name].scale + b.scale
            else:
                C[name] = b

        for v in C.values():
            if hasattr(v, 'scale'):
                s = v.scale.shape
                ss = [slice(None)]*len(base)
                ls = base.local_slice()
                for axis, shape in enumerate(s):
                    if shape > 1:
                        ss[axis] = ls[axis]
                v.scale = (v.scale[ss]).copy()

        return C


if __name__ == '__main__':
    import shenfun
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    N = 8
    K0 = shenfun.fourier.bases.C2CBasis(N)
    K1 = shenfun.fourier.bases.C2CBasis(N)
    K2 = shenfun.fourier.bases.C2CBasis(N)
    K3 = shenfun.fourier.bases.R2CBasis(N)
    T = TensorProductSpace(comm, (K0, K1, K2, K3))

    # Create data on rank 0 for testing
    if comm.Get_rank() == 0:
        f_g = np.random.random(T.shape())
        f_g_hat = pyfftw.interfaces.numpy_fft.rfftn(f_g, axes=(0, 1, 2, 3))
    else:
        f_g = np.zeros(T.shape())
        f_g_hat = np.zeros(T.spectral_shape(), dtype=np.complex)

    # Distribute test data to all ranks
    comm.Bcast(f_g, root=0)
    comm.Bcast(f_g_hat, root=0)

    # Create a function in real space to hold the test data
    fj = Function(T, spectral=False)
    fj[:] = f_g[T.local_slice(False)]

    # Perform forward transformation
    f_hat = T.forward(fj)

    assert np.allclose(f_g_hat[T.local_slice(True)], f_hat*N**4)

    # Perform backward transformation
    fj2 = Function(T, spectral=False)
    fj2 = T.backward(f_hat)

    assert np.allclose(fj, fj2)

    f_hat = T.scalar_product(fj)

    # Padding
    # Needs new instances of bases because arrays have new sizes
    Kp0 = shenfun.fourier.bases.C2CBasis(N)
    Kp1 = shenfun.fourier.bases.C2CBasis(N)
    Kp2 = shenfun.fourier.bases.C2CBasis(N)
    Kp3 = shenfun.fourier.bases.R2CBasis(N)
    Tp = TensorProductSpace(comm, (Kp0, Kp1, Kp2, Kp3), padding=True)

    f_g_pad = Tp.backward(f_hat)
    f_hat2 = Tp.forward(f_g_pad)

    assert np.allclose(f_hat2, f_hat)
