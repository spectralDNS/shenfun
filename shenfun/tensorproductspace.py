import numpy as np
from shenfun.fourier.bases import FourierBase, R2CBasis, C2CBasis
from mpi4py_fft.mpifft import Transform
from mpi4py_fft.pencil import Subcomm, Pencil

__all__ = ('TensorProductSpace', )


class TensorProductSpace(object):

    def __init__(self, comm, bases, axes=None, dtype=None, **kw):
        self.comm = comm
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
        dtype = self.dtype = np.dtype(dtype)
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

    def rank(self):
        return 1

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, i):
        return self.bases[i]

    def is_forward_output(self, u):
        return (np.all(u.shape == self.forward.output_array.shape) and
                u.dtype == self.forward.output_array.dtype)

    def as_function(self, u):
        from .arguments import Function
        assert isinstance(u, np.ndarray)
        forward_output = self.is_forward_output(u)
        return Function(self, forward_output=forward_output, buffer=u)

class VectorTransform(object):

    __slots__ = ('_transform', '_dim')

    def __init__(self, transform, dim):
        self._transform = transform
        self._dim = dim

    @property
    def dim(self):
        return object.__getattribute__(self, '_dim')

    @property
    def transform(self):
        return object.__getattribute__(self, '_transform')

    def __call__(self, input_array, output_array, **kw):
        for i in range(self.dim):
            output_array[i] = self.transform(input_array[i], output_array[i], **kw)
        return output_array


class VectorTensorProductSpace(TensorProductSpace):

    def __init__(self, comm, bases, axes=None, dtype=None, **kw):
        TensorProductSpace.__init__(self, comm, bases, axes=axes, dtype=dtype, **kw)
        self.forward = VectorTransform(self.forward, len(self.bases))
        self.backward = VectorTransform(self.backward, len(self.bases))
        self.scalar_product = VectorTransform(self.scalar_product, len(self.bases))

    def rank(self):
        return 2


if __name__ == '__main__':
    import shenfun
    import pyfftw
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
    fj = shenfun.Function(T, False)
    fj[:] = f_g[T.local_slice(False)]

    # Perform forward transformation
    f_hat = T.forward(fj)

    assert np.allclose(f_g_hat[T.local_slice(True)], f_hat*N**4)

    # Perform backward transformation
    fj2 = shenfun.Function(T, False)
    fj2 = T.backward(f_hat)

    assert np.allclose(fj, fj2)

    f_hat = T.scalar_product(fj)

    # Padding
    # Needs new instances of bases because arrays have new sizes
    Kp0 = shenfun.fourier.bases.C2CBasis(N, padding_factor=1.5)
    Kp1 = shenfun.fourier.bases.C2CBasis(N, padding_factor=1.5)
    Kp2 = shenfun.fourier.bases.C2CBasis(N, padding_factor=1.5)
    Kp3 = shenfun.fourier.bases.R2CBasis(N, padding_factor=1.5)
    Tp = TensorProductSpace(comm, (Kp0, Kp1, Kp2, Kp3))

    f_g_pad = Tp.backward(f_hat)
    f_hat2 = Tp.forward(f_g_pad)

    assert np.allclose(f_hat2, f_hat)
