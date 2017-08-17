import numpy as np
from shenfun.fourier.bases import FourierBase, R2CBasis, C2CBasis
from shenfun import chebyshev, legendre
from mpi4py_fft.mpifft import Transform
from mpi4py_fft.pencil import Subcomm, Pencil

__all__ = ('TensorProductSpace', 'VectorTensorProductSpace')


class TensorProductSpace(object):

    def __init__(self, comm, bases, axes=None, dtype=None, slab=False, **kw):
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

        if any(isinstance(base, (chebyshev.bases.ShenDirichletBasis,
                                 legendre.bases.ShenDirichletBasis))
                                 for base in self.bases):
            boundary_values = BoundaryValues(self)
            for base in self.bases:
                if isinstance(base, (legendre.bases.ShenDirichletBasis,
                                     chebyshev.bases.ShenDirichletBasis)):
                    base.bc = boundary_values

    def destroy(self):
        self.subcomm.destroy()
        for trans in self.transfer:
            trans.destroy()

    def wavenumbers(self, scaled=False):
        K = []
        N = self.shape()
        for axis, base in enumerate(self):
            K.append(base.wavenumbers(N, axis, scaled=scaled))
        return K

    def local_wavenumbers(self, broadcast=False, scaled=False):
        k = self.wavenumbers(scaled=scaled)
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

    def ndim(self):
        return len(self.bases)

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, i):
        return self.bases[i]

    def is_forward_output(self, u):
        return (u.shape == self.forward.output_array.shape and
                u.dtype == self.forward.output_array.dtype)

    def as_function(self, u):
        from .forms.arguments import Function
        assert isinstance(u, np.ndarray)
        forward_output = self.is_forward_output(u)
        return Function(self, forward_output=forward_output, buffer=u)


class MixedTensorProductSpace(object):

    def __init__(self, spaces):
        self.spaces = spaces
        self.forward = VectorTransform([space.forward for space in spaces])
        self.backward = VectorTransform([space.backward for space in spaces])
        self.scalar_product = VectorTransform([space.scalar_product for space in spaces])

    def ndim(self):
        return self.spaces[0].ndim()

    def rank(self):
        raise NotImplementedError

    def is_forward_output(self, u):
        return (u[0].shape == self.forward.output_array.shape and
                u[0].dtype == self.forward.output_array.dtype)

    def __getitem__(self, i):
        return self.spaces[i]

    def __getattr__(self, name):
        obj = object.__getattribute__(self, 'spaces')
        return getattr(obj[0], name)


class VectorTensorProductSpace(MixedTensorProductSpace):

    def __init__(self, spaces):
        MixedTensorProductSpace.__init__(self, spaces)

    def rank(self):
        return 2


class VectorTransform(object):

    __slots__ = ('_transforms')

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


class BoundaryValues(object):

    def __init__(self, T, bc=None):

        if isinstance(T, (legendre.bases.ShenDirichletBasis,
                          chebyshev.bases.ShenDirichletBasis)):
            self.bc0 = self.bc0_final = bc[0]
            self.bc1 = self.bc1_final = bc[1]
            self.sl0 = 0
            self.sl1 = 1
            self.slm1 = -1
            self.slm2 = -2
            self.axis = 0

        elif any(isinstance(base, (chebyshev.bases.ShenDirichletBasis,
                                   legendre.bases.ShenDirichletBasis))
                                   for base in T.bases):
            from shenfun import Array
            axis = None
            dirichlet_base = None
            number_of_bases_after_dirichlet = 0
            bases = []
            for axes in reversed(T.axes):
                base = T.bases[axes[0]]
                assert len(axes) == 1
                assert axes[0] == base.axis
                if isinstance(base, (legendre.bases.ShenDirichletBasis,
                                     chebyshev.bases.ShenDirichletBasis)):
                    axis = self.axis = base.axis
                    self.sl = base.sl(0)
                    dirichlet_base = base
                    bases.append('D')

                else:
                    if axis is None:
                        number_of_bases_after_dirichlet += 1
                    bases.append('F')

            self.set_slices(dirichlet_base)

            if axis is None or dirichlet_base.bc.has_nonhomogeneous_bcs() is False:
                self.bc0 = self.bc0_final = 0
                self.bc1 = self.bc1_final = 0
                return

            # Set boundary values
            # These are values set at the end of a transform in Dirichlet space,
            # but before any Fourier transforms
            # Shape is like real space, since Dirichlet does not alter shape
            b = Array(T, False)
            b[self.slm2] = dirichlet_base.bc.bc0
            b[self.slm1] = dirichlet_base.bc.bc1

            self.number_of_bases_after_dirichlet = number_of_bases_after_dirichlet

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
                #
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
            self.bc0 = b_hat[self.slm2].copy()
            self.bc1 = b_hat[self.slm1].copy()

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
            self.bc0_final = b_hat[self.slm2].copy()
            self.bc1_final = b_hat[self.slm1].copy()

    def set_slices(self, T):
        self.sl0 = T.sl(0)
        self.sl1 = T.sl(1)
        self.slm1 = T.sl(-1)
        self.slm2 = T.sl(-2)

    def apply_before(self, u, final=False, scales=(0.5, 0.5)):
        if final is True:
            u[self.sl0] += scales[0]*(self.bc0_final + self.bc1_final)
            u[self.sl1] += scales[1]*(self.bc0_final - self.bc1_final)

        else:
            u[self.sl0] += scales[0]*(self.bc0 + self.bc1)
            u[self.sl1] += scales[1]*(self.bc0 - self.bc1)

    def apply_after(self, u, final=False):
        if final is True:
            u[self.slm2] = self.bc0_final
            u[self.slm1] = self.bc1_final

        else:
            u[self.slm2] = self.bc0
            u[self.slm1] = self.bc1

    def has_nonhomogeneous_bcs(self):
        if self.bc0 == 0 and self.bc1 == 0:
            return False
        return True

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
