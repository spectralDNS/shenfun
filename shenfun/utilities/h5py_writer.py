#pylint: disable=missing-docstring, consider-using-enumerate

import warnings
import numpy as np
from shenfun import MixedTensorProductSpace

try:
    import h5py
except ImportError:
    warnings.warn('h5py not installed')

__all__ = ('HDF5Writer',)


class HDF5Writer(object):
    """Class for writing data in shenfun to HDF5 format

    args:
        h5name      string            Name of hdf5 file to be created
        names    list of strings      Names of fields to be stored
        T       TensorProductSpace    Instance of a TensorProductSpace
                                      Must be the same as the space used
                                      for storing with 'write_tstep'
                                      and 'write_slice_tstep'
    """
    def __init__(self, h5name, names, T):
        self.f = h5py.File(h5name, "w", driver="mpio", comm=T.comm)
        self.f.create_group("mesh")
        self.T = T
        self.N = T.shape()
        self.names = names
        x = T.mesh()
        for i in range(len(x)):
            xyz = {0:'x', 1:'y', 2:'z'}[i]
            self.f["mesh"].create_dataset(xyz, data=np.squeeze(x[i]))
        for name in names:
            self.f.create_group(name)

    def write_tstep(self, tstep, u):
        """Write field u to HDF5 format at a given time step

        args:
            tstep        int          Time step
            u      Function/Array     The field to be stored

        Fields with names 'name' will be stored under
            name/{2,3}D/tstep

        """
        assert isinstance(u, np.ndarray)

        if isinstance(self.T, MixedTensorProductSpace):
            assert self.T.ndim() == len(u.shape[1:])
            assert len(self.names) == u.shape[0]
            s = self.T.local_slice(False)
            for i in range(u.shape[0]):
                group = "/".join((self.names[i], "{}D".format(len(u[i].shape))))
                if group not in self.f:
                    self.f.create_group(group)
                self.f[group].create_dataset(str(tstep), shape=self.N, dtype=u.dtype)
                if self.T.ndim() == 3:
                    self.f["/".join((group, str(tstep)))][s[0], s[1], s[2]] = u[i]
                elif self.T.ndim() == 2:
                    self.f["/".join((group, str(tstep)))][s[0], s[1]] = u[i]
                else:
                    raise NotImplementedError
        else:
            assert len(self.names) == 1
            group = "/".join((self.names[0], "{}D".format(len(u.shape))))
            if group not in self.f:
                self.f.create_group(group)
            self.f[group].create_dataset(str(tstep), shape=self.N, dtype=u.dtype)
            s = self.T.local_slice(False)
            if self.T.ndim() == 3:
                self.f["/".join((group, str(tstep)))][s[0], s[1], s[2]] = u[:]
            elif self.T.ndim() == 2:
                self.f["/".join((group, str(tstep)))][s[0], s[1]] = u[:]
            else:
                raise NotImplementedError

    def write_slice_tstep(self, tstep, sl, u):
        """Write slice of field u to HDF5 format at a given time step

        args:
            tstep        int          Time step
            sl     list of slices     The slice to be stored
            u      Function/Array     The field to be stored

        Slices of fields with name 'name' will be stored for, e.g.,
        sl = [slice(None), 16, slice(None)], as
            name/2D/slice_16_slice/tstep

        """

        assert isinstance(u, np.ndarray)
        assert len(sl) == 3
        assert sl.count(slice(None)) == 2
        sl = list(sl)
        ii = [isinstance(s, int) for s in sl].index(True)
        si = sl[ii]
        sp = list(range(3))
        sp.pop(ii)
        slname = ''
        for ss in sl:
            if isinstance(ss, slice):
                slname += 'slice_'
            else:
                slname += str(ss)+'_'
        slname = slname[:-1]

        if isinstance(self.T, MixedTensorProductSpace):
            assert self.T.ndim() == len(u.shape[1:])
            assert len(self.names) == u.shape[0]
            assert self.T.ndim() == 3
            s = self.T.local_slice(False)
            sx = s[ii]
            s.pop(ii)
            sl.insert(0, 0)
            for i in range(u.shape[0]):
                group = "/".join((self.names[i], "2D", slname))
                if group not in self.f:
                    self.f.create_group(group)
                self.f[group].create_dataset(str(tstep), shape=np.take(self.N, sp), dtype=u.dtype)
                if si >= sx.start and si < sx.stop:
                    sl[0] = i
                    sl[ii+1] = sl[ii+1]-sx.start
                    self.f["/".join((group, str(tstep)))][s[0], s[1]] = u[sl]
        else:
            assert len(self.names) == 1
            group = "/".join((self.names[0], "2D", slname))
            if group not in self.f:
                self.f.create_group(group)
            self.f[group].create_dataset(str(tstep), shape=np.take(self.N, sp), dtype=u.dtype)
            s = self.T.local_slice(False)
            sx = s[ii]
            s.pop(ii)
            if si >= sx.start and si < sx.stop:
                sl[ii] = sl[ii]-sx.start
                self.f["/".join((group, str(tstep)))][s[0], s[1]] = u[sl]

    def close(self):
        self.f.close()
