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

    Parameters
    ----------
        h5name : str
                 Name of hdf5 file to be created
        names : list of str's
                Names of fields to be stored
        T : TensorProductSpace
            Instance of a TensorProductSpace. Must be the same as the space
            used for storing with 'write_tstep' and 'write_slice_tstep'
    """
    def __init__(self, h5name, names, T):
        self.f = h5py.File(h5name, "w", driver="mpio", comm=T.comm)
        self.f.create_group("mesh")
        self.T = T
        self.names = names
        x = T.mesh()
        for i in range(len(x)):
            self.f["mesh"].create_dataset("x{}".format(i), data=np.squeeze(x[i]))
        for name in names:
            self.f.create_group(name)

    def write_tstep(self, tstep, u, spectral=False):
        """Write field u to HDF5 format at a given time step

        Parameters
        ----------
            tstep : int
                    Time step
            u : array
                Function or Array. The field to be stored
            spectral : bool, optional
                       If False, then u is an array from real physical space,
                       if True, then u is an array from spectral space.

        Note
        ----
        Fields with names 'name' will be stored under

            - name/{2,3}D/tstep

        """
        assert isinstance(u, np.ndarray)

        if isinstance(self.T, MixedTensorProductSpace):
            assert self.T.ndim() == len(u.shape[1:])
            assert len(self.names) == u.shape[0]
            for i in range(u.shape[0]):
                self._write_group(self.names[i], u[i], tstep, spectral)
        else:
            assert len(self.names) == 1
            self._write_group(self.names[0], u, tstep, spectral)

    def write_slice_tstep(self, tstep, sl, u):
        """Write slice of field u to HDF5 format at a given time step

        Parameters
        ----------
            tstep : int
                    Time step
            sl : list of slices
                 The slice to be stored
            u : array
                Function or Array. The field to be stored

        Note
        ----
        Slices of fields with name 'name' will be stored for, e.g.,
        sl = [slice(None), 16, slice(None)], as

            name/2D/slice_16_slice/tstep

        whereas sl = [8, slice(None), 12] will be stored as

            name/1D/8_slice_12/tstep

        """

        assert isinstance(u, np.ndarray)
        ndims = sl.count(slice(None))
        sl = list(sl)
        ii = [isinstance(s, int) for s in sl].index(True)
        si = sl[ii]
        sp = []
        for i, j in enumerate(sl):
            if isinstance(j, slice):
                sp.append(i)
        slname = ''
        for ss in sl:
            if isinstance(ss, slice):
                slname += 'slice_'
            else:
                slname += str(ss)+'_'
        slname = slname[:-1]
        s = self.T.local_slice(False)

        # Check if slice is on this processor and make sl local
        inside = 1
        sf = []
        for i, j in enumerate(sl):
            if isinstance(j, slice):
                sf.append(s[i])
            else:
                if j >= s[i].start and j < s[i].stop:
                    inside *= 1
                    sl[i] -= s[i].start
                else:
                    inside *= 0

        if isinstance(self.T, MixedTensorProductSpace):
            assert self.T.ndim() == len(u.shape[1:])
            assert len(self.names) == u.shape[0]
            sl.insert(0, 0)
            for i in range(u.shape[0]):
                sl[0] = i
                self._write_slice_group(self.names[i], slname, ndims, sp, u, sl, sf, inside, tstep)

        else:
            assert len(self.names) == 1
            self._write_slice_group(self.names[0], slname, ndims, sp, u, sl, sf, inside, tstep)

    def close(self):
        self.f.close()

    def _write_group(self, name, u, tstep, spectral):
        s = self.T.local_slice(spectral)
        group = "/".join((name, "{}D".format(len(u.shape))))
        if group not in self.f:
            self.f.create_group(group)
        self.f[group].create_dataset(str(tstep), shape=self.T.shape(spectral), dtype=u.dtype)
        if self.T.ndim() == 5:
            self.f["/".join((group, str(tstep)))][s[0], s[1], s[2], s[3], s[4]] = u
        elif self.T.ndim() == 4:
            self.f["/".join((group, str(tstep)))][s[0], s[1], s[2], s[3]] = u
        elif self.T.ndim() == 3:
            self.f["/".join((group, str(tstep)))][s[0], s[1], s[2]] = u
        elif self.T.ndim() == 2:
            self.f["/".join((group, str(tstep)))][s[0], s[1]] = u
        else:
            raise NotImplementedError

    def _write_slice_group(self, name, slname, ndims, sp, u, sl, sf, inside, tstep):
        group = "/".join((name, "{}D".format(ndims), slname))
        if group not in self.f:
            self.f.create_group(group)
        self.f[group].create_dataset(str(tstep), shape=np.take(self.N, sp), dtype=u.dtype)
        if inside == 1:
            if len(sf) == 3:
                self.f["/".join((group, str(tstep)))][sf[0], sf[1], sf[2]] = u[sl]
            elif len(sf) == 2:
                self.f["/".join((group, str(tstep)))][sf[0], sf[1]] = u[sl]
            elif len(sf) == 1:
                self.f["/".join((group, str(tstep)))][sf[0]] = u[sl]

