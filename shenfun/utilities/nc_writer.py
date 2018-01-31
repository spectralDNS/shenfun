import numpy as np
import warnings
from shenfun import MixedTensorProductSpace

# https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py

try:
    from netCDF4 import Dataset
except:
    warnings.warn('netcdf not installed')

__all__ = ('NCWriter',)


class NCWriter(object):
    """Class for writing data in shenfun to netcdf format

    args:
        ncname   string               Name of netcdf file to be created
        names    list of strings      Names of fields to be stored
        T        TensorProductSpace   Instance of a TensorProductSpace
                                      Must be the same as the space used
                                      for storing with 'write_tstep'
                                      and 'write_slice_tstep'
        clobber  boolean
    """
    def __init__(self, ncname, names, T, **kwargs):
        self.f = Dataset(ncname, "w", parallel=True, comm=T.comm, **kwargs)
        self.T = T
        self.N = T.shape()
        self.names = names
        self._dtype = 'f8'

        self.f.createDimension('t', None)
        self.dims=['t']
        self.nc_t = self.f.createVariable('t', self._dtype, ('t'))
        self.nc_t.set_collective(True)

        x = T.mesh()
        s = self.T.local_slice(False)
        for i, xi in enumerate(x):
            xyz = {0:'x', 1:'y', 2:'z'}[i]
            self.f.createDimension(xyz, np.squeeze(x[i]).size)
            nc_xyz = self.f.createVariable(xyz, self._dtype, (xyz))
            self.dims.append(xyz)
            nc_xyz[s[i]] = np.squeeze(x[i][s[i]])

        self.handles = dict()
        for i,name in enumerate(names):
            self.handles[i] = self.f.createVariable(name, self._dtype, self.dims)
            # switch to collective mode, rewrite the data.
            self.handles[i].set_collective(True)

        self.f.sync()

    def write_tstep(self, tstep, u):
        """Write field u to netcdf format at a given time step

        args:
            tstep        int          Time step
            u      Function/Array     The field to be stored

        """
        assert isinstance(u, np.ndarray)

        # update time
        it = self.nc_t.size
        print(it)
        self.nc_t[it] = tstep

        if isinstance(self.T, MixedTensorProductSpace):
            assert self.T.ndim() == len(u.shape[1:])
            assert len(self.names) == u.shape[0]
            s = self.T.local_slice(False)
            for i in range(u.shape[0]):
                if self.T.ndim() == 3:
                    self.handles[i][it,s[0],s[1],s[2]] = u[i]
                elif self.T.ndim() == 2:
                    self.handles[i][it, s[0], s[1]] = u[i]
                else:
                    raise(NotImplementedError)
        else:
            assert len(self.names) == 1
            s = self.T.local_slice(False)
            if self.T.ndim() == 3:
                self.handles[0][it, s[0], s[1], s[2]] = u[:]
            elif self.T.ndim() == 2:
                self.handles[0][it, s[0], s[1]] = u[:]
            else:
                raise(NotImplementedError)

        self.f.sync()

    def close(self):
        self.f.close()

