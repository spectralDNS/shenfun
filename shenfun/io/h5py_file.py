#pylint: disable=missing-docstring, consider-using-enumerate
import numpy as np
from mpi4py_fft.io import HDF5File as BaseFile

__all__ = ('HDF5File',)

# Reimplement here because of allocated_shape vs global_shape
class HDF5File(BaseFile):
    def __init__(self, h5name, domain=None, mode='r', **kw):
        BaseFile.__init__(self, h5name, domain=domain, mode=mode, **kw)

    def _write_group(self, name, u, step, **kw):
        s = u.local_slice()
        group = "/".join((name, "{}D".format(u.dimensions)))
        if group not in self.f:
            self.f.create_group(group)
        self.f[group].require_dataset(str(step), shape=u.allocated_shape, dtype=u.dtype)
        self.f["/".join((group, str(step)))][s] = u

    def _write_slice_step(self, name, step, slices, field, **kw):
        rank = field.rank
        slices = (slice(None),)*rank + tuple(slices)
        slices = list(slices)
        ndims = slices[rank:].count(slice(None))
        slname = self._get_slice_name(slices[rank:])
        s = field.local_slice()
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = tuple(np.take(s, sp))
        sl = tuple(slices)
        group = "/".join((name, "{}D".format(ndims), slname))
        if group not in self.f:
            self.f.create_group(group)
        N = field.allocated_shape
        self.f[group].require_dataset(str(step), shape=tuple(np.take(N, sp)), dtype=field.dtype)
        if inside == 1:
            self.f["/".join((group, str(step)))][sf] = field[sl]
