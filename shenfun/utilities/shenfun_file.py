import six
import numpy as np

__all__ = ('ShenfunFile',)

def ShenfunFile(name, T, backend='hdf5', mode='r', **kw):
    """Return a file handler

    Parameters
    ----------
    name : str
        Name of file, without ending
    T : :class:`.TensorProductSpace`
        The space used for the data stored. Can also be
        :class:`.MixedTensorProductSpace` or
        :class:`.VectorTensorProductSpace`.
    backend : str, optional
        ``hdf5`` or ``netcdf4``. Default is ``hdf5``.
    mode : str, optional
        ``r`` or ``w``. Default is ``r``.

    Returns
    -------
    Class instance
        Instance of either :class:`.HDF5File` or :class:`.NCFile`
    """

    from .h5py_file import HDF5File
    from .nc_file import NCFile
    if backend.lower() == 'hdf5':
        return HDF5File(name+'.h5', T, mode=mode, **kw)
    assert kw.get('forward_output', False) is False, "NetCDF4 cannot store complex arrays, use HDF5"
    return NCFile(name+'.nc', T, mode=mode, **kw)

#def write_vector(self, step, fields, **kw):
#    for group, list_of_fields in six.iteritems(fields):
#        assert isinstance(list_of_fields, (tuple, list))
#        assert isinstance(group, str)
#
#        for field in list_of_fields:
#            if self.T.rank() == 1:
#                if isinstance(field, np.ndarray):
#                    g = group
#                    if self.backend() == 'hdf5':
#                        g = "/".join((group, "{}D".format(self.T.ndim())))
#                    self._write_group(g, field, step, **kw)
#                else:
#                    assert len(field) == 2
#                    u, sl = field
#                    ndims = sl.count(slice(None))
#                    slname = self._get_slice_name(sl)
#                    if self.backend() == 'hdf5':
#                        g = "/".join((group, "{}D".format(ndims), slname))
#                    else:
#                        g = "_".join((group, slname))
#                    self._write_slice_step(g, step, sl, u, **kw)
#            else:
#                if kw.get('as_scalar', False):
#                    if isinstance(field, np.ndarray):
#                        if len(self.T.shape()) == len(field.shape):  # A regular vector array
#                            for k in range(field.shape[0]):
#                                g = group + str(k)
#                                if self.backend() == 'hdf5':
#                                    g = "/".join((g, "{}D".format(self.T.ndim())))
#                                self._write_group(g, field[k], step, **kw)
#                        elif len(self.T.shape()) == len(field.shape)+1: # A scalar in the vector space
#                            g = group
#                            if self.backend() == 'hdf5':
#                                g = "/".join((group, "{}D".format(self.T.ndim())))
#                            self._write_group(g, field, step, **kw)
#                    else:
#                        assert len(field) == 2
#                        u, sl = field
#                        ndims = sl[1:].count(slice(None))
#                        if sl[0] == slice(None):
#                            for k in range(u.shape[0]):
#                                g = group + str(k)
#                                slname = self._get_slice_name(sl[1:])
#                                if self.backend() == 'hdf5':
#                                    g = "/".join((g, "{}D".format(ndims), slname))
#                                else:
#                                    g = "_".join((g, slname))
#                                self._write_slice_step(g, step, sl[1:], u[k], **kw)
#                        else:
#                            g = group + str(sl[0])
#                            slname = self._get_slice_name(sl[1:])
#                            if self.backend() == 'hdf5':
#                                g = "/".join((g, "{}D".format(ndims), slname))
#                            else:
#                                g = "_".join((g, slname))
#                            self._write_slice_step(g, step, sl, u, **kw)
#
#                else:  # not as_scalar
#                    if isinstance(field, np.ndarray):
#                        g = group
#                        if self.backend() == 'hdf5':
#                            if len(self.T.shape()) == len(field.shape):  # A regular vector array
#                                g = "/".join((group, "{}D".format(self.T.ndim())+"_Vector"))
#                            elif len(self.T.shape()) == len(field.shape)+1: # A scalar in the vector space
#                                g = "/".join((group, "{}D".format(self.T.ndim())))
#                        self._write_group(g, field, step, **kw)
#                    else:
#                        assert len(field) == 2
#                        u, sl = field
#                        ndims = sl[1:].count(slice(None))
#                        slname = self._get_slice_name(sl[1:])
#                        if sl[0] == slice(None):
#                            g = group
#                            if self.backend() == 'hdf5':
#                                g = "/".join((group, "{}D".format(ndims)+"_Vector", slname))
#                            else:
#                                g = "_".join((group, slname))
#                            self._write_slice_step(g, step, sl, u, **kw)
#                        else:
#                            g = group + str(sl[0])
#                            if self.backend() == 'hdf5':
#                                g = "/".join((g, "{}D".format(ndims), slname))
#                            else:
#                                g = "_".join((g, slname))
#                            self._write_slice_step(g, step, sl, u, **kw)
#