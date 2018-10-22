import six
import numpy as np

__all__ = ('ShenfunFile', 'write')

def ShenfunFile(name, T, backend='hdf5', mode='r', **kw):
    """Return a file handler

    Parameters
    ----------
    name : str
        Name of file, without ending
    T : TensorProductSpace
        The space used for the data stored. Can also be MixedTensorProductSpace
        or VectorTensorProductSpace.
    backend : str, optional
        ``hdf5`` or ``netcdf``. Default is ``hdf5``.
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

def write(self, step, fields, **kw):
    # Note - Overloaded write
    # Should enter here with T.rank() > 1 and kw['as_scalar'] == True.
    # However, a slice may already be a scalar if the vector component is
    # indexed and not sliced, e.g., [2, slice(None), slice(None), 4]. In
    # that case we need to let _write_slice_step know.
    assert self.T.rank() > 1
    assert kw.get('as_scalar', False) is True
    for group, list_of_fields in six.iteritems(fields):
        assert isinstance(list_of_fields, (tuple, list))
        assert isinstance(group, str)

        for field in list_of_fields:
            if isinstance(field, np.ndarray):
                for k in range(field.shape[0]):
                    g = group + str(k)
                    self._write_group(g, field[k], step, **kw)

            else:
                assert len(field) == 2
                u, sl = field
                if sl[0] == slice(None):
                    kw['slice_as_scalar'] = True
                    for k in range(u.shape[0]):
                        g = group + str(k)
                        self._write_slice_step(g, step, sl[1:], u[k], **kw)
                else:
                    kw['slice_as_scalar'] = False
                    self._write_slice_step(group, step, sl, u, **kw)
