import warnings
import six
import numpy as np
from shenfun import MixedTensorProductSpace


__all__ = ('ShenfunFile', 'write')

def ShenfunFile(name, T, backend='hdf5', mode='r', **kw):
    from .h5py_file import HDF5File
    from .nc_file import NCFile
    if backend.lower() == 'hdf5':
        return HDF5File(name+'.h5', T, mode=mode, **kw)
    assert kw.get('forward_output', False) is False, "NetCDF4 cannot store complex arrays, use HDF5"
    return NCFile(name+'.nc', T, mode=mode, **kw)

def write(self, step, fields, **kw):
    """Write snapshot ``step`` of ``fields`` to file

    Parameters
    ----------
    step : int
        Index of snapshot.
    fields : dict
        The fields to be dumped to file. (key, value) pairs are group name
        and either arrays or 2-tuples, respectively. The arrays are complete
        arrays to be stored, whereas 2-tuples are arrays with associated
        *global* slices.
    forward_output : bool, optional
        Whether fields to be stored are shaped as the output of a
        forward transform or not. Default is False.
    as_scalar : bool, optional
        Whether to store vectors as scalars. Default is False

    Example
    -------
    >>> from mpi4py import MPI
    >>> from shenfun import TensorProductSpace, HDF5File, Array, Basis
    >>> comm = MPI.COMM_WORLD
    >>> N = (14, 15, 16)
    >>> K0 = Basis(N[0], 'F', dtype='D')
    >>> K1 = Basis(N[1], 'F', dtype='D')
    >>> K2 = Basis(N[2], 'F', dtype='d')
    >>> T = TensorProductSpace(comm, (K0, K1, K2))
    >>> u = Array(T, val=1)
    >>> v = Array(T, val=2)
    >>> f = HDF5File('h5filename.h5', T)
    >>> f.write(0, {'u': [u, (u, [slice(None), 4, slice(None)])],
    ...             'v': [v, (v, [slice(None), 5, 5])]})
    >>> f.write(1, {'u': [u, (u, [slice(None), 4, slice(None)])],
    ...             'v': [v, (v, [slice(None), 5, 5])]})

    This stores data within two main groups ``u`` and ``v``. The HDF5 file
    will in the end contain groups::

        /u/3D/{0, 1}
        /u/2D/slice_4_slice/{0, 1}
        /v/3D/{0, 1}
        /v/1D/slice_5_5/{0, 1}

    Note
    ----
    The list of slices used in storing only parts of the arrays are views
    of the *global* arrays.

    """
    for group, list_of_fields in six.iteritems(fields):
        assert isinstance(list_of_fields, (tuple, list))
        assert isinstance(group, str)

        for field in list_of_fields:
            if isinstance(field, np.ndarray):
                if self.T.rank() > 1:
                    for k in range(field.shape[0]):
                        g = group + str(k)
                        self._write_group(g, field[k], step, **kw)
                else:
                    self._write_group(group, field, step, **kw)

            else:
                assert len(field) == 2
                u, sl = field
                if self.T.rank() > 1 and sl[0] == slice(None):
                    kw['slice_as_scalar'] = True
                    for k in range(u.shape[0]):
                        g = group + str(k)
                        self._write_slice_step(g, step, sl[1:], u[k], **kw)
                else:
                    kw['slice_as_scalar'] = False
                    self._write_slice_step(group, step, sl, u, **kw)
