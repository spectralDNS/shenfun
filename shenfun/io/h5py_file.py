#pylint: disable=missing-docstring, consider-using-enumerate
import six
import numpy as np
from mpi4py_fft.utilities import HDF5File as BaseFile

__all__ = ('HDF5File',)


class HDF5File(BaseFile):
    """Class for reading/writing data using the HDF5 format

    Parameters
    ----------
        h5name : str
            Name of hdf5 file to be created
        T : TensorProductSpace
            Instance of a :class:`.TensorProductSpace`. Can also be a
            :class:`.MixedTensorProductSpace`.
        mode : str, optional
            ``r`` or ``w`` for read or write. Default is ``r``.
    """
    def __init__(self, h5name, T, mode='r', **kw):
        BaseFile.__init__(self, h5name, T, domain=T.mesh(), mode=mode, **kw)

    def write(self, step, fields, **kw):
        """Write snapshot ``step`` of ``fields`` to HDF5 file

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
            Whether to store vectors as scalars. Should be used if one wants to
            visualize vectors, because XDMF cannot be used with C-type arrays.
            Default is False

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
        >>> f = HDF5File('h5filename.h5', T, mode='w')
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
                if self.T.rank() == 1:
                    if isinstance(field, np.ndarray):
                        g = "/".join((group, "{}D".format(self.T.ndim())))
                        self._write_group(g, field, step, **kw)
                    else:
                        assert len(field) == 2
                        u, sl = field
                        ndims = sl.count(slice(None))
                        slname = self._get_slice_name(sl)
                        g = "/".join((group, "{}D".format(ndims), slname))
                        self._write_slice_step(g, step, sl, u, **kw)
                else:
                    if kw.get('as_scalar', False):
                        if isinstance(field, np.ndarray):
                            if len(self.T.shape()) == len(field.shape):  # A regular vector array
                                for k in range(field.shape[0]):
                                    g = group + str(k)
                                    g = "/".join((g, "{}D".format(self.T.ndim())))
                                    self._write_group(g, field[k], step, **kw)
                            elif len(self.T.shape()) == len(field.shape)+1: # A scalar in the vector space
                                g = "/".join((group, "{}D".format(self.T.ndim())))
                                self._write_group(g, field, step, **kw)
                        else:
                            assert len(field) == 2
                            u, sl = field
                            ndims = sl[1:].count(slice(None))
                            if sl[0] == slice(None):
                                for k in range(u.shape[0]):
                                    g = group + str(k)
                                    slname = self._get_slice_name(sl[1:])
                                    g = "/".join((g, "{}D".format(ndims), slname))
                                    self._write_slice_step(g, step, sl[1:], u[k], **kw)
                            else:
                                g = group + str(sl[0])
                                slname = self._get_slice_name(sl[1:])
                                g = "/".join((g, "{}D".format(ndims), slname))
                                self._write_slice_step(g, step, sl, u, **kw)

                    else:  # not as_scalar
                        if isinstance(field, np.ndarray):
                            if len(self.T.shape()) == len(field.shape):  # A regular vector array
                                g = "/".join((group, "Vector", "{}D".format(self.T.ndim())))
                            elif len(self.T.shape()) == len(field.shape)+1: # A scalar in the vector space
                                g = "/".join((group, "{}D".format(self.T.ndim())))
                            self._write_group(g, field, step, **kw)
                        else:
                            assert len(field) == 2
                            u, sl = field
                            ndims = sl[1:].count(slice(None))
                            slname = self._get_slice_name(sl[1:])
                            if sl[0] == slice(None):
                                g = "/".join((group, "Vector", "{}D".format(ndims), slname))
                                self._write_slice_step(g, step, sl, u, **kw)
                            else:
                                g = group + str(sl[0])
                                g = "/".join((g, "{}D".format(ndims), slname))
                                self._write_slice_step(g, step, sl, u, **kw)

    def read(self, u, name, **kw):
        """Read into array ``u``

        Parameters
        ----------
        u : array
            The array to read into.
        name : str
            Name of array to be read.
        forward_output : bool, optional
            Whether the array to be read is the output of a forward transform
            or not. Default is False.
        step : int, optional
            Index of field to be read. Default is 0.
        """
        forward_output = kw.get('forward_output', False)
        step = kw.get('step', 0)
        s = self.T.local_slice(forward_output)
        ndim = self.T.ndim()
        group = "{}D".format(ndim)
        if self.T.rank() == 2:
            name += '/Vector'
        dset = "/".join((name, group, str(step)))
        u[:] = self.f[dset][tuple(s)]

    def _write_group(self, name, u, step, **kw):
        T = u.function_space()
        forward_output = kw.get('forward_output', False)
        s = tuple(T.local_slice(forward_output))
        if name not in self.f:
            self.f.create_group(name)
        self.f[name].create_dataset(str(step), shape=T.shape(forward_output), dtype=u.dtype)
        self.f["/".join((name, str(step)))][s] = u

    def _write_slice_step(self, name, step, slices, field, **kw):
        forward_output = kw.get('forward_output', False)
        slices = list(slices)
        T = field.function_space()
        s = T.local_slice(forward_output)
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = tuple(np.take(s, sp))
        sl = tuple(slices)
        if name not in self.f:
            self.f.create_group(name)
        N = T.shape(forward_output)
        self.f[name].create_dataset(str(step), shape=np.take(N, sp), dtype=field.dtype)
        if inside == 1:
            self.f["/".join((name, str(step)))][sf] = field[sl]
