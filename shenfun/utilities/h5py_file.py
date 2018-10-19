#pylint: disable=missing-docstring, consider-using-enumerate
import warnings
import numpy as np
from mpi4py_fft.utilities import HDF5File as BaseFile
from .shenfun_file import write

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
        as_scalar = kw.get('as_scalar', False)
        if self.T.rank() == 1 or (not as_scalar):
            BaseFile.write(self, step, fields, **kw)
        else:
            write(self, step, fields, **kw)

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
            group += '_Vector'
        dset = "/".join((name, group, str(step)))
        u[:] = self.f[dset][tuple(s)]

    def _write_group(self, name, u, step, **kw):
        as_scalar = kw.get('as_scalar', False)
        forward_output = kw.get('forward_output', False)
        T = self.T if not as_scalar else self.T[0]
        s = tuple(T.local_slice(forward_output))
        group = "/".join((name, "{}D".format(T.ndim())))
        if T.rank() == 2:
            group = group + "_Vector"
        if group not in self.f:
            self.f.create_group(group)
        self.f[group].create_dataset(str(step), shape=T.shape(forward_output), dtype=u.dtype)
        self.f["/".join((group, str(step)))][s] = u

    def _write_slice_step(self, name, step, slices, field, **kw):
        as_scalar = kw.get('slice_as_scalar', False)
        forward_output = kw.get('forward_output', False)
        slices = list(slices)
        T = self.T if not as_scalar else self.T[0]
        if T.rank() == 2:
            ndims = slices[1:].count(slice(None))
            slname = self._get_slice_name(slices[1:])
        else:
            ndims = slices.count(slice(None))
            slname = self._get_slice_name(slices)
        s = T.local_slice(forward_output)
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = tuple(np.take(s, sp))
        sl = tuple(slices)
        gdim = "{}D".format(ndims)
        if T.rank() > 1 and slices[0] == slice(None):
            gdim += "_Vector"
        group = "/".join((name, gdim, slname))
        if group not in self.f:
            self.f.create_group(group)
        N = T.shape(forward_output)
        self.f[group].create_dataset(str(step), shape=np.take(N, sp), dtype=field.dtype)
        if inside == 1:
            self.f["/".join((group, str(step)))][sf] = field[sl]
