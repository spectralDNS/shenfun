#pylint: disable=missing-docstring,consider-using-enumerate
import copy
import numpy as np
from mpi4py_fft.utilities import NCFile as BaseFile

# https://github.com/Unidata/netcdf4-python/blob/master/examples/mpi_example.py

__all__ = ('NCFile',)


class NCFile(BaseFile):
    """Class for reading/writing data using the netCDF4 format

    Parameters
    ----------
        ncname : str
                 Name of netcdf file to be created
        T : TensorProductSpace
            Instance of a :class:`.TensorProductSpace`. Can also be a
            :class:`.MixedTensorProductSpace`.
        mode : str, optional
            ``r`` or ``w`` for read or write. Default is ``r``.
        clobber : bool, optional
    """
    def __init__(self, ncname, T, mode='r', clobber=True, **kw):
        BaseFile.__init__(self, ncname, T, domain=T.mesh(), clobber=clobber, mode=mode, **kw)
        if T.rank() == 1 and mode == 'w':
            self.open()
            self.vdims = copy.copy(self.dims)
            self.f.createDimension('dim', T.num_components())
            d = self.f.createVariable('dim', int, ('dim'))
            d[:] = np.arange(T.num_components())
            self.vdims.insert(1, 'dim')
            self.close()

    def write(self, step, fields, **kw):
        """Write snapshot ``step`` of ``fields`` to netCDF4 file

        Parameters
        ----------
        step : int
            Index of snapshot.
        fields : dict
            The fields to be dumped to file. (key, value) pairs are group name
            and either arrays or 2-tuples, respectively. The arrays are complete
            arrays to be stored, whereas 2-tuples are arrays with associated
            *global* slices.
        as_scalar : bool, optional
            Whether to store vectors as scalars. Default is False.

        Example
        -------
        >>> from mpi4py import MPI
        >>> import numpy as np
        >>> from shenfun import TensorProductSpace, Array, Basis, NCFile
        >>> comm = MPI.COMM_WORLD
        >>> N = (24, 25, 26)
        >>> K0 = Basis(N[0], 'F', dtype='D')
        >>> K1 = Basis(N[1], 'F', dtype='D')
        >>> K2 = Basis(N[2], 'F', dtype='d')
        >>> T = TensorProductSpace(comm, (K0, K1, K2))
        >>> fl = NCFile('ncfile.nc', T, mode='w')
        >>> u = Array(T)
        >>> u[:] = np.random.random(T.forward.input_array.shape)
        >>> d = {'u': [u, (u, np.s_[4, :, :]), (u, np.s_[4, 4, :])]}
        >>> fl.write(0, d)
        >>> u[:] = 2
        >>> fl.write(1, d)

        The resulting NetCDF4 file ``ncfile.nc`` can be viewed using
        ``ncdump -h ncfile.nc``::

            netcdf ncfile {
            dimensions:
                    time = UNLIMITED ; // (2 currently)
                    x = 24 ;
                    y = 25 ;
                    z = 26 ;
            variables:
                    double time(time) ;
                    double x(x) ;
                    double y(y) ;
                    double z(z) ;
                    double u(time, x, y, z) ;
                    double u_4_slice_slice(time, y, z) ;
                    double u_4_4_slice(time, z) ;

            // global attributes:
                            :ndim = 3LL ;
                            :shape = 24LL, 25LL, 26LL ;
            }

        """
        self.open()
        nc_t = self.f.variables.get('time')
        nc_t.set_collective(True)
        it = nc_t.size
        nc_t[it] = step
        step = it
        for group, list_of_fields in fields.items():
            assert isinstance(list_of_fields, (tuple, list))
            assert isinstance(group, str)

            for field in list_of_fields:
                if self.T.rank() == 0:
                    if isinstance(field, np.ndarray):
                        g = group
                        self._write_group(g, field, step, **kw)
                    else:
                        assert len(field) == 2
                        u, sl = field
                        slname = self._get_slice_name(sl)
                        g = "_".join((group, slname))
                        self._write_slice_step(g, step, sl, u, **kw)
                else:
                    if kw.get('as_scalar', False):
                        if isinstance(field, np.ndarray):
                            if len(self.T.shape()) == len(field.shape):  # A regular vector array
                                for k in range(field.shape[0]):
                                    g = group + str(k)
                                    self._write_group(g, field[k], step, **kw)
                            elif len(self.T.shape()) == len(field.shape)+1: # A scalar in the vector space
                                g = group
                                self._write_group(g, field, step, **kw)
                        else:
                            assert len(field) == 2
                            u, sl = field
                            if sl[0] == slice(None):
                                for k in range(u.shape[0]):
                                    g = group + str(k)
                                    slname = self._get_slice_name(sl[1:])
                                    g = "_".join((g, slname))
                                    self._write_slice_step(g, step, sl[1:], u[k], **kw)
                            else:
                                g = group + str(sl[0])
                                slname = self._get_slice_name(sl[1:])
                                g = "_".join((g, slname))
                                self._write_slice_step(g, step, sl, u, **kw)

                    else:  # not as_scalar
                        if isinstance(field, np.ndarray):
                            self._write_group(group, field, step, **kw)
                        else:
                            assert len(field) == 2
                            u, sl = field
                            slname = self._get_slice_name(sl[1:])
                            if sl[0] == slice(None):
                                g = "_".join((group, slname))
                                self._write_slice_step(g, step, sl, u, **kw)
                            else:
                                g = group + str(sl[0])
                                g = "_".join((g, slname))
                                self._write_slice_step(g, step, sl, u, **kw)
        self.close()

    def _write_group(self, name, u, step, **kw):
        T = u.function_space()
        s = T.local_slice(False)
        dims = self.dims if T.rank() == 0 else self.vdims
        if name not in self.f.variables:
            h = self.f.createVariable(name, self._dtype, dims)
        else:
            h = self.f.variables[name]
        h.set_collective(True)
        s = (step,) + s
        h[s] = u
        self.f.sync()

    def _write_slice_step(self, name, step, slices, field, **kw):
        slices = list(slices)
        T = field.function_space()
        dims = self.dims if T.rank() == 0 else self.vdims
        s = T.local_slice(False)
        slices, inside = self._get_local_slices(slices, s)
        sp = np.nonzero([isinstance(x, slice) for x in slices])[0]
        sf = np.take(s, sp)
        sdims = ['time'] + list(np.take(dims, np.array(sp)+1))
        if name not in self.f.variables:
            h = self.f.createVariable(name, self._dtype, sdims)
        else:
            h = self.f.variables[name]
        h.set_collective(True)
        h[step] = 0 # collectively create dataset
        h.set_collective(False)
        sf = tuple([step] + list(sf))
        sl = tuple(slices)
        if inside:
            h[sf] = field[sl]
        h.set_collective(True)
        self.f.sync()
