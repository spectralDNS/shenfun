# pylint: disable=line-too-long
import copy
import re
from numpy import dtype, array, invert, take

__all__ = ('generate_xdmf',)

xdmffile = """<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Structured Grid" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List"><DataItem Format="XML" Dimensions="{1}"> {0} </DataItem></Time>
      {2}
    </Grid>
  </Domain>
</Xdmf>
"""

def get_grid(geometry, topology, attrs):
    return """<Grid GridType="Uniform">
        {0}
        {1}
        {2}
      </Grid>
      """.format(geometry, topology, attrs)

def get_geometry(kind=0, dim=2):
    assert kind in (0, 1)
    assert dim in (2, 3)
    if dim == 2:
        if kind == 0:
            return """<Geometry Type="ORIGIN_DXDY">
          <DataItem Format="XML" NumberType="Float" Dimensions="2">
            {0} {1}
          </DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="2">
            {2} {3}
          </DataItem>
        </Geometry>"""

        return """<Geometry Type="VXVYVZ">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
            {3}:{6}/mesh/{4}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
            {3}:{6}/mesh/{5}
          </DataItem>
          <DataItem Format="XML" NumberType="Float" Precision="8" Dimensions="1">
            0
          </DataItem>
        </Geometry>"""

    if dim == 3:
        if kind == 0:
            return """<Geometry Type="ORIGIN_DXDYDZ">
          <DataItem Format="XML" NumberType="Float" Dimensions="3">
            {0} {1} {2}
          </DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="3">
            {3} {4} {5}
          </DataItem>
        </Geometry>"""

        return """<Geometry Type="VXVYVZ">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{3}">
            {4}:{8}/mesh/{5}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
            {4}:{8}/mesh/{6}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
            {4}:{8}/mesh/{7}
          </DataItem>
        </Geometry>"""

def get_topology(dims, kind=0):
    assert len(dims) in (2, 3)
    co = 'Co' if kind == 0 else ''
    if len(dims) == 2:
        return """<Topology Dimensions="1 {0} {1}" Type="3D{2}RectMesh"/>""".format(dims[0], dims[1], co)
    if len(dims) == 3:
        return """<Topology Dimensions="{0} {1} {2}" Type="3D{3}RectMesh"/>""".format(dims[0], dims[1], dims[2], co)

def get_attribute(attr, h5filename, dims, prec):
    name = attr.split("/")[0]
    assert len(dims) in (2, 3)
    if len(dims) == 2:
        return """<Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{5}" Dimensions="1 {1} {2}">
            {3}:/{4}
          </DataItem>
        </Attribute>
        """.format(name, dims[0], dims[1], h5filename, attr, prec)

    return """<Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{6}" Dimensions="{1} {2} {3}">
            {4}:/{5}
          </DataItem>
        </Attribute>
        """.format(name, dims[0], dims[1], dims[2], h5filename, attr, prec)

def generate_xdmf(h5filename, periodic=True, order='visit'):
    """Generate XDMF-files

    Parameters
    ----------
    h5filename : str
        Name of hdf5-file that we want to decorate with xdmf
    periodic : bool or dim-sequence of bools, optional
        If true along axis i, assume data is periodic.
        Only affects the calculation of the domain size and only if the
        domain is given as 2-tuple of origin+dx.
    order : str
        ``paraview`` or ``visit``
        For some reason Paraview and Visit requires the mesh stored in
        opposite order in the XDMF-file for 2D slices. Make choice of
        order here.
    """
    import h5py
    f = h5py.File(h5filename, 'a')
    keys = []
    f.visit(keys.append)
    assert order.lower() in ('paraview', 'visit')

    # Find unique scalar groups of 2D and 3D datasets
    datasets = {2:{}, 3:{}}
    for key in keys:
        if f[key.split('/')[0]].attrs['rank'] > 0:
            continue
        if isinstance(f[key], h5py.Dataset):
            if not ('mesh' in key or 'domain' in key or 'Vector' in key):
                tstep = int(key.split("/")[-1])
                ndim = int(key.split("/")[1][0])
                if ndim in (2, 3):
                    ds = datasets[ndim]
                    if tstep in ds:
                        ds[tstep] += [key]
                    else:
                        ds[tstep] = [key]
    if periodic is True:
        periodic = [0]*5
    elif periodic is False:
        periodic = [1]*5
    else:
        assert isinstance(periodic, (tuple, list))
        periodic = list(array(invert(periodic), int))

    coor = ['x0', 'x1', 'x2', 'x3', 'x4']
    for ndim, dsets in datasets.items():
        timesteps = list(dsets.keys())
        per = copy.copy(periodic)
        if not timesteps:
            continue
        timesteps.sort(key=int)
        tt = ""
        for i in timesteps:
            tt += "%s " %i
        datatype = f[dsets[timesteps[0]][0]].dtype
        assert datatype.char not in 'FDG', "Cannot use generate_xdmf to visualize complex data."
        prec = 4 if datatype is dtype('float32') else 8
        xff = {}
        geometry = {}
        topology = {}
        attrs = {}
        grid = {}
        NN = {}
        for name in dsets[timesteps[0]]:
            group = name.split('/')[0]
            if 'slice' in name:
                slices = name.split("/")[2]
            else:
                slices = 'whole'
            cc = copy.copy(coor)
            if slices not in xff:
                xff[slices] = copy.copy(xdmffile)
                N = list(f[name].shape)
                kk = 0
                sl = 0
                if 'slice' in slices:
                    ss = slices.split("_")
                    ii = []
                    for i, sx in enumerate(ss):
                        if 'slice' in sx:
                            ii.append(i)
                        else:
                            if len(f[group].attrs.get('shape')) == 3:      # 2D slice in 3D domain
                                kk = i
                                sl = int(sx)
                                N.insert(i, 1)
                    cc = take(coor, ii)
                else:
                    ii = list(range(ndim))
                NN[slices] = N
                if 'domain' in f[group].keys():
                    if ndim == 2 and ('slice' not in slices or len(f[group].attrs.get('shape')) > 3):
                        geo = get_geometry(kind=0, dim=2)
                        assert len(ii) == 2
                        i, j = ii
                        if order.lower() == 'paraview':
                            data = [f[group+'/domain/{}'.format(coor[i])][0],
                                    f[group+'/domain/{}'.format(coor[j])][0],
                                    f[group+'/domain/{}'.format(coor[i])][1]/(N[0]-per[i]),
                                    f[group+'/domain/{}'.format(coor[j])][1]/(N[1]-per[j])]
                            geometry[slices] = geo.format(*data)
                        else:
                            data = [f[group+'/domain/{}'.format(coor[j])][0],
                                    f[group+'/domain/{}'.format(coor[i])][0],
                                    f[group+'/domain/{}'.format(coor[j])][1]/(N[0]-per[j]),
                                    f[group+'/domain/{}'.format(coor[i])][1]/(N[1]-per[i])]
                            geometry[slices] = geo.format(*data)
                    else:
                        if ndim == 2:
                            ii.insert(kk, kk)
                            per[kk] = 0
                        i, j, k = ii
                        geo = get_geometry(kind=0, dim=3)
                        data = [f[group+'/domain/{}'.format(coor[i])][0],
                                f[group+'/domain/{}'.format(coor[j])][0],
                                f[group+'/domain/{}'.format(coor[k])][0],
                                f[group+'/domain/{}'.format(coor[i])][1]/(N[0]-per[i]),
                                f[group+'/domain/{}'.format(coor[j])][1]/(N[1]-per[j]),
                                f[group+'/domain/{}'.format(coor[k])][1]/(N[2]-per[k])]
                        if ndim == 2:
                            origin, dx = f[group+'/domain/x{}'.format(kk)]
                            M = f[group].attrs.get('shape')
                            pos = origin+dx/(M[kk]-per[kk])*sl
                            data[kk] = pos
                            data[kk+3] = pos
                        geometry[slices] = geo.format(*data)
                    topology[slices] = get_topology(N, kind=0)
                elif 'mesh' in f[group].keys():
                    if ndim == 2 and ('slice' not in slices or len(f[group].attrs.get('shape')) > 3):
                        geo = get_geometry(kind=1, dim=2)
                    else:
                        geo = get_geometry(kind=1, dim=3)

                    if ndim == 2 and ('slice' not in slices or len(f[group].attrs.get('shape')) > 3):
                        if order.lower() == 'paraview':
                            sig = (prec, N[0], N[1], h5filename, cc[0], cc[1], group)
                        else:
                            sig = (prec, N[1], N[0], h5filename, cc[1], cc[0], group)
                    else:
                        if ndim == 2: # 2D slice in 3D domain
                            pos = f[group+"/mesh/x{}".format(kk)][sl]
                            z = re.findall(r'<DataItem(.*?)</DataItem>', geo, re.DOTALL)
                            geo = geo.replace(z[2-kk], ' Format="XML" NumberType="Float" Precision="{0}" Dimensions="{%d}">\n           {%d}\n          '%(1+kk, 7-kk))
                            cc = list(cc)
                            cc.insert(kk, pos)
                        sig = (prec, N[0], N[1], N[2], h5filename, cc[2], cc[1], cc[0], group)
                    geometry[slices] = geo.format(*sig)
                    topology[slices] = get_topology(N, kind=1)
                grid[slices] = ''

        # if slice of data, need to know along which axes
        # Since there may be many different slices, we need to create
        # one xdmf-file for each composition of slices
        attrs = {}
        for tstep in timesteps:
            d = dsets[tstep]
            slx = set()
            for i, x in enumerate(d):
                slices = x.split("/")[2]
                if not 'slice' in slices:
                    slices = 'whole'
                N = NN[slices]
                if slices not in attrs:
                    attrs[slices] = ''
                attrs[slices] += get_attribute(x, h5filename, N, prec)
                slx.add(slices)
            for slices in slx:
                grid[slices] += get_grid(geometry[slices], topology[slices],
                                         attrs[slices].rstrip())
                attrs[slices] = ''
        for slices, ff in xff.items():
            if 'slice' in slices:
                fname = h5filename[:-3]+"_"+slices+".xdmf"
            else:
                fname = h5filename[:-3]+".xdmf"
            xfl = open(fname, "w")
            h = ff.format(tt, len(timesteps), grid[slices].rstrip())
            xfl.write(h)
            xfl.close()
    f.close()

if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])
