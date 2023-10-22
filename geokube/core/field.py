from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any, Self
from warnings import warn

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis, indexes
from .crs import Geodetic
from .domain import Domain, Grid, Points, Profile
from .indexers import get_array_indexer, get_indexer
from .points import to_points_dict
from .quantity import get_magnitude
import pyarrow as pa
from .coord_system import CoordinateSystem

from .feature import PointsFeature


def to_pyarrow_tensor(data):
    # this method return a pyarrow tensor
    # tensor_type = pa.fixed_shape_tensor(pa.int32(), (2, 3))
    # arr = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    # storage = pa.array(arr, pa.list_(pa.int32(), 4))
    # tensor_array = pa.ExtensionArray.from_storage(tensor_type, storage)
    # 
    # arr = self.magnitude # this should be numpy array
    # storage = pa.array(arr, pa.list_(self.patype, self.size))
    # pa.ExtensionArray.from_storage(self._pyarrow_tensor_type(), storage)
    # type is given by tensor.type
    return pa.FixedShapeTensorArray.from_numpy_ndarray(data)

_ARRAY_TYPES = (np.ndarray, da.Array)

_FIELD_NAME_ATTR_ = '_geokube.field_name'

class Field():
    _DOMAIN_CLS_ = type[Domain]

    # TODO: Add cell methods
    # __slots__ = ('_cell_method_',)

    # 
    # - this is the same method name in Feature class ->
    # in order to have precedence Field should be inherited first
    # 
    def _from_xrdset(self, ds: xr.Dataset) -> Self:
        coords = {}        
        for ax in self.coord_system.axes:
            coords[ax] = ds.coords[ax]
        domain = self._DOMAIN_CLS_(
                    coords=coords, 
                    coord_system=self.coord_system)
        name = ds.attrs[_FIELD_NAME_ATTR_]
        data = ds[name].data
        properties = ds[name].attrs
        encoding = ds[name].encoding        
        ancillary = {}
        for c in ds.data_vars:
            if c != name:
                ancillary[c] = ds[c].data
        
        return type(self)(
            name=name,
            domain=domain,
            data=data,
            ancillary=ancillary,
            properties=properties,
            encoding=encoding
        )

    @property # TODO: define setter method
    def domain(self) -> Domain:
        coords = {}
        for ax in self.coord_system.axes:
            coords[ax] = self.coords[ax]
        return self._DOMAIN_CLS_(
                coords=coords, 
                coord_system=self._coord_system)    

    @property
    def ancillary(self, name: str | None = None) -> dict | pint.Quantity:
        if name is not None:
            return self[name].data
        ancillary = {}
        for c in self.data_vars:
            if c != self.name:
                ancillary[c] = self._dset[c].data
        return self.ancillary

    @property # return field name
    def name(self) -> str:
        return self._dset.attrs[_FIELD_NAME_ATTR_]
    
    @property # define data method to return field data
    def data(self):
        return self._dset[self.name].data

    @property
    def properties(self):
        return self._dset[self.name].attrs

    @property
    def encoding(self):
        return self._dset[self.name].encoding

    # Pyarrow conversions -----------------------------------------------------

    def build_pyarrow_metadata(self):
        # geokube Field JSON Schema
        # -> name: ‘string’
        # -> kind: ‘string’ (Gridded, Points, Profile, Timeseries)
        # -> properties: dict
        # -> cf-encoding: dict
        # -> coord_system: dict
        import json
        metadata = {'name': self.name,
                    'kind': str(type(self.domain)),
                    'properties': json.dumps(self.properties).encode('utf-8'),
                    'cf_encoding': json.dumps(self.encoding).encode('utf-8')
        }
        metadata['coord_system'] = {} 
        metadata['coord_system']['horizontal'] = str(self.coord_system.spatial.crs)
        metadata['coord_system']['elevation'] = str(self.domain.coord_system.spatial.elevation)
        metadata['coord_system']['time'] = str(self.domain.coord_system.time)
        metadata['coord_system']['ud_axes'] = []   
        for axis in self.domain.coord_system.user_axes:
             metadata['coord_system']['ud_axes'].append(str(axis))
        metadata['coord_system'] = json.dumps(metadata['coord_system']).encode('utf-8')
        
        return metadata

    def to_pyarrow_table(self): 
        # this method return a pyarrow Table with a schema
        # data contains tensors for the field and domain
        # 
        # schema -> schema for field and domain
        tensor_data = []
        schema_data = []

        field_tensor = to_pyarrow_tensor(self.data.magnitude)
        tensor_data.append(field_tensor)
        schema_data.append(pa.field(self.name, field_tensor.type))

        # Add coordinates to tensor and schema
        for ax, coord in self.domain.coordinates().items():
            coord_tensor = to_pyarrow_tensor(coord.magnitude)
            tensor_data.append(coord_tensor)
            schema_data.append(pa.field(ax, coord_tensor.type))

        # create Table
        return pa.Table.from_arrays(tensor_data, 
                                    schema=pa.schema(schema_data, 
                                                     metadata=self.build_pyarrow_metadata()) 
                                    )


class PointsField(Field, PointsFeature):
    _DOMAIN_CLS_ = Points

    def __init__(
        self,
        name: str,
        domain: Points,
        data: npt.ArrayLike | pint.Quantity | None = None,
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        self._n_points = domain.number_of_points
        match data:
            case pint.Quantity():
                data_ = (
                    data
                    if isinstance(data.magnitude, _ARRAY_TYPES) else
                    pint.Quantity(np.asarray(data.magnitude), data.units)
                )
            case np.ndarray() | da.Array():
                # NOTE: The pattern arr * unit does not work when arr has
                # strings.
                data_ = pint.Quantity(data)
            case None:
                data_ = pint.Quantity(
                    np.full(shape=self._n_points, fill_value=np.nan, dtype=np.float32)
                )
            case _:
                data_ = pint.Quantity(np.asarray(data))
        if data_.shape != (self._n_points,):
            raise ValueError(
                "'data' must have one-dimensional values and the same size as "
                "the coordinates"
            )
        
        data_vars = {}
        attrs = properties if not None else {}
        data_vars[name] = xr.DataArray(data=data_, dims=self._DIMS, attrs=attrs)
        data_vars[name].encoding = encoding if not None else {} # This is not working!!!
        
        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=self._DIMS)
        
        ds_attrs = {_FIELD_NAME_ATTR_: name}

        super().__init__(
            data_vars = data_vars,
            coords = domain.coords,
            attrs = ds_attrs,
            coord_system=domain.coord_system
        )

class ProfileField(Field):
    _DOMAIN_TYPE = Profile

    __slots__ = ()

    def __init__(
        self,
        name: str,
        domain: Profile,
        data: npt.ArrayLike | pint.Quantity | None = None,
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        n_prof, n_lev = domain.number_of_profiles, domain.number_of_levels
        data_shape = (n_prof, n_lev)
        if isinstance(data, Sequence):
            if len(data) != n_prof:
                raise ValueError(
                    "'data' does not contain the same number of profiles as "
                    "the coordinates do"
                )
            all_sizes, all_units, all_data = [], set(), []
            for data_item in data:
                all_sizes.append(len(data_item))
                if isinstance(data_item, pint.Quantity):
                    all_units.add(data_item.units)
                    all_data.append(data_item.magnitude)
                else:
                    all_data.append(data_item)
            if max(all_sizes) > n_lev:
                raise ValueError(
                    "'data' contains more levels than the coordinates do"
                )
            match len(all_units):
                case 0:
                    unit = None
                case 1:
                    unit = all_units.pop()
                case _:
                    # TODO: Consider supporting unit conversion in such cases.
                    raise ValueError("'data' has items with different units")
            data_vals = np.empty(shape=data_shape, dtype=np.float32)
            for i, (stop_idx, vals) in enumerate(zip(all_sizes, all_data)):
                if stop_idx == n_lev:
                    data_vals[i, :] = vals
                else:
                    data_vals[i, :stop_idx] = vals
                    data_vals[i, stop_idx:] = np.nan
            data_ = pint.Quantity(data_vals, unit)
        else:
            match data:
                case pint.Quantity():
                    data_ = (
                        data
                        if isinstance(data.magnitude, _ARRAY_TYPES) else
                        pint.Quantity(np.asarray(data.magnitude), data.units)
                    )
                case np.ndarray() | da.Array():
                    data_ = pint.Quantity(data)
                case None:
                    data_ = pint.Quantity(
                        np.full(
                            shape=data_shape,
                            fill_value=np.nan,
                            dtype=np.float32
                        )
                    )
                case _:
                    data_ = pint.Quantity(np.asarray(data))
            if data_.shape != data_shape:
                raise ValueError(
                    "'data' must be two-dimensional and have the same shape "
                    "as the coordinates"
                )
        
        name = str(name)
        dims = ('_profiles', '_levels')

        _create_feature_dset(name)

        dset = xr.Dataset(
            data_vars={name: (('_profiles', '_levels'), data_)},
            coords=domain._coords
        )
        coord_system = domain.coord_system
        spat_axes = set(coord_system.spatial.axes)
        for axis_ in coord_system.axes:
            if axis_ not in spat_axes:
                dset = dset.set_xindex(axis_, indexes.OneDimIndex)
        dset = dset.set_xindex(
            axis.vertical,
            indexes.TwoDimVertProfileIndex,
            data=dset[name],
            name=name
        )
        dset = dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )
        super().__init__(name, domain, dset, ancillary, properties, encoding)

    # Spatial operations ------------------------------------------------------

    def bounding_box(
        self,
        south: Number | None = None,
        north: Number | None = None,
        west: Number | None = None,
        east: Number | None = None,
        bottom: Number | None = None,
        top: Number | None = None
    ) -> Self:
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        new_data = self._data.sel(h_idx)
        if not (bottom is None and top is None):
            # TODO: Try to move this functionality to
            # `indexes.TwoDimHorPointsIndex.sel`.
            warn(
                "'bounding_box' loads in memory and makes a copy of the data "
                "and vertical coordinate when 'bottom' or 'top' is not 'None'"
            )
            v_slice = slice(bottom, top)
            v_idx = {axis.vertical: v_slice}
            new_data = self._new_field(new_data)._data
            new_data = new_data.sel(v_idx)
            vert = new_data[axis.vertical]
            vert_dims = vert.dims
            vert_data = vert.data
            vert_mag, vert_units = vert_data.magnitude, vert_data.units
            data = new_data[self.name].data
            mask = get_indexer(
                [vert_mag], [get_magnitude(v_slice, vert_units)]
            )[0]
            masked_vert = np.where(mask, vert_mag, np.nan)
            vert_ = pint.Quantity(masked_vert, vert_units)
            new_data[axis.vertical] = xr.Variable(dims=vert_dims, data=vert_)
            masked_data = np.where(mask, data.magnitude, np.nan)
            data_ = pint.Quantity(masked_data, data.units)
            new_data[self.name] = xr.Variable(dims=vert_dims, data=data_)
        return self._new_field(new_data)

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        # TODO: Try to move this functionality to
        # `indexes.TwoDimHorPointsIndex.sel`.
        dset = self._data
        data_qty = dset[self.name].data
        data_mag = data_qty.magnitude
        vert = dset[axis.vertical]
        vert_qty = vert.data
        vert_mag, vert_units = vert_qty.magnitude, vert_qty.units
        dims = vert.dims
        profile_idx = dims.index('_profiles')
        n_profiles = vert_mag.shape[profile_idx]
        kwa = {
            'y_data': [get_magnitude(elevation, vert_units)],
            'return_all': False,
            'method': 'nearest',
            'tolerance': np.inf
        }
        order = slice(None, None, -1 if profile_idx else 1)
        shape = (n_profiles, len(elevation))[order]
        new_data_mag = np.empty(shape=shape, dtype=data_mag.dtype)
        new_vert_mag = np.empty(shape=shape, dtype=vert_mag.dtype)
        # TODO: Try to implement this in a more efficient way.
        for p_idx in range(n_profiles):
            all_l_idx = (p_idx, slice(None))[order]
            l_idx = get_indexer([vert_mag[all_l_idx]], **kwa)
            p_l_idx = (p_idx, l_idx)[order]
            new_data_mag[all_l_idx] = data_mag[p_l_idx]
            new_vert_mag[all_l_idx] = vert_mag[p_l_idx]
        domain = self.domain
        new_coords = self.domain.coordinates().copy()
        new_coords[axis.vertical] = pint.Quantity(new_vert_mag, vert_units)
        return type(self)(
            name=self.name,
            domain=type(domain)(new_coords, domain.coord_system),
            data=pint.Quantity(new_data_mag, data_qty.units)
        )


class GridField(Field):
    # NOTE: The default order of axes is assumed.

    _DOMAIN_TYPE = Grid

    __slots__ = ('__dim_axes',)

    def __init__(
        self,
        name: str,
        domain: Grid,
        data: npt.ArrayLike | pint.Quantity | None = None,
        dim_axes: Sequence[axis.Axis] | None = None,
        anciliary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        coord_system = domain.coord_system
        coords = domain._coords
        crs = coord_system.spatial.crs
        dim_axes_: tuple[axis.Axis, ...]
        aux_axes: tuple[axis.Axis, ...]
        if dim_axes is None:
            if isinstance(crs, Geodetic):
                dim_axes_, aux_axes = coord_system.axes, ()
            else:
                default_axes = coord_system.axes
                aux_hor_axes = {axis.latitude, axis.longitude}
                dim_axes_tmp, aux_axes_tmp = [], []
                for axis_ in default_axes:
                    if axis_ in aux_hor_axes:
                        aux_axes_tmp.append(axis_)
                    else:
                        dim_axes_tmp.append(axis_)
                dim_axes_, aux_axes = tuple(dim_axes_tmp), tuple(aux_axes_tmp)
        else:
            dim_axes_ = tuple(dim_axes)
            aux_axes = tuple(axis for axis in coords if axis not in dim_axes)
        self.__dim_axes = dim_axes_

        match data:
            # case pint.Quantity() if isinstance(data.magnitude, _ARRAY_TYPES):
            #     data_ = data
            # case pint.Quantity():
            #     data_ = pint.Quantity(np.asarray(data.magnitude), data.units)
            case pint.Quantity():
                data_ = (
                    data
                    if isinstance(data.magnitude, _ARRAY_TYPES) else
                    pint.Quantity(np.asarray(data.magnitude), data.units)
                )
            case np.ndarray() | da.Array():
                # NOTE: The pattern arr * unit does not work when arr has
                # stings.
                data_ = pint.Quantity(data)
            case None:
                data_ = None
            case _:
                data_ = pint.Quantity(np.asarray(data))

        dset = xr.Dataset(
            data_vars={
                str(name): (tuple(f'_{axis}' for axis in dim_axes_), data_)
            },
            coords=domain._coords
        )
        for axis_ in dim_axes_:
            dset = dset.set_xindex(axis_, indexes.OneDimPandasIndex)
        if {axis.latitude, axis.longitude} <= set(aux_axes):
            dset = dset.set_xindex(aux_axes, indexes.TwoDimHorGridIndex)
        super().__init__(name, domain, dset, anciliary, properties, encoding)

    def _new_field(
        self, new_data: xr.Dataset, result_type: type[Any] | None = None
    ) -> Any:
        field_type = type(self) if result_type is None else result_type
        domain_type = field_type._DOMAIN_TYPE
        name = self.name
        return field_type(
            name=name,
            domain=domain_type(
                coords={
                    axis_: coord.data
                    for axis_, coord in new_data.coords.items()
                },
                coord_system=self.domain.coord_system
            ),
            data=new_data[name].data,
            dim_axes=self.__dim_axes
        )

    # Spatial operations ------------------------------------------------------

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> PointsField:  # Self:
        # NOTE: This code works with geodetic grids and returns the Cartesian
        # product.
        # idx = {axis.latitude: latitude, axis.longitude: longitude}
        # new_data = self._data.sel(idx, method='nearest', tolerance=np.inf)
        # return self._new_field(new_data)

        # NOTE: This code works with all tested grids and returns the nearest
        # points.
        # Preparing data, labels, units, and dimensions.
        name = self.name
        coord_system = self.domain.coord_system
        dset = self._data
        lat = dset[axis.latitude]
        lat_data = lat.data
        lat_vals = lat_data.magnitude
        lon = dset[axis.longitude]
        lon_data = lon.data
        lon_vals = lon_data.magnitude

        lat_labels = get_magnitude(latitude, lat_data.units)
        lon_labels = get_magnitude(longitude, lon_data.units)

        if isinstance(coord_system.spatial.crs, Geodetic):
            lat_vals, lon_vals = np.meshgrid(lat_vals, lon_vals, indexing='ij')
            dims = ('_latitude', '_longitude')
        else:
            all_dims = {lat.dims, lon.dims}
            if len(all_dims) != 1:
                raise ValueError(
                    "'dset' must contain latitude and longitude with the same"
                    "dimensions for rotated geodetic and projection grids"
                )
            dims = all_dims.pop()

        # Calculating indexers and subsetting.
        idx = get_array_indexer(
            [lat_vals, lon_vals],
            [lat_labels, lon_labels],
            method='nearest',
            tolerance=np.inf,
            return_all=False
        )
        pts_dim = ('_points',)
        pts_idx = [(pts_dim, dim_idx) for dim_idx in idx]
        result_idx = dict(zip(dims, pts_idx))
        dset = dset.isel(indexers=result_idx)

        # Creating the resulting points field.
        new_coords = to_points_dict(name=name, dset=dset)
        del new_coords['points']
        new_data = new_coords.pop(name)

        return PointsField(
            name=name,
            domain=Points(coords=new_coords, coord_system=coord_system),
            data=new_data
        )

    def regrid(
        self, 
        target: GridField | Grid,
        method: str
    ) -> Self:
        
        if not isinstance(target, Domain):
            if isinstance(target, GridField):
                target = target.domain
            else:
                raise TypeError(
                    "'target' must be an instance of Domain or Field"
                )
        #
        # TODO: check if they have the same CRS
        # if source CRS and target CRS are different
        # first transform source CRS to target CRS
        # 
        # get spatial coordinates
        # 
        lat = target.coordinates(axis.latitude)
        lon = target.coordinates(axis.longitude)
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], lat.magnitude, lat.units),
                "lon": (["lon"], lon.magnitude, lon.units)
            }
        )

def from_pyarrow_table(table): 
    # this method return a geokube field starting from a pyarrow Table 
    # the schema metadata contains 
    # data contains tensors for the field and domain
    # 
    # schema -> schema for field and domain
    import json 
    from .coord_system import CoordinateSystem
    from .crs import Geodetic

    metadata = table.schema.metadata
    name = metadata[b'name'].decode()
    feature_type = metadata[b'feature_type'].decode()
    properties = json.loads(metadata[b'properties'].decode())
    encoding = json.loads(metadata[b'cf_encoding'].decode())
    cs = json.loads(metadata[b'coord_system'].decode())

    data = table[name].combine_chunks().to_numpy_ndarray()

    coord_system = CoordinateSystem(
        horizontal=Geodetic(),
        elevation=axis._from_string(cs['elevation']),
        time=axis._from_string(cs['time']),
    )

    coords = {}
    for ax in coord_system.axes:
        coords[ax] = table[ax].combine_chunks().to_numpy_ndarray()
    domain = cls.__DOMAIN_CLS__(coords = coords, coord_system = coord_system)
    return cls(name = name, data = data, domain=domain,properties=properties,encoding=encoding)        
