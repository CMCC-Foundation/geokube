from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import Any, Self

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis, indexes
from .crs import Geodetic
from .domain import Domain, Grid, Points, Profiles
from .indexers import get_array_indexer, get_indexer
from .points import to_points_dict
from .quantity import get_magnitude
import pyarrow as pa
from .coord_system import CoordinateSystem

from .feature import PointsFeature, ProfilesFeature, GridFeature


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
    __slots__ = ()
    _DOMAIN_CLS_: type[Domain]

    # TODO: Add cell methods
    # __slots__ = ('_cell_method_',)

    # 
    # - this is the same method name in Feature class ->
    # in order to have precedence Field should be inherited first
    # 
    @classmethod
    def _from_xrdset(cls, 
                     ds: xr.Dataset, 
                     coord_system: CoordinateSystem) -> Self:
        coords = {}        
        for ax in coord_system.axes:
            coords[ax] = ds.coords[ax]
        domain = cls._DOMAIN_CLS_(
                    coords=coords, 
                    coord_system=coord_system)
        name = ds.attrs[_FIELD_NAME_ATTR_]
        data = ds[name].data
        properties = ds[name].attrs
        encoding = ds[name].encoding
        if 'ancillary_variables' in ds[name].attrs:
            ancillary = {}
            for c in ds[name].attrs['ancillary_variables'].split():
                    ancillary[c] = ds[c].data
        else:
            ancillary = None
        
        return cls(
            name=name,
            domain=domain,
            data=data,
            ancillary=ancillary,
            properties=properties,
            encoding=encoding
        )

    # def _prepare_dset(self,
    #                   name,
    #                   data, 
    #                   domain,
    #                   ancillary,
    #                   encoding: Mapping | None = None,
    #                   properties: Mapping | None = None):
    #     data_vars = {}
    #     attrs = properties if not None else {}
    #     data_vars[name] = xr.DataArray(data=data, dims=self._DIMS_, attrs=attrs)
    #     data_vars[name].encoding = encoding if not None else {} # This is not working!!!
        
    #     if ancillary is not None:
    #         for anc_name, anc_data in ancillary.items():
    #             data_vars[anc_name] = xr.DataArray(data=anc_data, dims=self._DIMS)
        
    #     ds_attrs = {_FIELD_NAME_ATTR_: name}

    #     self.super().__init__(
    #         data_vars = data_vars,
    #         coords = domain.coords,
    #         attrs = ds_attrs,
    #         coord_system=domain.coord_system
    #     )

    @property # TODO: define setter method
    def domain(self) -> Domain:
        coords = {}
        for ax in self.coord_system.axes:
            coords[ax] = self.coords[ax]
        return self._DOMAIN_CLS_(
                coords=coords, 
                coord_system=self.coord_system)    

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

    # return an xarray dataset CF-Compliant
    def to_xarray(self) -> xr.Dataset:
        # we assume that we do not have any CF attributes in the xarray data structure
        # underline the Field. We need to convert Field to an xarray Dataset with CF attributes

        # grid_mapping = self.domain.coord_system.spatial.crs.to_cf()
        # # add grid_mapping variable
        # grid_mapping_var = xr.DataArray(name=grid_mapping['grid_mapping_name'],
        #                                 attrs = grid_mapping)

        # coords = {}
        # for ax, coord in self.coords.items():
        #     coords[ax] = xr.DataArray(name=str(ax), 
        #                               data=coord.data, 
        #                               dims=self._dset.coords[ax].dims, 
        #                               attrs=ax.encoding)

        # coords[grid_mapping_var.name] = grid_mapping_var

        # field_var = xr.DataArray(data=self.data, 
        #                          coords=coords,
        #                          dims=self.coord_system.dim_axes)
        # field_var.attrs={}
        # field_var.attrs['grid_mapping'] = grid_mapping['grid_mapping_name'] 

        # ds = xr.Dataset(
        #     data_vars = {
        #         self.name: field_var,
        #     },
        #     coords = coords
        # )
        # return ds
                
        ds = self._dset  # we need to copy the ds metadata (and not the data) maybe .copy()
        return ds
    
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
# TODO: move outside of the class and associate featuretype with class
    @classmethod
    def from_pyarrow_table(cls, table): 
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


class PointsField(Field, PointsFeature):
    __slots__ = ()
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
        n_pts = domain.number_of_points
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
                    np.full(shape=n_pts, fill_value=np.nan, dtype=np.float32)
                )
            case _:
                data_ = pint.Quantity(np.asarray(data))
        if data_.shape != (n_pts,):
            raise ValueError(
                "'data' must have one-dimensional values and the same size as "
                "the coordinates"
            )

        data_vars = {}
        attrs = properties if not None else {}
        data_vars[name] = xr.DataArray(
            data=data_, dims=self._DIMS_, attrs=attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(
                    data=anc_data, dims=self._DIMS_
                )

        ds_attrs = {_FIELD_NAME_ATTR_: name}

        super().__init__(
            data_vars=data_vars,
            coords=domain.coords,
            attrs=ds_attrs,
            coord_system=domain.coord_system
        )


class ProfilesField(Field, ProfilesFeature):
    __slots__ = ()
    _DOMAIN_CLS_ = Profiles

    __slots__ = ()

    def __init__(
        self,
        name: str,
        domain: Profiles,
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
        
        data_vars = {}
        attrs = properties if not None else {}
        data_vars[name] = xr.DataArray(data=data_, dims=self._DIMS_, attrs=attrs)
        data_vars[name].encoding = encoding if encoding is not None else {}
        
        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=self._DIMS)
        
        ds_attrs = {_FIELD_NAME_ATTR_: name}

        super().__init__(
            data_vars=data_vars,
            coords=domain.coords,
            attrs=ds_attrs,
            coord_system=domain.coord_system
        )


class GridField(Field, GridFeature):
    # NOTE: The default order of axes is assumed.

    _DOMAIN_CLS_ = Grid

    def __init__(
        self,
        name: str,
        domain: Grid,
        data: npt.ArrayLike | pint.Quantity | None = None,
#        dim_axes: Sequence[axis.Axis] | None = None, # This should not be used ... we fix the field to have all axis of the Domain!
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:

#        aux_axes = domain.coord_system.aux_axes
#        self._DIMS_ = domain.coord_system.dim_axes if dim_axes is None else tuple(dim_axes)
#        
        match data:
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
        
        #NOTE: THIS CODE CAN BE PUT IN ONE METHOD COMMON FOR ALL FIELDS! (MAYBE!)
        coords = {}
        for ax, coord in domain.coords.items():
            coords[ax] = xr.DataArray(coord, 
                                      dims=domain._dset[ax].dims, 
                                      attrs=domain._dset[ax].attrs)
        
        grid_mapping_attrs = domain.coord_system.spatial.crs.to_cf()
        grid_mapping_name = grid_mapping_attrs['grid_mapping_name']
        coords[grid_mapping_name] = xr.DataArray(data=np.byte(1),
                                                 name=grid_mapping_name,
                                                 attrs = grid_mapping_attrs)

        data_vars = {}

        field_attrs = properties if not None else {} # TODO: attrs can contain both properties and CF attrs
        data_vars[name] = xr.DataArray(data=data_, dims=domain._dset.dims, attrs=field_attrs)
        data_vars[name].attrs['grid_mapping'] = grid_mapping_attrs['grid_mapping_name'] 
        data_vars[name].encoding = encoding if encoding is not None else {}
        
        if ancillary is not None:
            ancillary_names = []
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=domain._dset.dims)
                data_vars[name].attrs['grid_mapping'] = grid_mapping_attrs['grid_mapping_name']
                ancillary_names.append(anc_name)

            data_vars[name].attrs['ancillary_variables'] = " ".join(ancillary_names)
        
        ds_attrs = {_FIELD_NAME_ATTR_: name}

# UNTIL HERE

        super().__init__(
            data_vars = data_vars,
            coords = coords, 
            attrs = ds_attrs,
            coord_system=domain.coord_system
        )

    # Spatial operations ------------------------------------------------------

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> PointsField:  # Self:
        # NOTE: This code works with all tested grids and returns the nearest
        # points.
        # Preparing data, labels, units, and dimensions.
        coord_system = self.domain.coord_system

        lat = self.coords[axis.latitude]
        lon = self.coords[axis.latitude]

        lat_vals = lat.magnitude
        lon_vals = lon.magnitude

        lat_labels = get_magnitude(latitude, lat.units)
        lon_labels = get_magnitude(longitude, lon.units)

        if isinstance(coord_system.spatial.crs, Geodetic):
            lat_vals, lon_vals = np.meshgrid(lat_vals, lon_vals, indexing='ij')
            dims = ('_latitude', '_longitude')
        else:
            all_dims = {lat.dims, lon.dims} # TODO: Review!!
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

        name = self.name
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
        method: str = 'bilinear'
    ) -> Self:
        import xesmf as xe
        if not isinstance(target, Domain):
            if isinstance(target, GridField):
                target = target.domain
            else:
                raise TypeError(
                    "'target' must be an instance of Domain or GridField"
                )
        #
        # TODO: check if they have the same CRS
        # if source CRS and target CRS are different
        # first transform source CRS to target CRS
        # 
        # get spatial lat/lon coordinates
        # we should get all horizontal coordinates -> e.g. Projection, RotatedPole ...
        # 
        lat = target.coords[axis.latitude]
        lon = target.coords[axis.longitude]
        
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], lat),
                "lon": (["lon"], lon)
            }
        )

        # NOTE: before regridding we need to dequantify  
        ds = self._dset.pint.dequantify()
        #
        # if we have ancillary data how they should be regridded?
        # for the moment we assume the same method for the field
        # TODO: maybe user should specify method for ancillary too!
        #
        regridder = xe.Regridder(ds, ds_out, method, unmapped_to_nan=True)
        dset_reg = regridder(ds, keep_attrs=True)
        
        ancillary = {}
        for v in dset_reg.data_vars:
            if v != self.name:
                ancillary[v] = dset_reg[v].pint.quantify()

        new_cs = CoordinateSystem(horizontal=target.coord_system.spatial.crs,
                            elevation=self.coord_system.spatial.elevation,
                            time=self.coord_system.time,
                            user_axes=self.coord_system.user_axes)
    
        coords = {}
        for ax in new_cs.axes:
            if isinstance(ax, axis.Horizontal):
                coords[ax] = target.coords[ax]
            else:
                coords[ax] = self.coords[ax]

        return GridField(
            name=self.name,
            data=dset_reg[self.name].pint.quantify(),
            domain=Grid(coords=coords, coord_system=new_cs),
            ancillary=ancillary,
            properties=self.properties,
            encoding=self.encoding
        )
