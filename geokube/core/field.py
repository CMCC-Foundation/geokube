"""
Field
=====

A field construct that contains a data variable with related units,
domain, ancillary constructs, properties, etc.

Classes
-------

:class:`geokube.core.field.Field`
    Base class for field constructs

:class:`geokube.core.field.PointsField`
    Field defined on a point domain

:class:`geokube.core.field.ProfilesField`
    Field defined on a profile domain

:class:`geokube.core.field.GridField`
    Field defined on a gridded domain

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable, Self

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import pyarrow as pa
from pyproj import Transformer
import xarray as xr

from . import axis
from .cell_method import CellMethod
from .coord_system import CoordinateSystem
from .crs import Geodetic
from .domain import Domain, Grid, Points, Profiles
from .feature import (
    PointsFeature, ProfilesFeature, GridFeature, _as_points_dataset
)


# TODO: Check whether `pyarrow` stuff should be documented or removed.

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


# TODO: Consider making this class internal.
class Field:
    __slots__ = ()
    _DOMAIN_CLS_: type[Domain]

    _dset: xr.Dataset

    # TODO: Add cell methods
    # __slots__ = ('_cell_method_',)

    # NOTE: `Domain` and `Field` have exactly the same method.
    @classmethod
    def _from_xarray_dataset(
        cls,
        ds: xr.Dataset, # This should be CF-compliant or use cf_mapping to be a CF-compliant
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> Self:
        obj = object.__new__(cls)
        # TODO: Make sure that `cls.__mro__[2]` returns the correct `Feature`
        # class from the inheritance hierarchy.
        feature_cls = cls.__mro__[2]
        # pylint: disable=unnecessary-dunder-call
        feature_cls.__init__(obj, ds, cf_mappings)
        return obj

    # 
    # - this is the same method name in Feature class ->
    # in order to have precedence Field should be inherited first
    # 
    @classmethod
    def _from_xrdset(
        cls, dset: xr.Dataset, coord_system: CoordinateSystem
    ) -> Self:
        coords = {}        
        for ax in coord_system.axes:
            coords[ax] = dset.coords[ax]
        domain = cls._DOMAIN_CLS_(coords=coords, coord_system=coord_system)
        name = dset.attrs[_FIELD_NAME_ATTR_]
        data = dset[name].data
        properties = dset[name].attrs
        encoding = dset[name].encoding
        if 'ancillary_variables' in dset[name].attrs:
            ancillary = {}
            for c in dset[name].attrs['ancillary_variables'].split():
                ancillary[c] = dset[c].data
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

    @classmethod
    def as_xarray_dataset(
        cls,
        data_vars: Mapping[str, npt.ArrayLike | pint.Quantity | xr.DataArray],
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity | xr.DataArray],
        coord_system: CoordinateSystem
    ) -> xr.Dataset:
        da = coord_system.crs.as_xarray()
        r_coords = coords
        r_coords[da.name] = da
        ds = xr.Dataset(
            coords=r_coords
        )
        ds.attrs['grid_mapping'] = da.name
        return ds

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
        ancillary_ = {}
        for c in self._dset.data_vars:
            if c != self.name:
                ancillary_[c] = self._dset[c].data
        return ancillary_

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

    @property
    def cell_method(self) -> CellMethod | None:
        darr = self._dset[self.name]
        if (cmethod := darr.attrs.get('cell_methods')) is not None:
            return CellMethod.parse(cmethod)
        return None

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
        encoding: Mapping | None = None,
        cell_method: str = ''
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
        attrs = properties if properties is not None else {}
        attrs['units'] = data_.units
        if cell_method:
            attrs['cell_methods'] = cell_method
        data_vars[name] = xr.DataArray(
            data=data_.magnitude, dims=self._DIMS_, attrs=attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(
                    data=anc_data, dims=self._DIMS_
                )

        # ds_attrs = {_FIELD_NAME_ATTR_: name}

        dset = domain._dset
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(data_vars)
        dset.attrs[_FIELD_NAME_ATTR_] = name

        super().__init__(dset)


class ProfilesField(Field, ProfilesFeature):
    __slots__ = ()
    _DOMAIN_CLS_ = Profiles

    def __init__(
        self,
        name: str,
        domain: Profiles,
        data: npt.ArrayLike | pint.Quantity | None = None,
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None,
        cell_method: str = ''
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
        attrs = properties if properties is not None else {}
        attrs['units'] = data_.units
        if cell_method:
            attrs['cell_methods'] = cell_method
        data_vars[name] = xr.DataArray(
            data=data_.magnitude, dims=self._DIMS_, attrs=attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=self._DIMS)

        # ds_attrs = {_FIELD_NAME_ATTR_: name}

        dset = domain._dset
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(data_vars)
        dset.attrs[_FIELD_NAME_ATTR_] = name

        super().__init__(dset)

    def as_points(self) -> PointsField:
        return PointsField._from_xarray_dataset(_as_points_dataset(self))


class GridField(Field, GridFeature):
    __slots__ = ()
    _DOMAIN_CLS_ = Grid
    # NOTE: The default order of axes is assumed.

    def __init__(
        self,
        name: str,
        domain: Grid,
        data: npt.ArrayLike | pint.Quantity | None = None,
#        dim_axes: Sequence[axis.Axis] | None = None, # This should not be used ... we fix the field to have all axis of the Domain!
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None,
        cell_method: str = ''
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

        # coords = {}
        # for ax, coord in domain.coords.items():
        #     coords[ax] = xr.DataArray(coord, 
        #                               dims=domain._dset[ax].dims, 
        #                               attrs=domain._dset[ax].attrs)

        grid_mapping_attrs = domain.coord_system.spatial.crs.to_cf()
        grid_mapping_name = grid_mapping_attrs['grid_mapping_name']
        # coords[grid_mapping_name] = xr.DataArray(data=np.byte(1),
        #                                          name=grid_mapping_name,
        #                                          attrs = grid_mapping_attrs)

        data_vars = {}

        field_attrs = properties if properties is not None else {} # TODO: attrs can contain both properties and CF attrs
        field_attrs |= {
            'units': data_.units, 'grid_mapping': grid_mapping_name
        }
        if cell_method:
            field_attrs['cell_methods'] = cell_method
        data_vars[name] = xr.DataArray(
            data=None if data_ is None else data_.magnitude,
            dims=domain._dset.dims,
            attrs=field_attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            ancillary_names = []
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=domain._dset.dims)
                data_vars[name].attrs['grid_mapping'] = grid_mapping_name
                ancillary_names.append(anc_name)

            data_vars[name].attrs['ancillary_variables'] = " ".join(ancillary_names)

        # ds_attrs = {_FIELD_NAME_ATTR_: name}

# UNTIL HERE

        # ds = xr.Dataset(
        #     data_vars=data_vars,
        #     coords=coords,
        #     attrs=ds_attrs
        # )

        # super().__init__(
        #     ds=ds
        # )

        dset = domain._dset
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(data_vars)
        dset.attrs[_FIELD_NAME_ATTR_] = name

        super().__init__(dset)

    # Spatial operations ------------------------------------------------------

    # def nearest_horizontal(
    #     self,
    #     latitude: npt.ArrayLike | pint.Quantity,
    #     longitude: npt.ArrayLike | pint.Quantity
    # ) -> PointsField:  # Self:
    #     feature = super().nearest_horizontal(latitude, longitude)

    #     return PointsField(
    #         name=self.name,
    #         domain=Points(
    #             coords=feature.coords, coord_system=feature.coord_system
    #         ),
    #         data=feature._dset[self.name].data
    #     )

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
        
        # NOTE: Maybe it is better to use `target._dset.coords` instead of
        # `target.coords` because the former keeps the attributes.
        # lat = target.coords[axis.latitude]
        # lon = target.coords[axis.longitude]
        target_coords = dict(target._dset.coords)
        lat = target_coords[axis.latitude].to_numpy()
        lon = target_coords[axis.longitude].to_numpy()

        ds_out = xr.Dataset({"lat": (["lat"], lat), "lon": (["lon"], lon)})

        # NOTE: before regridding we need to dequantify  
        # ds = self._dset.pint.dequantify()
        #
        # if we have ancillary data how they should be regridded?
        # for the moment we assume the same method for the field
        # TODO: maybe user should specify method for ancillary too!
        #
        # NOTE: The new underlying dataset does not need dequantification
        # because coordinates and data variables are already dequantified,
        # while the units are in the attributes.
        regridder = xe.Regridder(
            self._dset, ds_out, method, unmapped_to_nan=True
        )
        dset_reg = regridder(self._dset, keep_attrs=True)

        ancillary = {
            name: darr
            for name, darr in dset_reg.data_vars.items()
            if name != self.name
        }

        new_cs = CoordinateSystem(
            horizontal=target.coord_system.spatial.crs,
            elevation=self.coord_system.spatial.elevation,
            time=self.coord_system.time,
            user_axes=self.coord_system.user_axes
        )

        coords = {}
        for ax in new_cs.axes:
            if isinstance(ax, axis.Horizontal):
                coords[ax] = target_coords[ax]
            else:
                coords[ax] = self._dset.coords[ax]

        return GridField(
            name=self.name,
            data=dset_reg[self.name],  # .pint.quantify(),
            domain=Grid(coords=coords, coord_system=new_cs),
            ancillary=ancillary,
            properties=self.properties,
            encoding=self.encoding
        )

    def interpolate_(
        self, target: Domain | Field, method: str = 'nearest', **kwargs
    ) -> Field:
        dset = self._dset.pint.dequantify()
        coords = dict(dset.coords)
        coord_system = self._coord_system
        spatial = coord_system.spatial
        crs = spatial.crs
        attrs = dset.attrs
        name = attrs[_FIELD_NAME_ATTR_]
        del coords[dset[name].attrs['grid_mapping']]

        match target:
            case Domain():
                target_domain = target
            case Field():
                target_domain = target.domain
            case _:
                raise TypeError("'target' must be a domain or field")
        target_dims = (axis.latitude, axis.longitude)
        target_coords = target_domain.coords
        target_coords = {axis: target_coords[axis] for axis in target_dims}

        if isinstance(self.crs, Geodetic):
            target_coords = coords | target_coords
            kwargs.setdefault('fill_value', 'extrapolate')
            result_dset = dset.interp(
                coords=target_coords, method=method, kwargs=kwargs
            )
        else:
            target_lat = target_coords[axis.latitude]
            target_lon = target_coords[axis.longitude]
            if target_lat.ndim == target_lon.ndim == 1:
                target_lon, target_lat = np.meshgrid(target_lon, target_lat)
            transformer = Transformer.from_crs(
                crs_from=target_domain.coord_system.spatial.crs._crs,
                crs_to=crs._crs,
                always_xy=True
            )
            target_x, target_y = transformer.transform(target_lon, target_lat)
            # NOTE: This is required to get the same axes as given in `coords`.
            axis_x, axis_y = coords.keys() & crs.dim_axes
            target_coords = xr.Dataset(
                data_vars={
                    axis_x: xr.DataArray(data=target_x, dims=target_dims),
                    axis_y: xr.DataArray(data=target_y, dims=target_dims)
                },
                coords=target_coords
            )
            dset = dset.drop(labels=(axis.latitude, axis.longitude))
            kwargs.setdefault('fill_value', None)
            result_dset = dset.interp(
                coords=target_coords, method=method, kwargs=kwargs
            )
            result_dset = result_dset.drop(labels=(axis_x, axis_y))

        result_dset = result_dset.pint.quantify()
        result_dset.attrs[_FIELD_NAME_ATTR_] = attrs[_FIELD_NAME_ATTR_]
        result_coord_system = CoordinateSystem(
            horizontal=target_domain.coord_system.spatial.crs,
            elevation=spatial.elevation,
            time=coord_system.time,
            user_axes=coord_system.user_axes
        )
        result_field = self._from_xrdset(result_dset, result_coord_system)
        return result_field

    def interpolate(
        self, target: Domain | Field, method: str = 'nearest', **kwargs
    ) -> Field:
        # spatial interpolation
        # dset = self._dset.pint.dequantify() - it is needed since units are not kept!
        # dset = self._dset.pint.dequantify()
        dset = self._dset

        match target:
            case Domain():
                pass
            case Field():
                target = target.domain
            case _:
                raise TypeError("'target' must be a domain or field")

        if self.crs._crs != target.crs._crs:
            # we need to transform the target domain to the same crs of the domain to 
            # be interpolated and we perform the interpolation on the horizontal axes
            #
            target_x, target_y = target.spatial_transform_to(self.crs)
            kwargs.setdefault('fill_value', None)
            target_dims = target.crs.dim_axes
            dims = (axis.latitude, axis.longitude)
            target_ = xr.Dataset(
                data_vars={
                    self.crs.dim_X_axis: xr.DataArray(data=target_x, dims=dims),
                    self.crs.dim_Y_axis: xr.DataArray(data=target_y, dims=dims)
                },
                coords={axis: target.coords[axis] for axis in dims}
            )
            dset = dset.drop(labels=(axis.latitude, axis.longitude))
            ds = dset.interp(coords=target_, method=method, kwargs=kwargs)
            # TODO: workaround - remove when adding attributes to field for ancillary data.
            ds = ds.drop(labels=(self.crs.dim_X_axis, self.crs.dim_Y_axis))
        else:
            coords = dict(dset.coords)
            del coords[dset.attrs['grid_mapping']]
            coords = {axis: coord.to_numpy() for axis, coord in coords.items()}
            target_coords = {
                axis: coord.to_numpy()
                for axis, coord in target._dset.coords.items()
                if axis in target.crs.axes
            }
            target_coords = coords | target_coords
            kwargs.setdefault('fill_value', 'extrapolate')
            ds = dset.interp(
                coords=target_coords, method=method, kwargs=kwargs
            )

        # ds contains the data interpolated on the new domain
        ancillary = {}
        for v in ds.data_vars:
            if v != self.name:
                ancillary[v] = ds[v].pint.quantify()

        cs = CoordinateSystem(
            horizontal=target.crs,
            elevation=self.coord_system.spatial.elevation,
            time=self.coord_system.time,
            user_axes=self.coord_system.user_axes
        )

        coords = {}
        for ax in cs.axes:
            if isinstance(ax, axis.Horizontal):
                coords[ax] = target.coords[ax]
            else:
                coords[ax] = self.coords[ax]

        return GridField(
            name=self.name,
            data=ds[self.name].pint.quantify(),
            domain=Grid(coords=coords, coord_system=cs),
            ancillary=ancillary,
            properties=self.properties,
            encoding=self.encoding
        )

    def is_geodetic(self):
        return isinstance(self.crs, Geodetic) 

    def as_geodetic(self):
        if self.is_geodetic():
            return self
        
        return self.interpolate(
            target=self.domain.as_geodetic(), # this has a semantic -> domain geodetic grid can be different from the original
            method="nearest"
        )

    def as_points(self) -> PointsField:
        return PointsField._from_xarray_dataset(_as_points_dataset(self))

    def resample(
        self, freq: str, operator: Callable | str = 'nanmean', **kwargs
    ) -> Self:
        dset = self._dset
        data = dset[self.name].data
        time_idx = dset.xindexes[axis.time].index.index
        n_time = time_idx.size

        match time_idx:
            case pd.DatetimeIndex():
                time = pd.Series(data=np.arange(n_time), index=time_idx)
            case pd.IntervalIndex():
                # TODO: Check if this code has issue with ERA5 (rotated pole),
                #  when `freq`is e.g. `'5H'`.
                if kwargs:
                    raise ValueError(
                        "'kwargs' are not allowed for interval indices"
                    )
                left_bnd, right_bnd = time_idx.left, time_idx.right
                src_freqs = pd.unique(right_bnd - left_bnd)
                if src_freqs.size == 1:
                    src_freq = abs(src_freqs[0])
                else:
                    raise ValueError(
                        "'time_idx' must have equal differences for resampling"
                    )
                src_diff = n_time * src_freq
                dst_freq = pd.to_timedelta(freq).to_timedelta64()
                ratio = float(dst_freq / src_freq)
                if ratio.is_integer() and ratio >= 1 and dst_freq <= src_diff:
                    time = pd.Series(data=np.arange(n_time), index=left_bnd)
                else:
                    raise ValueError(
                        "'freq' does not correspond to the interval durations"
                    )
            case _:
                raise NotImplementedError(
                    f"'time_idx' has the type {type(time_idx).__name__}, "
                    "which is not supported; it must be an instance of "
                    "'DatetimeIndex' or 'IntervalIndex'"
                )

        left_time_res = time.resample(
            rule=freq, label='left', origin='start', **kwargs
        )
        left_gr = left_time_res.grouper
        # TODO: Try to avoid the second call to `time.resample` and find the
        # right bound another way, e.g. with `time_res.freq.delta.to_numpy()`.
        right_time_res = time.resample(
            rule=freq, label='right', origin='start', **kwargs
        )
        right_gr = right_time_res.grouper
        new_time = pd.IntervalIndex.from_arrays(
            left=left_gr.binlabels, right=right_gr.binlabels, closed='both'
        )
        time_axis_idx = self.dim_axes.index(axis.time)
        new_shape = list(data.shape)
        new_shape[time_axis_idx] = new_time.size
        # NOTE: For NumPy arrays, it is possible to use `numpy.split`, but it
        # seems that Dask does not have such a function.
        bins = left_gr.bins
        slices = (
            [slice(bins[0])]
            + [slice(bins[i], bins[i + 1]) for i in range(bins.size - 1)]
        )
        arr_lib = da if isinstance(data, da.Array) else np
        func = operator if callable(operator) else getattr(arr_lib, operator)
        # FIXME: If `data.dtype` is integral, we want a floating-point result
        # for some operators like `mean` or `median` and integral for others
        # like `min` or `max`.
        new_data = arr_lib.empty(shape=new_shape, dtype=data.dtype)
        whole_axis = (slice(None),)
        for i, s in enumerate(slices):
            idx_before = whole_axis * time_axis_idx
            idx_after = whole_axis * (len(new_shape) - time_axis_idx - 1)
            i_ = idx_before + (i,) + idx_after
            s_ = idx_before + (s,) + idx_after
            # TODO: Test optimization with something like:
            # `func(data[s_], axis=time_axis_idx, out=new_data[i_])`.
            new_data[i_] = func(data[s_], axis=time_axis_idx)

        domain = self.domain
        new_coords = domain.coords.copy()
        new_coords[axis.time] = new_time

        return type(self)(
            name=self.name,
            domain=type(domain)(new_coords, domain.coord_system),
            data=pint.Quantity(new_data, dset[self.name].attrs.get('units')),
            ancillary=self.ancillary,
            properties=self.properties,
            encoding=self.encoding
        )

    def resample_(
        self, freq: str, operator: str = 'mean', **kwargs
    ) -> Self:
        dset = self._dset
        time = dset[axis.time]
        idx = {axis.time: freq}
        diff = pd.to_timedelta(freq).to_timedelta64()

        match time_idx := dset.xindexes[axis.time].index.index:
            case pd.DatetimeIndex():
                res = dset.resample(indexer=idx, label='left', **kwargs)
                result_dset = getattr(res, operator)()
                left = result_dset[axis.time].to_numpy()
                right = left + diff
                # grouper = res.groupers[0]
                # closed = grouper.grouper.closed or 'both'
                # if (label := grouper.index_grouper.label) == 'left':
                #     left = result_dset.xindexes[axis.time].index.to_numpy()
                #     right = left + diff
                # else:
                #     right = result_dset.xindexes[axis.time].index.to_numpy()
                #     left = right - diff
                result_dset[axis.time] = xr.Variable(
                    dims=time.dims,
                    data=pd.IntervalIndex.from_arrays(
                        left, right, closed=kwargs.get('closed', 'both')
                    ),
                    attrs=time.attrs,
                    encoding=time.encoding
                )
            case pd.IntervalIndex():
                # TODO: Check if this code has issue with ERA5 (rotated pole),
                #  when `freq`is e.g. `'5H'`.
                if kwargs:
                    raise ValueError(
                        "'kwargs' are not allowed for interval indices"
                    )
                left_bnd, right_bnd = time_idx.left, time_idx.right
                src_freqs = pd.unique(right_bnd - left_bnd)
                if src_freqs.size == 1:
                    src_freq = abs(src_freqs[0])
                else:
                    raise ValueError(
                        "'time_idx' must have equal differences for resampling"
                    )
                src_diff = time.size * src_freq
                dst_freq = pd.to_timedelta(freq).to_timedelta64()
                ratio = float(dst_freq / src_freq)
                if ratio.is_integer() and ratio >= 1 and dst_freq <= src_diff:
                    interm_data_vars = {
                        name: var.variable
                        for name, var in dset.data_vars.items()
                    }
                    interm_coords = dict(dset.coords)
                    # interm_coords = {
                    #     axis_: coord
                    #     for axis_, coord in dset.coords.items()
                    #     if isinstance(axis_, axis.Axis)
                    # }
                    time_axis = (interm_coords.keys() & {axis.time}).pop()
                    interm_coords[time_axis] = xr.Variable(
                        dims=time.dims,
                        data=time_idx.left,
                        attrs=time.attrs,
                        encoding=time.encoding
                    )
                    interm_dset = xr.Dataset(
                        data_vars=interm_data_vars,
                        coords=interm_coords,
                        attrs=dset.attrs.copy()
                    )
                    res = interm_dset.resample(
                        indexer=idx,
                        closed='left',
                        label='left',
                        origin='start'
                    )
                    result_dset = getattr(res, operator)()
                    left = result_dset[axis.time].to_numpy()
                    right = left + diff
                    result_dset[axis.time] = xr.Variable(
                        dims=time.dims,
                        data=pd.IntervalIndex.from_arrays(
                            left, right, closed='both'
                        ),
                        attrs=time.attrs,
                        encoding=time.encoding
                    )
                else:
                    raise ValueError(
                        "'freq' does not correspond to the interval durations"
                    )
            case _:
                raise NotImplementedError(
                    f"'time_idx' has the type {type(time_idx).__name__}, "
                    "which is not supported; it must be an instance of "
                    "'DatetimeIndex' or 'IntervalIndex'"
                )

        domain = self.domain
        name = self.name
        new_coords = {
            axis_: coord.pint.quantify().data
            for axis_, coord in result_dset.coords.items()
            if isinstance(axis_, axis.Axis)
        }

        return type(self)(
            name=name,
            domain=type(domain)(new_coords, domain.coord_system),
            data=result_dset[name].data,
            # ancillary=self.ancillary,
            properties=self.properties,
            encoding=self.encoding
        )
