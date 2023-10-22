from typing import Mapping, Sequence, Self
import numpy.typing as npt
import numpy as np
import pyarrow as pa
from datetime import date, datetime

from . import axis, indexes
from .crs import Geodetic
from .domain import Domain
from .field import Field
from .indexers import get_array_indexer, get_indexer
import geokube.core.feature as xr
import pint
from .coord_system import CoordinateSystem
import pandas as pd

from numbers import Number

from enum import Enum

__FIELD_NAME_ATTR__ = '_geokube.field_name'

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

def _build_xrdset_without_index(
        self,
        dims,
        coords: Mapping[axis.Axis, pint.Quantity],
        name: str | None = None,
        data: pint.Quantity | None = None, 
        ancillary: Mapping[str, pint.Quantity] | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> xr.Dataset:

    data_vars = {}
    if data is not None: # field data is present    
        attrs = properties if not None else {}
        encoding = encoding if not None else {}
        data_vars[name]: xr.DataArray(data=data, dims=dims, attrs=attrs, encoding=encoding)
        
    if ancillary is not None:
        for anc_name, anc_data in ancillary.items():
            data_vars[anc_name] = xr.DataArray(data=anc_data, dims=dims)

    ds_attrs = {}
    if name is not None:
        ds_attrs = {__FIELD_NAME_ATTR__: name}
    
    return xr.Dataset(data_vars=data_vars,
                        coords=coords,
                        attrs=ds_attrs)

class FeatureType(Enum):
    Points = 1
    Profile = 2
    Grid = 3

class Feature(xr.Dataset):
    __slots__ = ('_coord_system')

    def __init__(
        self,
        coords: xr.Coordinates,
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity] | None = None,
        attrs: Mapping | None = None,
    ) -> None:
        
        super().__init__()(
            data_vars=data_vars, 
            coords=coords, 
            attrs=attrs
        )

        self._coord_system = coord_system

    def _from_xrdset(self, dset:xr.Dataset):
        
        feature = Feature(
            data_vars = dset.data_vars,
            coords = dset.coords,
            attrs = dset.attrs,
            coord_system = self.coord_system
        )
        
        if isinstance(self, Feature):
            return feature
        else:
            type(self)(
                coords=feature.coords,
                coord_system=feature.coord_system,
                name=feature.name,
                data=feature.data, 
                ancillary=feature.ancillary,
                properties=feature.properties,
                encoding=feature.encoding
        )

    #TODO: does __getitem__ works with Axis? does it returns an xr.DataArray?

    @property # return field name
    def name(self) -> str:
        return self.attrs[__FIELD_NAME_ATTR__]
    
    @property # define data method to return field data
    def data(self):
        return self[self.name].data

    @property
    def properties(self):
        return self[self.name].attrs

    @property
    def encoding(self):
        return self[self.name].encoding

    @property  # override xr.Dataset coords method to return Mapping axis -> data
    def coords(
        self, coord_axis: axis.Axis | None = None
    ) -> dict[axis.Axis, pint.Quantity] | pint.Quantity:
        if coord_axis is None:
            coords = {}
            for ax in self.coord_system.axes:
                 coords[ax] = self[ax].data
            return coords
        return self.coords[coord_axis]

    @property # TODO: define setter method
    def domain(self) -> Domain:
        if (self.__DOMAIN_CLS__):
            coords = {}
            for ax in self.coord_system.axes:
                coords[ax] = self.coords[ax]
            return self.__DOMAIN_CLS__(
                        coords=coords, 
                        coord_system=self._coord_system)    

    @property
    def ancillary(self, name: str | None = None) -> dict | pint.Quantity:
        if name is not None:
            return self._ds[name].data
        ancillary = {}
        for c in self._ds.data_vars:
            if c != self.name:
                ancillary[c] = self._ds[c].data
        return self.ancillary

    @property
    def coord_system(self):
        return self._coord_system

    # spatial operations
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
        new_ds = self.sel(h_idx)
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            new_ds = new_ds.sel(v_idx)
        return self._from_xrdset(new_ds)
    
    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        dset = self.sel(idx, method='nearest', tolerance=np.inf)
        return self._from_xrdset(dset)
        
    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.vertical: elevation}
        dset = self._dset.sel(idx, method='nearest', tolerance=np.inf)
        return self._from_xrdset(dset)

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        dset = self.sel({axis.time: slice(start, end)})
        return self._from_xrdset(dset)
        
    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        dset = self.sel(idx, method='nearest', tolerance=None)
        return self._from_xrdset(dset)

    def latest(self) -> Self:
        if axis.time not in self.coordinates():
            raise NotImplementedError()
        latest = self[axis.time].max().astype(str).item().magnitude
        idx = {axis.time: slice(latest, latest)}
        dset = self.sel(idx)
        return self._from_xrdset(dset)

    # Pyarrow conversions -----------------------------------------------------

    def build_pyarrow_metadata(self):
        # geokube Field JSON Schema
        # -> name: ‘string’
        # -> kind: ‘string’ (Gridded, Points, Profile, Timeseries)
        # -> properties: dict
        # -> cf-encoding: dict
        # -> coord_system: list
        import json
        metadata = {'name': self.name,
                    'kind': str(self._DOMAIN_TYPE),
                    'properties': json.dumps(self.properties).encode('utf-8'),
                    'cf_encoding': json.dumps(self.encoding).encode('utf-8')
        }
        metadata['coord_system'] = {} 
        metadata['coord_system']['horizontal'] = str(self.domain.coord_system.spatial.crs)
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

class PointsFeature(Feature):

    __slots__ = ('_n_points', '_feature_type')

    def __init__(
        self,
        coords: Mapping[axis.Axis, pint.Quantity],
        coord_system: CoordinateSystem,
        name: str | None = None,
        data: pint.Quantity | None = None, 
        ancillary: Mapping[str, pint.Quantity] | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        
        dims = ('_points')

        ds = _build_xrdset_without_index(
            dims=dims,
            coords=coords,
            name=name,
            data=data,
            ancillary=ancillary,
            properties=properties,
            encoding=encoding
        )
        
        hor_axes = set(coord_system.spatial.crs.axes)
        for axis_ in coord_system.axes:
            if axis_ not in hor_axes:
                ds = ds.set_xindex(axis_, indexes.OneDimIndex)
   
        ds = ds.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )

        super().__init__(data_vars=ds.data_vars,
                         coords=ds.coords,
                         coord_system=coord_system)
        
        self._feature_type = FeatureType.Points
    
    @property
    def number_of_points(self) -> int:
        return self.__n_pts