import os
import json
import logging
import re
import tempfile
import uuid
import warnings
from collections import defaultdict
from enum import Enum
from html import escape
from itertools import chain
from numbers import Number
from string import Template
from types import MethodType
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr
import math

from ..utils.decorators import geokube_logging
from ..utils.hcube_logger import HCubeLogger
from ..utils import util_methods
from .errs import EmptyDataError
from .axis import Axis, AxisType
from .coord_system import GeogCS, RegularLatLon
from .domain import Domain, DomainType
from .enums import RegridMethod
from .field import Field
from .domainmixin import DomainMixin

IndexerType = Union[slice, List[slice], Number, List[Number]]


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


# TODO: Not a priority
# dc = datacube.set_id_pattern('{standard_name}')
# dc['air_temperature']
# dc['latitude']
#
class DataCube(DomainMixin):
    __slots__ = (
        "_fields",
        "_domain",
        "_properties",
        "_encoding",
        "_ncvar_to_name",
    )

    _LOG = HCubeLogger(name="DataCube")

    def __init__(
        self,
        fields: List[Field],
        properties: Mapping[Any, Any],
        encoding: Mapping[Any, Any],
    ) -> None:
        if len(fields) == 0:
            warnings.warn("No fields provided for the DataCube!")
            self._fields = {}
            self._domain = None
            self._ncvar_to_name = None
        else:
            self._ncvar_to_name = {f.ncvar: f.name for f in fields}
            self._fields = {f.name: f for f in fields}
            self._domain = Domain.merge([f.domain for f in fields])
        self._properties = properties if properties is not None else {}
        self._encoding = encoding if encoding is not None else {}

    @property
    def properties(self) -> dict:
        return self._properties

    @property
    def encoding(self) -> dict:
        return self._encoding

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def fields(self) -> dict:
        return self._fields

    @property
    def nbytes(self) -> int:
        return sum(field.nbytes for field in self.fields.values())

    def __len__(self):
        return len(self._fields)

    def __contains__(self, key: str) -> bool:
        return (
            (key in self._fields)
            or (key in self._ncvar_to_name)
            or (key in self._domain)
        )

    @geokube_logging
    def __getitem__(
        self,
        key: Union[
            Iterable[str], Iterable[Tuple[str, str]], str, Tuple[str, str]
        ],
    ):
        if isinstance(key, str) and (
            (key in self._fields) or key in self._ncvar_to_name
        ):
            return self._fields.get(
                key, self._fields.get(self._ncvar_to_name.get(key))
            )
        elif isinstance(key, Iterable) and not isinstance(key, str):
            return DataCube(
                fields=[self[k] for k in key],
                properties=self.properties,
                encoding=self.encoding,
            )
        else:
            item = self.domain[key]
            if item is None:
                raise KeyError(
                    f"Key `{key}` of type `{type(key)}` is not found in the"
                    " DataCube"
                )
            return item

    def __next__(self):
        for f in self._fields.values():
            yield f
        raise StopIteration

    def __repr__(self) -> str:
        return self.to_xarray(encoding=False).__repr__()

    #        return formatting.array_repr(self.to_xarray())

    def _repr_html_(self):
        return self.to_xarray(encoding=False)._repr_html_()
        # if OPTIONS["display_style"] == "text":
        #     return f"<pre>{escape(repr(self.to_xarray()))}</pre>"
        # return formatting_html.array_repr(self)

    @geokube_logging
    def geobbox(
        self,
        north=None,
        south=None,
        west=None,
        east=None,
        top=None,
        bottom=None,
    ):
        return DataCube(
            fields=[
                self._fields[k].geobbox(
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                    top=top,
                    bottom=bottom,
                )
                for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def locations(
        self,
        latitude,
        longitude,
        vertical: Optional[List[Number]] = None,
    ):
        return DataCube(
            fields=[
                self._fields[k].locations(
                    latitude=latitude, longitude=longitude, vertical=vertical
                )
                for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def sel(
        self,
        indexers: Mapping[Union[Axis, str], Any] = None,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        roll_if_needed: bool = True,
        **indexers_kwargs: Any,
    ) -> "DataCube":  # this can be only independent variables
        fields = []
        for k in self._fields.keys():
            try:
                fields.append(
                    self._fields[k].sel(
                        indexers=indexers,
                        roll_if_needed=roll_if_needed,
                        method=method,
                        tolerance=tolerance,
                        drop=drop,
                        **indexers_kwargs,
                    )
                )
            except EmptyDataError as err:
                DataCube._LOG.info(f"skipping field `{k}` due to `{err}`")
                continue
        return DataCube(
            fields=fields,
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def interpolate(
        self, domain: Domain, method: str = "nearest"
    ) -> "DataCube":
        return DataCube(
            fields=[
                self._fields[k].interpolate(domain=domain, method=method)
                for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def resample(
        self,
        operator: Union[Callable, MethodType, str],
        frequency: str,
        **resample_kwargs,
    ) -> "DataCube":
        return DataCube(
            fields=[
                self._fields[k].resample(
                    operator=operator, frequency=frequency, **resample_kwargs
                )
                for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def average(self, dim: str | None = None) -> "DataCube":
        return DataCube(
            fields=[
                self._fields[k].average(dim=dim) for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def to_regular(
        self,
    ) -> "DataCube":
        return DataCube(
            fields=[self._fields[k].to_regular() for k in self._fields.keys()],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def extract_polygons(self, geometry, crop=True, return_mask=False):
        return DataCube(
            fields=[
                self._fields[k].extract_polygons(
                    geometry=geometry, crop=crop, return_mask=return_mask
                )
                for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    @geokube_logging
    def regrid(
        self,
        target: Union[Domain, "Field"],
        method: Union[str, RegridMethod] = "bilinear",
        weights_path: Optional[str] = None,
        reuse_weights: bool = True,
    ) -> "DataCube":
        return DataCube(
            fields=[
                self._fields[k].regrid(
                    target=target,
                    method=method,
                    weights_path=weights_path,
                    reuse_weights=reuse_weights,
                )
                for k in self._fields.keys()
            ],
            properties=self.properties,
            encoding=self.encoding,
        )

    def to_geojson(self, target=None):
        if self.domain.type is DomainType.POINTS:
            if self.latitude.size != 1 or self.longitude.size != 1:
                raise NotImplementedError(
                    "'self.domain' must have exactly 1 point"
                )
            units = {
                field.name: str(field.units) for field in self.fields.values()
            }
            coords = [self.longitude.item(), self.latitude.item()]
            result = {"type": "FeatureCollection", "features": []}
            for time in self.time.values.flat:
                time_ = pd.to_datetime(time).strftime("%Y-%m-%dT%H:%M")
                feature = {
                    "type": "Feature",
                    "units": units,
                    "geometry": {"type": "Point", "coordinates": coords},
                    "properties": {"time": time_},
                }
                for field in self.fields.values():
                    field.load()
                    try:
                        value = (
                            field.sel(time=time_)
                            if field.time.size > 1
                            else field
                        )
                    except EmptyDataError:
                        continue
                    else:
                        feature["properties"][field.name] = float(value)
                result["features"].append(feature)
        elif (
            self.domain.type is DomainType.GRIDDED or self.domain.type is None
        ):
            # HACK: The case `self.domain.type is None` is included to be able
            # to handle undefined domain types temporarily.
            if self.time.size == 1:
                result = {}
            else:
                raise NotImplementedError(
                    f"multiple times are not supported for geojson"
                )
            cube = (
                self
                if isinstance(self.domain.crs, GeogCS)
                else self.to_regular()
            )
            for field in self.fields.values():
                field.load()
            axis_names = cube.domain._axis_to_name
            units = {
                field.name: str(field.units) for field in self.fields.values()
            }
            lon_min = self.longitude.min().item()
            lat_min = self.latitude.min().item()
            lon_max = self.longitude.max().item()
            lat_max = self.latitude.max().item()      
            grid_x, grid_y = cube.domain._infer_resolution()
            grid_x = grid_x/2.0
            grid_y = grid_y/2.0
            for time in self.time.values.flat:
                time_ = pd.to_datetime(time).strftime("%Y-%m-%dT%H:%M")
                time_data = {
                    "type": "FeatureCollection",
                    "date": time_,
                    "bbox": [
                        lon_min,  # West
                        lat_min,  # South
                        lon_max,  # East
                        lat_max,  # North
                    ],
                    "units": units,
                    "features": [],
                }
                for lat in cube.latitude.values.flat:
                    for lon in cube.longitude.values.flat:
                        idx = {
                            axis_names[AxisType.LATITUDE]: lat,
                            axis_names[AxisType.LONGITUDE]: lon,
                        }
                        # if self.time.shape:
                        if self.time.size > 1:
                            idx[axis_names[AxisType.TIME]] = time_
                        # TODO: Check whether this works now:
                        # this gives an error if only 1 time is selected before to_geojson()
                        # Polygon:
                        # for each lat/lon we have to define a polygon with lan/lon centered
                        # the cell length depends on the grid resolution (that should be computed)
                        lonv = lon.item()
                        latv = lat.item()
                        lon_lower = np.clip(lonv - grid_x, a_min=lon_min, a_max=lon_max)
                        lat_upper = np.clip(latv + grid_y, a_min=lat_min, a_max=lat_max)
                        lon_upper = np.clip(lonv + grid_x, a_min=lon_min, a_max=lon_max)
                        lat_lower = np.clip(latv - grid_y, a_min=lat_min, a_max=lat_max)
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates":[  
                                   [ [lon_lower, lat_upper], [lon_upper, lat_upper],
                                     [lon_upper, lat_lower], [lon_lower, lat_lower],
                                     [lon_lower, lat_upper]
                                   ]
                                ]
                            },
                            "properties": {},
                        }
                        for field in cube.fields.values():
                            try:
                                value = field.sel(indexers=idx)
                                value_ = float(value)
                                if math.isnan(value):
                                    value_ = None                                
                            except EmptyDataError:
                                continue
                            except ValueError:
                                try:
                                    value_ = value.item()
                                except AttributeError:
                                    value_ = value                                
                            else:
                                feature["properties"][field.name] = value_                     
                        time_data["features"].append(feature)
                result = time_data
        else:
            raise NotImplementedError(
                f"'self.domain.type' is {self.domain.type}, which is currently"
                " not supported"
            )

        if target is not None:
            with open(target, mode="w") as file:
                json.dump(result, file, indent=4)

        return result

    def to_image(
        self, 
        filepath, 
        width, 
        height, 
        dpi=100, 
        format='png', 
        transparent=True, 
        bgcolor='FFFFFF'
    ):
        # TODO: support multiple fields
        if len(self.fields) > 1:
            raise ValueError("to_image support only 1 field")
        else:
            next(iter(self.fields.values())).to_image(filepath, width, height, dpi, format, transparent, bgcolor)

    @classmethod
    @geokube_logging
    def from_xarray(
        cls,
        ds: xr.Dataset,
        id_pattern: Optional[str] = None,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> "DataCube":
        fields = []
        #
        # we assume that data_vars contains only variable + ancillary
        # and coords all coordinates, grid_mapping and so on ...
        # TODO ancillary variables
        #
        for dv in ds.data_vars:
            fields.append(
                Field.from_xarray(
                    ds, ncvar=dv, id_pattern=id_pattern, mapping=mapping
                )
            )
        # Issue https://github.com/opengeokube/geokube/issues/221
        attrs = ds.attrs.copy()
        encoding = ds.encoding.copy()
        attrs.pop("_NCProperties", None)
        encoding.pop("_NCProperties", None)
        return DataCube(fields=fields, properties=attrs, encoding=encoding)

    @geokube_logging
    def to_xarray(self, encoding=True):
        xarray_fields = [
            f.to_xarray(encoding=encoding) for f in self.fields.values()
        ]
        dset = xr.merge(
            xarray_fields, join="outer", combine_attrs="no_conflicts"
        )
        dset.attrs = self.properties
        dset.encoding = self.encoding
        return dset

    @geokube_logging
    def to_netcdf(self, path, encoding: bool = True):
        self.to_xarray(encoding=encoding).to_netcdf(path=path)

    def persist(self, path=None) -> str:
        if path is None:
            path = os.path.join(
                tempfile.gettempdir(), f"{str(uuid.uuid4())}.nc"
            )
        if os.path.isdir(path):
            path = os.path.join(path, f"{str(uuid.uuid4())}.nc")
        if not path.endswith(".nc"):
            self._LOG.warn(
                f"Provided persistance path: `{path}` has not `.nc` extension."
                " Adding automatically!"
            )
            warnings.warn(
                f"Provided persistance path: `{path}` has not `.nc` extension."
                " Adding automatically!"
            )
            path = path + ".nc"
        self.assert_not_empty()
        self.to_netcdf(path)
        return path

    def to_dict(self, unique_values=False) -> dict:
        return {
            "domain": self.domain.to_dict(unique_values),
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }

    def assert_not_empty(self):
        if not len(self):
            self._LOG.warn("No fields in DataCube")
            raise EmptyDataError("No fields in DataCube!")
        for fname, field in self.fields.items():
            if 0 in field.shape:
                self._LOG.warn(
                    f"One of coordinate is empty for the field `{fname}`."
                    f" Shape=`{field.shape}`. Dimensions=`{field.dim_names}`!"
                )
                raise EmptyDataError(
                    f"One of coordinate is empty for the field `{fname}`."
                    f" Shape=`{field.shape}`. Dimensions=`{field.dim_names}`!"
                )
