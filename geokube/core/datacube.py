import json
import logging
import re
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

import pandas as pd
import xarray as xr

from ..utils.decorators import geokube_logging
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis, AxisType
from .coord_system import RegularLatLon
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

    def __iter__(self):
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
        return DataCube(
            fields=[
                self._fields[k].sel(
                    indexers=indexers,
                    roll_if_needed=roll_if_needed,
                    method=method,
                    tolerance=tolerance,
                    drop=drop,
                    **indexers_kwargs,
                )
                for k in self._fields.keys()
            ],
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
    def to_regular(
        self,
    ) -> "DataCube":
        return DataCube(
            fields=[self._fields[k].to_regular() for k in self._fields.keys()],
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
            coords = [self.longitude.item(), self.latitude.item()]
            result = {"type": "FeatureCollection", "features": []}
            for time in self.time.values.flat:
                time_ = pd.to_datetime(time).strftime("%Y-%m-%dT%H:%M")
                feature = {
                    "geometry": {"type": "Point", "coordinates": coords},
                    "properties": {"time": time_},
                }
                for field in self.fields.values():
                    value = (
                        field.sel(time=time_) if field.time.size > 1 else field
                    )
                    feature["properties"][field.name] = float(value)
                result["features"].append(feature)
        elif (
            self.domain.type is DomainType.GRIDDED or self.domain.type is None
        ):
            # HACK: The case `self.domain.type is None` is included to be able
            # to handle undefined domain types temporarily.
            result = {"data": []}
            cube = (
                self
                if isinstance(self.domain.crs, RegularLatLon)
                else self.to_regular()
            )
            axis_names = cube.domain._axis_to_name
            units = {
                field.name: str(field.units) for field in self.fields.values()
            }
            for time in self.time.values.flat:
                time_ = pd.to_datetime(time).strftime("%Y-%m-%dT%H:%M")
                time_data = {
                    "type": "FeatureCollection",
                    "date": time_,
                    "bbox": [
                        self.longitude.min().item(),  # West
                        self.latitude.min().item(),  # South
                        self.longitude.max().item(),  # East
                        self.latitude.max().item(),  # North
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
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon.item(), lat.item()],
                            },
                            "properties": {},
                        }
                        for field in self.fields.values():
                            value = field.sel(indexers=idx)
                            feature["properties"][field.name] = float(value)
                        time_data["features"].append(feature)
                result["data"].append(time_data)
        else:
            raise NotImplementedError(
                f"'self.domain.type' is {self.domain.type}, which is currently"
                f" not supported"
            )

        if target is not None:
            with open(target, mode="w") as file:
                json.dump(result, file, indent=4)

        return result

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
        return DataCube(
            fields=fields, properties=ds.attrs, encoding=ds.encoding
        )

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
    def to_netcdf(self, path):
        self.to_xarray().to_netcdf(path=path)

    @geokube_logging
    def to_dict(self) -> dict:
        # NOTE: it should return concise dict representation without returning each lat/lon/time value
        dset = self.to_xarray(encoding=True)
        return {
            "variables": list(dset.data_vars.keys()),
            "coordinates": list(dset.coords.keys()),
        }
