import json
import logging
import re
from types import MethodType
import warnings
from collections import defaultdict
from enum import Enum
from html import escape
from itertools import chain
from numbers import Number
from string import Template
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
from geokube.core.axis import Axis
from geokube.core.enums import RegridMethod
import pyarrow as pa
import xarray as xr

from geokube.core.domain import Domain
from geokube.core.field import Field
from geokube.utils import util_methods
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
import geokube.utils.exceptions as ex
from .domainmixin import DomainMixin


IndexerType = Union[slice, List[slice], Number, List[Number]]

# TODO: Not a priority
# dc = datacube.set_id_pattern('{standard_name}')
# dc['air_temperature']
# dc['latitude']
#
class DataCube(DomainMixin):

    __slots__ = ("_fields", "_domain", "_properties", "_encoding")

    _LOG = HCubeLogger(name="DataCube")

    def __init__(
        self,
        fields: List[Field],
        properties: Mapping[Any, Any],
        encoding: Mapping[Any, Any],
    ) -> None:
        # TO DO save only fields variables and coordinates names to build the field domain
        self._fields = {f.name: f for f in fields}
        self._domain = Domain.merge([f.domain for f in fields])
        self._properties = properties if properties is not None else {}
        self._encoding = encoding if encoding is not None else {}

    @property
    def properties(self):
        return self._properties

    @property
    def encoding(self):
        return self._encoding

    @property
    def fields(self):
        return self._fields

    @property
    def nbytes(self) -> int:
        return sum(field.nbytes for field in self.fields.values())

    def __len__(self):
        return len(self._fields)

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __getitem__(self, key: Union[Iterable[str], str]):
        if isinstance(key, str):
            return self._fields[key]
        elif isinstance(key, Iterable):
            return DataCube(fields=[self._fields[k] for k in key], **self.properties)
        else:
            raise ex.HCubeTypeError(
                f"`{type(key)}` is not a supported index for geokube.DataCube",
                logger=self._LOG,
            )

    def __next__(self):
        for f in self._fields:
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

    @log_func_debug
    def geobbox(
        self,
        north=None,
        south=None,
        west=None,
        east=None,
        top=None,
        bottom=None,
        roll_if_needed=True,
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
                    roll_if_needed=roll_if_needed,
                )
                for k in self._fields.keys()
            ],
            **self.properties,
        )

    @log_func_debug
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
            **self.properties,
        )

    @log_func_debug
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
            **self.properties,
        )

    @log_func_debug
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

    @log_func_debug
    def to_regular(
        self,
    ) -> "DataCube":
        return DataCube(
            fields=[self._fields[k].to_regular() for k in self._fields.keys()],
            properties=self.properties,
            encoding=self.encoding,
        )

    @log_func_debug
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

    @classmethod
    @log_func_debug
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
            print(dv)
            fields.append(
                Field.from_xarray(ds, ncvar=dv, id_pattern=id_pattern, mapping=mapping)
            )
        return DataCube(fields=fields, properties=ds.attrs, encoding=ds.encoding)

    @log_func_debug
    def to_xarray(self, encoding=True):
        xarray_fields = [f.to_xarray(encoding) for f in self.fields.values()]
        dset = xr.merge(xarray_fields, join="outer", combine_attrs="no_conflicts")
        dset.attrs = self.properties
        dset.encoding = self.encoding
        return dset

    @log_func_debug
    def to_netcdf(self, path):
        self.to_xarray().to_netcdf(path=path)
