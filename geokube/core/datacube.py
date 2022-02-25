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

import numpy as np
import pyarrow as pa
import xarray as xr
import pandas as pd

from ..utils import exceptions as ex
from ..utils.decorators import log_func_debug
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis
from .domain import Domain
from .enums import RegridMethod
from .field import Field
from .domainmixin import DomainMixin

IndexerType = Union[slice, List[slice], Number, List[Number]]

# TODO: Not a priority
# dc = datacube.set_id_pattern('{standard_name}')
# dc['air_temperature']
# dc['latitude']
#
class DataCube(DomainMixin):

    __slots__ = ("_fields", "_domain", "_properties", "_encoding", "_ncvar_to_name")

    _LOG = HCubeLogger(name="DataCube")

    def __init__(
        self,
        fields: List[Field],
        properties: Mapping[Any, Any],
        encoding: Mapping[Any, Any],
    ) -> None:
        if len(fields) == 0:
            warnings.warn("No fields provided for the DataCube!")
            self._fields = pd.Series()
            self._domain = None
            self._ncvar_to_name = None
        else:
            multilevel_index = pd.MultiIndex.from_tuples(
                tuple(zip([f.name for f in fields], [f.ncvar for f in fields])),
                names=["name", "ncvar"],
            )
            self._fields = pd.Series(fields, index=multilevel_index)
            self._ncvar_to_name = {f.ncvar: f.name for f in fields}
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
    def domain(self):
        return self._domain

    @property
    def fields(self):
        return self._fields.values

    @property
    def nbytes(self) -> int:
        return sum(field.nbytes for field in self.fields)

    @property
    def _name_index(self):
        return self._fields.index.get_level_values("name")

    @property
    def _ncvar_index(self):
        return self._fields.index.get_level_values("ncvar")

    def __len__(self):
        return len(self._fields)

    def __contains__(self, key: str) -> bool:
        return (
            (key in self._name_index)
            or (key in self._ncvar_index)
            or (key in self._domain)
        )

    def _get_index_level(self, key: str) -> Optional[str]:
        if key in self._name_index:
            return "name"
        elif key in self._ncvar_index:
            return "ncvar"
        return None

    def __getitem__(
        self, key: Union[Iterable[str], Iterable[Tuple[str, str]], str, Tuple[str, str]]
    ):
        if isinstance(key, str):
            index_level = self._get_index_level(key)
            if index_level is None:
                return self.domain[key]
            selected_field = self._fields.xs(key=key, level=index_level).values
            if len(selected_field) > 1:
                raise ex.HCubeKeyError(
                    f"There are multiple fields withe name `{key}`! Use `ncvar` or combination `(name, ncvar)`!",
                    logger=DataCube._LOG,
                )
            return selected_field.item()
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ex.HCubeValueError(
                    f"Tuple index should have exactly two values: (name, ncvar)!",
                    logger=DataCube._LOG,
                )
            if key[0] not in self._name_index:
                raise ex.HCubeKeyError(
                    f"`{key[0]}` not found among fields' names!", logger=DataCube._LOG
                )
            if key[1] not in self._ncvar_index:
                raise ex.HCubeKeyError(
                    f"`{key[0]}` not found among fields' ncvars!", logger=DataCube._LOG
                )
            return self._fields[key].values
        elif isinstance(key, Iterable) and not isinstance(key, str):
            return DataCube(
                fields=[self[k] for k in key],
                properties=self.properties,
                encoding=self.encoding,
            )
        return self.domain[key]

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
            properties=self.properties,
            encoding=self.encoding,
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
            properties=self.properties,
            encoding=self.encoding,
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
        xarray_fields = [f.to_xarray(encoding=encoding) for f in self.fields]
        dset = xr.merge(xarray_fields, join="outer", combine_attrs="no_conflicts")
        dset.attrs = self.properties
        dset.encoding = self.encoding
        return dset

    @log_func_debug
    def to_netcdf(self, path):
        self.to_xarray().to_netcdf(path=path)
