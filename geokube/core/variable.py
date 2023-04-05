from __future__ import annotations

import warnings
from html import escape
from numbers import Number
from string import Formatter, Template
from typing import (
    Any,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dask.array as da
import numpy as np
import xarray as xr
import pandas as pd
from xarray.core.options import OPTIONS

from ..utils import formatting, formatting_html, util_methods
from ..utils.decorators import geokube_logging
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis, AxisType
from .unit import Unit


class Variable(xr.Variable):
    __slots__ = (
        "_dimensions",
        "_units",
    )

    _LOG = HCubeLogger(name="Variable")

    def __init__(
        self,
        data: Union[np.ndarray, da.Array, xr.Variable, Number, Variable],
        dims: Optional[Union[Tuple[Axis], Tuple[AxisType], Tuple[str]]] = None,
        units: Optional[Union[Unit, str]] = None,
        properties: Optional[Mapping[Hashable, str]] = None,
        encoding: Optional[Mapping[Hashable, str]] = None,
    ):
        if isinstance(data, pd.core.indexes.datetimes.DatetimeIndex):
            data = np.array(data)
        if not (
            isinstance(data, np.ndarray)
            or isinstance(data, da.Array)
            or isinstance(data, Variable)
            or isinstance(data, Number)
        ):
            raise TypeError(
                "Expected argument is one of the following types"
                " `number.Number`, `numpy.ndarray`, `dask.array.Array`, or"
                f" `xarray.Variable`, but provided {type(data)}"
            )
        _is_scalar = False
        if isinstance(data, Number):
            data = np.array(data, ndmin=1)
            _is_scalar = True
        if isinstance(data, Variable):
            self._dimensions = data._dimensions
            self._units = data._units
            super().__init__(
                data=data.data,
                dims=data.dim_names,
                attrs=data.properties,
                encoding=data.encoding,
            )
        else:
            self._dimensions = None
            if dims is not None:
                dims = self._as_dimension_tuple(dims)
                dims = np.array(dims, ndmin=1, dtype=Axis)
                if (not _is_scalar) and len(dims) != data.ndim:
                    raise ValueError(
                        f"Provided data have {data.ndim} dimension(s) but"
                        f" {len(dims)} Dimension(s) provided in `dims`"
                        " argument"
                    )

                self._dimensions = dims
            # xarray.Variable must be created with non-None `dims`
            super().__init__(
                data=data,
                dims=self.dim_names,
                attrs=properties,
                encoding=encoding,
                fastpath=True,
            )
            self._units = (
                Unit(units)
                if isinstance(units, str) or units is None
                else units
            )

    def _as_dimension_tuple(self, dims) -> Tuple[Axis, ...]:
        if isinstance(dims, str):
            return (Axis(dims, is_dim=True),)
        elif isinstance(dims, Axis):
            return (dims,)
        elif isinstance(dims, AxisType):
            return (Axis(dims.axis_type_name, axistype=dims, is_dim=True),)
        elif isinstance(dims, Iterable):
            _dims = []
            for d in dims:
                if isinstance(d, str):
                    _dims.append(Axis(name=d, is_dim=True))
                elif isinstance(d, AxisType):
                    _dims.append(
                        Axis(name=d.axis_type_name, axistype=d, is_dim=True)
                    )
                elif isinstance(d, Axis):
                    _dims.append(d)
                else:
                    raise TypeError(
                        "Expected argument of collection item is one of the"
                        " following types `str` or `geokube.Axis`, but"
                        f" provided {type(d)}"
                    )
            return tuple(_dims)
        raise ValueError(
            "Expected argument is one of the following types `str`, `iterable"
            " of str`, `iterable of geokub.Axis`, or `iterable of str`, but"
            f" provided {type(dims)}"
        )

    @property
    def dims(self) -> Tuple[Axis, ...]:
        return self._dimensions

    @property
    def dim_names(self):
        return (
            tuple([d.name for d in self._dimensions])
            if self._dimensions is not None
            else ()
        )

    @property
    def dim_ncvars(self):
        return (
            tuple([d.ncvar for d in self._dimensions])
            if self._dimensions is not None
            else ()
        )

    @property
    def properties(self):
        return self.attrs

    @property
    def units(self) -> Unit:
        return self._units

    def __repr__(self) -> str:
        return self.to_xarray(encoding=False).__repr__()

    def _repr_html_(self):
        return self.to_xarray(encoding=False)._repr_html_()

    def convert_units(self, unit, inplace=True):
        unit = Unit(unit) if isinstance(unit, str) else unit
        if not isinstance(self.data, np.ndarray):
            Variable._LOG.warn(
                "Converting units is supported only for np.ndarray inner data"
                " type. Data will be loaded into the memory!"
            )
            self.data = np.array(
                self.data
            )  # TODO: inplace for cf.Unit doesn't work!
        res = self.units.convert(self.data, unit, inplace)
        if not inplace:
            return Variable(
                data=res,
                dims=self.dims,
                units=unit,
                properties=self.properties,
                encoding=self.encoding,
            )
        self.data = res
        self.units = unit

    @classmethod
    @geokube_logging
    def _get_name(
        cls,
        da: Union[xr.Dataset, xr.DataArray],
        mapping: Optional[Mapping[Hashable, str]],
        id_pattern: str,
    ) -> str:
        if mapping is not None and da.name in mapping:
            return mapping[da.name].get("name", da.name)
        if id_pattern is None:
            return da.attrs.get("standard_name", da.name)
        fmt = Formatter()
        _, field_names, _, _ = zip(*fmt.parse(id_pattern))
        field_names = [f for f in field_names if f]
        # Replace intake-like placeholder to string.Template-like ones
        for k in field_names:
            if k not in da.attrs:
                warnings.warn(
                    f"Requested id_pattern component - `{k}` is not present"
                    " among provided attributes!"
                )
                return da.name
            id_pattern = id_pattern.replace(
                f"{{{k}}}", f"${{{k}}}"
            )  # "{some_field}" -> "${some_field}"
        template = Template(id_pattern)
        return template.substitute(**da.attrs)

    @classmethod
    @geokube_logging
    def from_xarray(
        cls,
        da: xr.DataArray,
        id_pattern: Optional[str] = None,
        copy: Optional[bool] = False,
        mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    ):
        if not isinstance(da, xr.DataArray):
            raise TypeError(
                "Expected argument of the following type `xarray.DataArray`,"
                f" but provided {type(da)}"
            )
        data = da.data.copy() if copy else da.data
        dims = []
        for d in da.dims:
            if d in da.coords:
                d_name = Variable._get_name(da[d], mapping, id_pattern)
                # If id_pattern is defined, AxisType might be improperly parsed (to GENERIC)
                d_axis = da[d].attrs.get("axis", AxisType.parse(d))
                dims.append(
                    Axis(
                        name=d_name,
                        axistype=d_axis,
                        encoding={"name": da[d].encoding.get("name", d)},
                        is_dim=True,
                    )
                )
            else:
                dims.append(Axis(name=d, is_dim=True))

        dims = tuple(dims)
        attrs = da.attrs.copy()
        encoding = da.encoding.copy()

        units = Unit(
            encoding.pop("units", attrs.pop("units", None)),
            calendar=encoding.pop("calendar", attrs.pop("calendar", None)),
        )

        return Variable(
            data=data,
            dims=dims,
            units=units,
            properties=attrs,
            encoding=encoding,
        )

    @geokube_logging
    def to_xarray(self, encoding=True) -> xr.Variable:
        nc_attrs = self.properties
        nc_encoding = self.encoding
        if encoding:
            dims = self.dim_ncvars
        else:
            dims = self.dim_names
        if self.units is not None and not self.units.is_unknown:
            if self.units.is_time_reference():
                nc_encoding["units"] = self.units.cftime_unit
                nc_encoding["calendar"] = self.units.calendar
            elif np.issubdtype(self.dtype, np.timedelta64) or np.issubdtype(
                self.dtype, np.datetime64
            ):
                # NOTE: issue while using xarray.to_netcdf if units
                # are stored as attributes,
                # example: fapar/10-daily/LENGTH_AFTER
                nc_encoding["units"] = str(self.units)
            else:
                nc_attrs["units"] = str(self.units)

        return xr.Variable(
            data=self._data,
            dims=dims,
            attrs=nc_attrs,
            encoding=nc_encoding,
            fastpath=True,
        )
