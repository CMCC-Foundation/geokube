from __future__ import annotations

from html import escape
from numbers import Number
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
from geokube.core.axis import AxisStrType, Axis, AxisType
from geokube.core.cfobject import CFObjectAbstract
import xarray as xr
from xarray.core.options import OPTIONS

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.utils import formatting, formatting_html, util_methods
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger
import geokube.utils.xarray_parser as xrp
from geokube.utils.type_utils import AllowedDataType, OptStrMapType, XrDsDaType


class Variable(xr.Variable):

    __slots__ = (
        "_dimensions",
        "_units",
    )

    _LOG = HCubeLogger(name="Variable")

    def __init__(
        self,
        data: AllowedDataType,
        dims: Optional[Union[Sequence[AxisStrType], AxisStrType]] = None,
        units: Optional[Union[Unit, str]] = None,
        properties: OptStrMapType = None,
        encoding: OptStrMapType = None,
    ):
        if not (
            isinstance(data, np.ndarray)
            or isinstance(data, da.Array)
            or isinstance(data, Variable)
            or isinstance(data, Number)
        ):
            raise ex.HCubeTypeError(
                f"Expected argument is one of the following types `number.Number`, `numpy.ndarray`, `dask.array.Array`, or `xarray.Variable`, but provided {type(data)}",
                logger=Variable._LOG,
            )
        if isinstance(data, Number):
            data = np.array(data)
        if isinstance(data, Variable):
            Variable.apply_from_other(self, data)
        else:
            self._dimensions = None
            if dims is not None:
                dims = self._as_dimension_tuple(dims)
                dims = np.array(dims, ndmin=1, dtype=Axis)
                if len(dims) != data.ndim:
                    raise ex.HCubeValueError(
                        f"Provided data have {data.ndim} dimension(s) but {len(dims)} Dimension(s) provided in `dims` argument",
                        logger=Variable._LOG,
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
                Unit(units) if isinstance(units, str) or units is None else units
            )

    def _as_dimension_tuple(self, dims) -> Tuple[Axis, ...]:
        if isinstance(dims, str):
            return (Axis(dims, is_dim=True),)
        elif isinstance(dims, Axis):
            return (dims,)
        elif isinstance(dims, Iterable):
            _dims = []
            for d in dims:
                if isinstance(d, str):
                    _dims.append(Axis(name=d, is_dim=True))
                elif isinstance(d, AxisType):
                    _dims.append(Axis(name=d.axis_type_name, axistype=d, is_dim=True))
                elif isinstance(d, Axis):
                    _dims.append(d)
                else:
                    raise ex.HCubeTypeError(
                        f"Expected argument of collection item is one of the following types `str` or `geokube.Axis`, but provided {type(d)}",
                        logger=Variable._LOG,
                    )
            return tuple(_dims)
        raise ex.HCubeValueError(
            f"Expected argument is one of the following types `str`, `iterable of str`, `iterable of geokub.Axis`, or `iterable of str`, but provided {type(dims)}",
            logger=Variable._LOG,
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
                "Converting units is supported only for np.ndarray inner data type. Data will be loaded into the memory!"
            )
            self.data = np.array(self.data)  # TODO: inplace for cf.Unit doesn't work!
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
    @log_func_debug
    def _get_name(cls, da: XrDsDaType, mapping: OptStrMapType, id_pattern: str) -> str:
        if mapping is not None and da.name in mapping:
            return mapping[da.name]["name"]
        if id_pattern is not None:
            return xrp.form_id(id_pattern, da.attrs)
        return da.attrs.get("standard_name", da.name)

    @classmethod
    @log_func_debug
    def from_xarray(
        cls,
        da: xr.DataArray,
        id_pattern: Optional[str] = None,
        copy: Optional[bool] = False,
        mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    ):
        if not isinstance(da, xr.DataArray):
            raise ex.HCubeTypeError(
                f"Expected argument of the following type `xarray.DataArray`, but provided {type(da)}",
                logger=Variable._LOG,
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
                        name=d_name, axistype=d_axis, encoding={"name": d}, is_dim=True
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

    @log_func_debug
    def to_xarray(self, encoding=True) -> xr.Variable:
        nc_attrs = self.properties
        nc_encoding = self.encoding
        if encoding:
            dims = self.dim_ncvars
            if self.units is not None and not self.units.is_unknown:
                if self.units.is_time_reference():
                    nc_encoding["units"] = self.units.cftime_unit
                    nc_encoding["calendar"] = self.units.calendar
                else:
                    nc_attrs["units"] = str(self.units)
        else:
            dims = self.dim_names

        return xr.Variable(
            data=self._data,
            dims=dims,
            attrs=nc_attrs,
            encoding=nc_encoding,
            fastpath=True,
        )

    @staticmethod
    def apply_from_other(current, other, shallow=False):
        current._dimensions = other._dimensions
        current._units = other._units
        xr.Variable.__init__(
            current,
            data=other.data,
            dims=other.dim_names,
            attrs=other.properties,
            encoding=other.encoding,
        )
