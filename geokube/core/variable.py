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
from ..utils.attrs_encoding import is_undecodable_time_unit, parse_time_reference
from ..utils.decorators import geokube_logging
from ..utils.hcube_logger import HCubeLogger
from .axis import Axis, AxisType
from .unit import Unit


def _decode_month_year_reference(values, unit_str, calendar=None):
    """Decode raw numeric offsets against a non-CF ``"months since"`` / ``"years since"``
    reference into ``datetime64[ns]`` using pandas calendar arithmetic.

    xarray/cftime cannot decode month/year reference units for real-world calendars
    (variable month length; cftime only supports ``360_day``), so datasets with such units
    are opened raw (``decode_times=False``) and decoded here. Fractional offsets are rounded
    to whole calendar steps -- monthly/yearly climate data uses integer offsets and a
    fractional month has no exact calendar length. Returns ``None`` if ``unit_str`` is not a
    month/year reference (nothing to decode). Preserves the input array shape (so ``time_bnds``
    of shape ``(n, 2)`` round-trips).
    """
    parsed = parse_time_reference(unit_str)
    if parsed is None:
        return None
    step, ref_str = parsed
    if step not in ("month", "year"):
        return None
    arr = np.asarray(values)
    off = np.rint(arr.astype("float64")).astype("int64").ravel()
    ref = pd.Timestamp(ref_str)
    if step == "year":
        years, months = ref.year + off, np.full(off.shape, ref.month, dtype="int64")
    else:
        total = (ref.year * 12 + (ref.month - 1)) + off
        years, months = total // 12, total % 12 + 1
    if ref.day <= 28:
        result = pd.to_datetime(
            {"year": years, "month": months, "day": ref.day,
             "hour": ref.hour, "minute": ref.minute, "second": ref.second}
        ).values
    else:
        # ``ref.day`` may not exist in every target month (e.g. day 31 -> February); let
        # pandas' DateOffset clamp to the valid month-end, per element.
        offset = pd.DateOffset(years=1) if step == "year" else pd.DateOffset(months=1)
        result = np.array(
            [(ref + int(n) * offset).to_datetime64() for n in off],
            dtype="datetime64[ns]",
        )
    return result.reshape(arr.shape).astype("datetime64[ns]")


class Variable(xr.Variable):
    __slots__ = (
        "_dimensions",
        "_units",
    )

    _LOG = HCubeLogger(name="Variable")

    @property
    def nbytes(self) -> int:
        # Estimate the *storage* footprint rather than the decoded in-memory
        # size: when the variable is packed for serialization (e.g.
        # scale_factor/add_offset -> int16) use the encoded dtype's itemsize,
        # so the estimate reflects the persisted size (consistent with how
        # Dataset.nbytes reports on-disk sizes for already-persisted cubes).
        # Falls back to the in-memory dtype when no packing is declared.
        enc_dtype = self.encoding.get("dtype")
        itemsize = (
            np.dtype(enc_dtype).itemsize
            if enc_dtype is not None
            else self.dtype.itemsize
        )
        return int(self.size) * itemsize

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

        units_str = encoding.pop("units", attrs.pop("units", None))
        calendar = encoding.pop("calendar", attrs.pop("calendar", None))
        units = Unit(units_str, calendar=calendar)

        # Non-CF "months since"/"years since" time (undecodable by xarray/cftime for any
        # calendar but 360_day) is opened raw (decode_times=False); decode it to datetime64
        # here -- the single funnel both the cached and non-cached read paths pass through --
        # so the whole model sees a real time axis. Guard: numeric and 1-D/2-D (a coordinate
        # axis or its bounds), never an already-decoded or a large lazy data variable.
        dtype = getattr(data, "dtype", None)
        ndim = getattr(data, "ndim", None)
        if (
            is_undecodable_time_unit(units_str, calendar)
            and dtype is not None
            and np.issubdtype(dtype, np.number)
            and ndim is not None
            and ndim <= 2
        ):
            decoded = _decode_month_year_reference(data, units_str, calendar)
            if decoded is not None:
                data = decoded

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
                units_str = str(self.units)
                calendar = getattr(self.units, "calendar", None)
                if np.issubdtype(self.dtype, np.datetime64) and is_undecodable_time_unit(
                    units_str, calendar
                ):
                    # datetime64 data decoded from a non-CF "months/years since" reference:
                    # xarray/cftime cannot re-encode that unit on write, so drop it and let
                    # xarray pick a CF-decodable encoding; keep the original for provenance.
                    nc_encoding.pop("units", None)
                    nc_encoding.pop("calendar", None)
                    nc_attrs.setdefault("original_time_units", units_str)
                else:
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
