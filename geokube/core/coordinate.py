from enum import Enum
from numbers import Number
from typing import Any, Hashable, Iterable, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import xarray as xr

from ..utils.decorators import geokube_logging
from ..utils.hcube_logger import HCubeLogger
from ..utils.attrs_encoding import CFAttributes
from ..utils.serialization import maybe_convert_to_json_serializable
from .bounds import Bounds, Bounds1D, BoundsND
from .axis import Axis, AxisType
from .enums import LatitudeConvention, LongitudeConvention
from .unit import Unit
from .variable import Variable


class CoordinateType(Enum):
    SCALAR = "scalar"
    DEPENDENT = "dependent"  # equivalent to CF AUXILIARY Coordinate
    INDEPENDENT = "independent"  # equivalent to CF DIMENSION Coordinate


# NOTE: coordinate is a dimension or axis with data and units
# NOTE: coordinate name is dimension/axis name
# NOTE: coordinate axis type is dimension/axis type

FREQ_CODES = {
    "T": "minute",
    "H": "hour",
    "D": "day",
    "M": "month",
    "Y": "year",
    "N": "nanosecond",
}


class Coordinate(Variable, Axis):
    __slots__ = ("_bounds",)

    _LOG = HCubeLogger(name="Coordinate")

    def __init__(
        self,
        data: Union[np.ndarray, da.Array, xr.Variable],
        axis: Union[str, Axis],
        dims: Optional[Tuple[Axis]] = None,
        units: Optional[Union[Unit, str]] = None,
        bounds: Optional[
            Union[Bounds, np.ndarray, da.Array, xr.Variable]
        ] = None,
        properties: Optional[Mapping[Hashable, str]] = None,
        encoding: Optional[Mapping[Hashable, str]] = None,
    ):
        if data is None:
            raise ValueError("`data` cannot be `None`")
        if not isinstance(axis, (Axis, str)):
            raise TypeError(
                "Expected argument is one of the following types"
                f" `geokube.Axis` or `str`, but provided {type(data)}"
            )
        Axis.__init__(self, name=axis, is_dim=dims is None)
        # We need to update as when calling constructor of Variable, encoding will be overwritten
        if encoding is not None:
            # encoding stored in axis
            self.encoding.update(encoding)
        if (
            not self.is_dim
            and dims is None
            and not isinstance(data, Number)
            and not hasattr(data, "dims")
        ):
            raise ValueError(
                "If coordinate is not a dimension, you need to supply `dims`"
                " argument!",
            )
        if self.is_dim:
            if isinstance(dims, (list, tuple)):
                dims_names = [Axis.get_name_for_object(o) for o in dims]
                dims_tuple = tuple(dims_names)
            elif isinstance(dims, str):
                dims_tuple = (Axis.get_name_for_object(dims),)
            else:
                dims_tuple = ()
            if dims is None or len(dims_tuple) == 0:
                dims = (self.name,)
            else:
                if dims is not None and len(dims_tuple) > 1:
                    raise ValueError(
                        "If the Coordinate is a dimension, it has to depend"
                        f" only on itself, but provided `dims` are: {dims}",
                    )
                if len(dims_tuple) == 1 and dims_tuple[0] != self.name:
                    raise ValueError(
                        "`dims` parameter for dimension coordinate should"
                        " have the same name as axis name!"
                    )
        Variable.__init__(
            self,
            data=data,
            dims=dims,
            units=units if units is not None else self.default_unit,
            properties=properties,
            encoding=self.encoding,
        )
        # Coordinates are always stored as NumPy data
        # import pdb;pdb.set_trace()
        self._data = np.array(self._data)
        self.bounds = bounds
        self._update_properties_and_encoding()

    def __hash__(self):
        # NOTE: maybe hash for Cooridnate should be more complex.
        return Axis.__hash__(self)

    def __eq__(self, other):
        # NOTE: it doesn't take into account real values at all
        return Axis.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def _update_properties_and_encoding(self):
        if CFAttributes.STANDARD_NAME.value not in self.properties:
            self.properties[
                CFAttributes.STANDARD_NAME.value
            ] = self.axis_type.axis_type_name
        if CFAttributes.NETCDF_NAME.value not in self.encoding:
            self.encoding[CFAttributes.NETCDF_NAME.value] = self.ncvar
        Coordinate._handle_fill_value_encoding(self)

    @classmethod
    def _handle_fill_value_encoding(cls, obj):
        # NOTE: _FillValue is not applicable for Coordinate
        # as it shouldn't contain missing data
        # To avoid implicit addition of _FillValue encoding
        # it needs to be set to False for netcdf4 engine
        # see https://github.com/pydata/xarray/issues/1598
        if hasattr(obj, "encoding"):
            obj.encoding[CFAttributes.FILL_VALUE.value] = None

    @classmethod
    @geokube_logging
    def _process_bounds(cls, bounds, name, variable_shape, units, axis):
        if bounds is None:
            return None
        if isinstance(bounds, dict):
            if len(bounds) > 0:
                _bounds = {}
            for k, v in bounds.items():
                if isinstance(v, pd.core.indexes.datetimes.DatetimeIndex):
                    v = np.array(v)
                if isinstance(v, Bounds):
                    bound_class = Coordinate._get_bounds_cls(
                        v.shape, variable_shape
                    )
                    _bounds[k] = v
                if isinstance(v, Variable):
                    bound_class = Coordinate._get_bounds_cls(
                        v.shape, variable_shape
                    )
                    _bounds[k] = bound_class(data=v)
                elif isinstance(v, (np.ndarray, da.Array)):
                    # in this case when only a numpy array is passed
                    # we assume 2-D numpy array with shape(coord.dim, 2)
                    #
                    bound_class = Coordinate._get_bounds_cls(
                        v.shape, variable_shape
                    )
                    _bounds[k] = bound_class(
                        data=v,
                        units=units,
                        dims=(axis, Axis("bounds", AxisType.GENERIC)),
                    )
                else:
                    raise TypeError(
                        "Each defined bound is expected to be one of the"
                        " following types `geokube.Variable`, `numpy.array`,"
                        f" or `dask.Array`, but provided {type(bounds)}"
                    )
        elif isinstance(bounds, Bounds):
            bound_class = Coordinate._get_bounds_cls(
                bounds.shape, variable_shape
            )
            _bounds = {f"{name}_bounds": bounds}
        elif isinstance(bounds, Variable):
            bound_class = Coordinate._get_bounds_cls(
                bounds.shape, variable_shape
            )
            _bounds = {f"{name}_bounds": bound_class(bounds)}
        elif isinstance(bounds, (np.ndarray, da.Array)):
            bound_class = Coordinate._get_bounds_cls(
                bounds.shape, variable_shape
            )
            _bounds = {
                f"{name}_bounds": bound_class(
                    data=bounds,
                    units=units,
                    dims=(axis, Axis("bounds", AxisType.GENERIC)),
                )
            }
        else:
            raise TypeError(
                "Expected argument is one of the following types `dict`,"
                " `numpy.ndarray`, or `geokube.Variable`, but provided"
                f" {type(bounds)}"
            )
        return _bounds

    @classmethod
    def _is_valid_1d_bounds(cls, provided_bnds_shape, provided_data_shape):
        ndim = len(provided_bnds_shape) - 1
        if (
            2 * ndim == 2
            and provided_bnds_shape[-1] == 2
            and provided_bnds_shape[0] == provided_data_shape[0]
        ):
            return True
        if (
            provided_data_shape == ()
            and ndim == 0
            and provided_bnds_shape[0] == 2
        ):
            # The case where there is a scalar coordinate with bounds, e.g.
            # after single value selection
            return True
        return False

    @classmethod
    def _is_valid_nd_bounds(cls, provided_bnds_shape, provided_data_shape):
        ndim = len(provided_bnds_shape) - 1
        if (
            provided_bnds_shape[-1] == 2 * ndim
            and tuple(provided_bnds_shape[:-1]) == provided_data_shape
        ):
            return True
        if (
            len(provided_bnds_shape) == 2
            and len(provided_data_shape) == 1
            and provided_bnds_shape[0] == provided_data_shape[0]
            and (provided_bnds_shape[1] == 2 or provided_bnds_shape[1] == 4)
        ):
            # The case of points domain
            return True
        return False

    @classmethod
    @geokube_logging
    def _get_bounds_cls(cls, provided_bnds_shape, provided_data_shape):
        if cls._is_valid_1d_bounds(provided_bnds_shape, provided_data_shape):
            return Bounds1D
        if cls._is_valid_nd_bounds(provided_bnds_shape, provided_data_shape):
            return BoundsND
        raise ValueError(
            "Bounds should have dimensions: (2,), (N,2), (N,M,4), (N,M,L,6),"
            f" ... Provided shape is `{provided_bnds_shape}`"
        )

    @property
    def is_dimension(self) -> bool:
        return super().is_dim

    @property
    def is_independent(self) -> bool:
        return self.is_dimension or self.type is CoordinateType.SCALAR

    @property
    def is_dependent(self) -> bool:
        return not self.is_independent

    @property
    def type(self):
        # Cooridnate is scalar if data shows so. Dim(s) --  always defined
        if self.is_dimension:
            if self.shape == () or self.shape == (1,):
                return CoordinateType.SCALAR
            else:
                return CoordinateType.INDEPENDENT
        elif self.shape == ():
            return CoordinateType.SCALAR
        else:
            return CoordinateType.DEPENDENT

    @property
    def axis_type(self):
        return self._type

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = Coordinate._process_bounds(
            value,
            name=self.name,
            variable_shape=self.shape,
            units=self.units,
            axis=(Axis)(self),
        )
        if self._bounds is not None:
            self.encoding["bounds"] = next(iter(self.bounds))

    @property
    def has_bounds(self) -> bool:
        return self._bounds is not None

    @property
    # TODO:  check! I think this works only if lat/lon are independent!
    def convention(
        self,
    ) -> Optional[Union[LatitudeConvention, LongitudeConvention]]:
        if self.axis_type is AxisType.LATITUDE:
            return (
                LatitudeConvention.POSITIVE_TOP
                if self.first() > self.last()
                else LatitudeConvention.NEGATIVE_TOP
            )
        if self.axis_type is AxisType.LONGITUDE:
            return (
                LongitudeConvention.POSITIVE_WEST
                if self.min() >= 0
                else LongitudeConvention.NEGATIVE_WEST
            )

    @geokube_logging
    def to_xarray(
        self, encoding=True
    ) -> xr.core.coordinates.DatasetCoordinates:
        var = Variable.to_xarray(self, encoding=encoding)
        Coordinate._handle_fill_value_encoding(var)
        res_name = self.ncvar if encoding else self.name
        dim_names = self.dim_ncvars if encoding else self.dim_names
        da = xr.DataArray(
            var, name=res_name, coords={res_name: var}, dims=dim_names
        )[res_name]
        if self.has_bounds:
            bounds = {
                k: xr.DataArray(
                    Variable.to_xarray(b, encoding=encoding), name=k
                )
                for k, b in self.bounds.items()
            }
            da.encoding["bounds"] = " ".join(bounds.keys())
        else:
            bounds = {}
        return xr.Dataset(coords={da.name: da, **bounds})

    def to_dict(self, unique_values=False):
        axis_specific_details = {}
        values = self.data
        if self.axis_type is AxisType.TIME:
            values = np.array(values).astype(np.datetime64)
            time_unit = time_step = None
            if len(self.data) > 1:
                time_offset = to_offset(pd.Series(values).diff().mode()[0])
                time_unit = time_offset.name
                time_step = time_offset.n
                if time_unit in {
                    "L",
                    "U",
                    "N",
                }:  # skip mili, micro, and nanoseconds
                    values = values.astype(
                        "datetime64[m]"
                    )  # with minute resoluton
                    time_offset = to_offset(
                        pd.Series(values).diff().mode().item()
                    )
                    time_unit = time_offset.name
                    time_step = time_offset.n
                axis_specific_details = {
                    "time_unit": FREQ_CODES[time_unit],
                    "time_step": time_step,
                }
        elif (
            self.axis_type is AxisType.VERTICAL
            or self.axis_type is AxisType.GENERIC
        ):
            # e.g. numpy.float32 is not JSON serializable
            axis_specific_details = {
                "values": maybe_convert_to_json_serializable(
                    np.unique(np.array(values))
                )
                if unique_values
                else maybe_convert_to_json_serializable(
                    np.array(np.atleast_1d(values))
                )
            }
        return dict(
            **{
                "min": maybe_convert_to_json_serializable(np.nanmin(values)),
                "max": maybe_convert_to_json_serializable(np.nanmax(values)),
                "units": str(self.units),
                "axis": self.axis_type.name,
            },
            **axis_specific_details,
        )

    @classmethod
    @geokube_logging
    def from_xarray(
        cls,
        ds: xr.Dataset,
        ncvar: str,
        id_pattern: Optional[str] = None,
        mapping: Optional[Mapping[str, str]] = None,
        copy: Optional[bool] = False,
    ) -> "Coordinate":
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"Expected type `xarray.Dataset` but provided `{type(ds)}`"
            )

        da = ds[ncvar]
        var = Variable.from_xarray(da, id_pattern=id_pattern, mapping=mapping)
        encoded_ncvar = da.encoding.get("name", ncvar)
        var.encoding.update(name=encoded_ncvar)

        axis_name = Variable._get_name(da, mapping, id_pattern)
        # `axis` attribute cannot be used below, as e.g for EOBS `latitude` has axis `Y`, so wrong AxisType is chosen
        axistype_name = None
        if mapping is not None and da.name in mapping:
            axistype_name = mapping[da.name].get("axis")
        if axistype_name is None:
            axistype_name = da.attrs.get("standard_name", ncvar)
        axistype = AxisType.parse(axistype_name)
        axis = Axis(
            name=axis_name,
            is_dim=ncvar in da.dims,
            axistype=axistype,
            encoding={"name": encoded_ncvar},
        )
        bnds_ncvar = da.encoding.get("bounds", da.attrs.get("bounds"))
        if bnds_ncvar:
            bnds_name = Variable._get_name(ds[bnds_ncvar], mapping, id_pattern)
            bounds = {
                bnds_name: Variable.from_xarray(
                    ds[bnds_ncvar],
                    id_pattern=id_pattern,
                    copy=copy,
                    mapping=mapping,
                )
            }
            if (
                "units" not in ds[bnds_ncvar].attrs
                and "units" not in ds[bnds_ncvar].encoding
            ):
                bounds[bnds_name]._units = var.units
        else:
            bounds = None
        return Coordinate(data=var, axis=axis, bounds=bounds)


class ArrayCoordinate(Coordinate):
    pass


class ParametricCoordinate(Coordinate):
    pass
