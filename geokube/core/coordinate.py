from enum import Enum
from numbers import Number
from typing import Any, Hashable, Iterable, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from ..utils import exceptions as ex
from ..utils.decorators import log_func_debug
from ..utils.hcube_logger import HCubeLogger
from .bounds import Bounds, Bounds1D, BoundsND
from .axis import Axis, AxisType
from .enums import LatitudeConvention, LongitudeConvention
from .unit import Unit
from .variable import Variable


class CoordinateType(Enum):
    SCALAR = "scalar"
    DEPENDENT = "dependent"  # equivalent to CF AUXILIARY Coordinate
    INDEPENDENT = "independent"  # equivalent to CF DIMENSION Coordinate


# coordinate is a dimension or axis with data and units
# coordinate name is dimension/axis name
# coordinate axis type is dimension/axis type


class Coordinate(Variable, Axis):
    __slots__ = ("_bounds",)

    _LOG = HCubeLogger(name="Coordinate")

    def __init__(
        self,
        data: Union[np.ndarray, da.Array, xr.Variable],
        axis: Union[str, Axis],
        dims: Optional[Tuple[Axis]] = None,
        units: Optional[Union[Unit, str]] = None,
        bounds: Optional[Union[Bounds, np.ndarray, da.Array, xr.Variable]] = None,
        properties: Optional[Mapping[Hashable, str]] = None,
        encoding: Optional[Mapping[Hashable, str]] = None,
    ):
        if data is None:
            raise ex.HCubeValueError("`data` cannot be `None`", logger=Coordinate._LOG)
        if not isinstance(axis, (Axis, str)):
            raise ex.HCubeTypeError(
                f"Expected argument is one of the following types `geokube.Axis` or `str`, but provided {type(data)}",
                logger=Coordinate._LOG,
            )
        Axis.__init__(self, name=axis)
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
            raise ex.HCubeValueError(
                "If coordinate is not a dimension, you need to supply `dims` argument!",
                logger=Coordinate._LOG,
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
                    raise ex.HCubeValueError(
                        f"If the Coordinate is a dimension, it has to depend only on itself, but provided `dims` are: {dims}",
                        logger=Coordinate._LOG,
                    )
                if len(dims_tuple) == 1 and dims_tuple[0] != self.name:
                    raise ex.HCubeValueError(
                        f"`dims` parameter for dimension coordinate should have the same name as axis name!",
                        logger=Coordinate._LOG,
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
        if "standard_name" not in self.properties:
            self.properties["standard_name"] = self.axis_type.axis_type_name
        if "name" not in self.encoding:
            self.encoding["name"] = self.ncvar

    @classmethod
    @log_func_debug
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
                    bound_class = Coordinate._get_bounds_cls(v.shape, variable_shape)
                    _bounds[k] = v
                if isinstance(v, Variable):
                    bound_class = Coordinate._get_bounds_cls(v.shape, variable_shape)
                    _bounds[k] = bound_class(data=v)
                elif isinstance(v, (np.ndarray, da.Array)):
                    # in this case when only a numpy array is passed
                    # we assume 2-D numpy array with shape(coord.dim, 2)
                    #
                    bound_class = Coordinate._get_bounds_cls(v.shape, variable_shape)
                    _bounds[k] = bound_class(
                        data=v,
                        units=units,
                        dims=(axis, Axis("bounds", AxisType.GENERIC)),
                    )
                else:
                    raise ex.HCubeTypeError(
                        f"Each defined bound is expected to be one of the following types `geokube.Variable`, `numpy.array`, or `dask.Array`, but provided {type(bounds)}",
                        logger=Coordinate._LOG,
                    )
        elif isinstance(bounds, Bounds):
            bound_class = Coordinate._get_bounds_cls(bounds.shape, variable_shape)
            _bounds = {f"{name}_bounds": bounds}
        elif isinstance(bounds, Variable):
            bound_class = Coordinate._get_bounds_cls(bounds.shape, variable_shape)
            _bounds = {f"{name}_bounds": bound_class(bounds)}
        elif isinstance(bounds, (np.ndarray, da.Array)):
            bound_class = Coordinate._get_bounds_cls(bounds.shape, variable_shape)
            _bounds = {
                f"{name}_bounds": bound_class(
                    data=bounds,
                    units=units,
                    dims=(axis, Axis("bounds", AxisType.GENERIC)),
                )
            }
        else:
            raise ex.HCubeTypeError(
                f"Expected argument is one of the following types `dict`, `numpy.ndarray`, or `geokube.Variable`, but provided {type(bounds)}",
                logger=Coordinate._LOG,
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
        if provided_data_shape == () and ndim == 0 and provided_bnds_shape[0] == 2:
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
    def _get_bounds_cls(cls, provided_bnds_shape, provided_data_shape):
        if cls._is_valid_1d_bounds(provided_bnds_shape, provided_data_shape):
            return Bounds1D
        if cls._is_valid_nd_bounds(provided_bnds_shape, provided_data_shape):
            return BoundsND
        raise ex.HCubeValueError(
            f"Bounds should have dimensions: (2,), (N,2), (N,M,4), (N,M,L,6), ... Provided shape is `{provided_bnds_shape}`",
            logger=Coordinate._LOG,
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
        if self.shape == ():
            return CoordinateType.SCALAR
        else:
            return (
                CoordinateType.INDEPENDENT
                if self.is_dimension
                else CoordinateType.DEPENDENT
            )

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
            self.encoding['bounds'] = next(iter(self.bounds))

    @property
    def has_bounds(self) -> bool:
        return self._bounds is not None

    @property
    # TODO:  check! I think this works only if lat/lon are independent!
    def convention(self) -> Optional[Union[LatitudeConvention, LongitudeConvention]]:
        if self._axis.type is AxisType.LATITUDE:
            return (
                LatitudeConvention.POSITIVE_TOP
                if self.first() > self.last()
                else LatitudeConvention.NEGATIVE_TOP
            )
        if self._axis.type is AxisType.LONGITUDE:
            return (
                LongitudeConvention.POSITIVE_WEST
                if self.min() >= 0
                else LongitudeConvention.NEGATIVE_WEST
            )

    @log_func_debug
    def to_xarray(self, encoding=True) -> xr.core.coordinates.DatasetCoordinates:
        var = Variable.to_xarray(self, encoding=encoding)
        # _ = var.attrs.pop("bounds", var.encoding.pop("bounds", None))
        res_name = self.ncvar if encoding else self.name
        dim_names = self.dim_ncvars if encoding else self.dim_names
        da = xr.DataArray(var, name=res_name, coords={res_name: var}, dims=dim_names)[
            res_name
        ]
        if self.has_bounds:
            bounds = {
                k: xr.DataArray(Variable.to_xarray(b, encoding=encoding), name=k)
                for k, b in self.bounds.items()
            }
            da.encoding["bounds"] = " ".join(bounds.keys())
        else:
            bounds = {}
        return xr.Dataset(coords={da.name: da, **bounds})

    @classmethod
    @log_func_debug
    def from_xarray(
        cls,
        ds: xr.Dataset,
        ncvar: str,
        id_pattern: Optional[str] = None,
        mapping: Optional[Mapping[str, str]] = None,
        copy: Optional[bool] = False,
    ) -> "Coordinate":

        if not isinstance(ds, xr.Dataset):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.Dataset` but provided `{type(ds)}`",
                logger=cls._LOG,
            )

        da = ds[ncvar]
        var = Variable.from_xarray(da, id_pattern=id_pattern, mapping=mapping)
        encoded_ncvar = da.encoding.get("name", ncvar)
        var.encoding.update(name=encoded_ncvar)

        axis_name = Variable._get_name(da, mapping, id_pattern)
        # `axis` attribute cannot be used below, as e.g for EOBS `latitude` has axis `Y`, so wrong AxisType is chosen
        axistype = AxisType.parse(da.attrs.get("standard_name", ncvar))
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
                    ds[bnds_ncvar], id_pattern=id_pattern, copy=copy, mapping=mapping
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
