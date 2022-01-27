from enum import Enum
from numbers import Number
from typing import Any, Hashable, Iterable, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
from geokube.utils.type_utils import AllowedDataType, OptStrMapType
from geokube.utils import util_methods
import xarray as xr

import geokube.utils.exceptions as ex
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger

from .axis import Axis, AxisType
from .enums import LatitudeConvention, LongitudeConvention
from .variable import Variable
from .unit import Unit


class CoordinateType(Enum):
    SCALAR = "scalar"
    DEPENDENT = "dependent"  # equivalent to CF AUXILIARY Coordinate
    INDEPENDENT = "independent"  # equivalent to CF DIMENSION Coordinate


#
# coordinate is a dimension or axis with data and units
# coordinate name is dimension/axis name
# coordinate axis type is dimension/axis type
#

# class Coordinate(Variable, Dimension):
# __slots__ = ("_bounds")
#    def __init__(
#        self,
#        name
#        data: Union[np.ndarray, da.Array, Variable],
# axis: Optional[Union[str, AxisType, Axis, Dimension]],
# dims: Optional[Tuple[Dimension]] = None,
# units: Optional[Union[Unit, str]] = None,
# bounds: Optional[Union[np.ndarray, da.Array, Variable]] = None,
# properties: Optional[Mapping[Any, Any]] = None,
# encoding: Optional[Mapping[Any, Any]] = None

# )


class Coordinate(Variable, Axis):
    __slots__ = ("_bounds",)

    _LOG = HCubeLogger(name="Coordinate")

    def __init__(
        self,
        data: AllowedDataType,
        axis: Union[str, Axis],
        dims: Optional[Tuple[Axis]] = None,
        units: Optional[Union[Unit, str]] = None,
        bounds: Optional[AllowedDataType] = None,
        properties: OptStrMapType = None,
        encoding: OptStrMapType = None,
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
            self.encoding.update(encoding)
        if not self.is_dim and dims is None and not isinstance(data, Number):
            raise ex.HCubeValueError(
                "If coordinate is not a dimension, you need to supply `dims` argument!",
                logger=Coordinate._LOG,
            )
        if self.is_dim and (dims is None or len(dims) == 0):
            dims = ()
        Variable.__init__(
            self,
            data=data,
            dims=dims,
            units=units,
            properties=properties,
            encoding=self.encoding,
        )
        self._bounds = Coordinate._process_bounds(
            bounds,
            name=self.name,
            variable_shape=self.shape,
            units=self.units,
            axis=(Axis)(self),
        )

    @classmethod
    @log_func_debug
    def _process_bounds(cls, bounds, name, variable_shape, units, axis):
        if bounds is None:
            return None
        if isinstance(bounds, dict):
            if len(bounds) > 0:
                _bounds = {}
            for k, v in bounds.items():
                if isinstance(v, Variable):
                    Coordinate._assert_dims_compliant(v.shape, (variable_shape[0], 2))
                    _bounds[f"{k}_bounds"] = v
                elif isinstance(v, (np.ndarray, da.Array)):
                    # in this case when only a numpy array is passed
                    # we assume 2-D numpy array with shape(coord.dim, 2)
                    #
                    Coordinate._assert_dims_compliant(v.shape, (variable_shape[0], 2))
                    _bounds[f"{k}_bounds"] = Variable(
                        data=v,
                        units=units,
                        dims=(axis, Axis("bounds", AxisType.GENERIC)),
                    )
                else:
                    raise ex.HCubeTypeError(
                        f"Each defined bound is expected to be one of the following types `geokube.Variable`, `numpy.array`, or `dask.Array`, but provided {type(bounds)}",
                        logger=Coordinate._LOG,
                    )
        elif isinstance(bounds, Variable):
            Coordinate._assert_dims_compliant(bounds.shape, (variable_shape[0], 2))
            _bounds = {f"{name}_bounds": bounds}
        elif isinstance(bounds, (np.ndarray, da.Array)):
            Coordinate._assert_dims_compliant(bounds.shape, (variable_shape[0], 2))
            _bounds = {
                f"{name}_bounds": Variable(
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
    def _assert_dims_compliant(cls, provided_shape, required_shape):
        if not util_methods.are_dims_compliant(provided_shape, required_shape):
            raise ex.HCubeValueError(
                f"Expected shape is `{required_shape}` but provided one is `{provided_shape}`",
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
        return self._bounds if self._bounds is not None else {}

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

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

    # @classmethod
    # @log_func_debug
    # def to_xarray_with_bounds(cls,
    # )

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
        name = Variable._get_name(ds[ncvar], mapping, id_pattern)
        var.encoding.update(name=ncvar)
        axis = Axis(da.attrs.get("axis", Variable._get_name(da, mapping, id_pattern)))
        if ncvar in da.dims:
            axis = Dimension(
                name=axis.name, axistype=axis.type, encoding={"name": ncvar}
            )
        bnds_ncvar = da.encoding.pop("bounds", da.attrs.pop("bounds", None))
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
                bounds[bnds_name].units = var.units
        else:
            bounds = None
        return Coordinate(data=var, axis=axis, bounds=bounds)
