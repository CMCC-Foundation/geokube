import warnings
from enum import Enum
from typing import Any, Hashable, Iterable, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
from geokube.utils import util_methods
import xarray as xr

import geokube.utils.exceptions as ex
from geokube.core.agg_mixin import AggMixin
from geokube.core.axis import Axis, AxisType
from geokube.core.dimension import Dimension
from geokube.core.enums import LatitudeConvention, LongitudeConvention
from geokube.core.variable import Variable
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger


class CoordinateType(Enum):
    SCALAR = "scalar"
    DEPENDENT = "dependent"  # equivalent to CF AUXILIARY Coordinate
    INDEPENDENT = "independent"  # equivalent to CF DIMENSION Coordinate


class Coordinate(AggMixin):
    __slots__ = ("_variable", "_axis", "_bounds")

    _LOG = HCubeLogger(name="Coordinate")

    def __init__(
        self,
        variable: Union[np.ndarray, da.Array, Variable],
        axis: Axis,
        bounds: Optional[Union[np.ndarray, da.Array, Variable]] = None,
        mapping: Optional[Mapping[str, str]] = None,
    ):
        self._bounds = None
        if mapping is None:
            mapping = {}
        if isinstance(variable, Variable):
            self._variable = variable
            if self._variable.name in mapping:
                self._variable.properties.update(
                    util_methods.trim_key(mapping[self._variable.name], exclude="api")
                )
                self._variable._name = mapping[self._variable.name]["api"]
        elif isinstance(variable, np.ndarray) or isinstance(variable, da.Array):
            # in this case, we assume an independent variable - passing only a numpy array
            # as name the axis name will be used
            # example:
            #    latitude = Coordinate([70, 68, 66, 64, 62, 60], Axis('latitude', AxisType.LATITUDE))
            #
            props = {}
            name = axis.name
            if name in mapping:
                name = mapping[name]["api"]
                props = util_methods.trim_key(
                    mapping[self._variable.name], exclude="api"
                )
            name = mapping.get(axis.name, axis.name)
            self._variable = Variable(
                name=name,
                data=variable,
                dims=Dimension(name=axis.name, axis=axis),
                units=axis.default_units,
                properties=props,
            )
        else:
            raise ex.HCubeTypeError(
                f"Expected types `numpy.ndarray`, `dask.Array`, or `geokube.Variable` but provided `{type(variable)}`",
                logger=self._LOG,
            )
        if not isinstance(axis, Axis):
            raise ex.HCubeTypeError(
                f"Expected types `geokube.Axis` but provided `{type(axis)}`",
                logger=self._LOG,
            )
        self._axis = axis
        if bounds is not None:
            if isinstance(bounds, Variable):
                self._bounds = bounds
            else:
                # in this case when only a numpy array is passed
                # we assume 2-D numpy array with shape(coord.dim, 2)
                #
                if bounds.ndim != 2 or bounds.shape != (self._variable.shape[0], 2):
                    raise ex.HCubeValueError(
                        f"Expected shape for bounds is `({len(self._variable)},2)` but provided shape is `{bounds.shape}`",
                        logger=self._LOG,
                    )
                self._bounds = Variable(
                    name=f"{self._variable.name}_bounds",
                    data=bounds,
                    units=self._variable.units,
                    dims=[self.axis, Dimension("bounds", Axis(AxisType.GENERIC))],
                )

    @property
    def variable(self):
        return self._variable

    @property
    def name(self) -> str:
        return self._variable.name

    @property
    def dims_names(self):
        return self._variable.dims_names

    @property
    def dims_axis_names(self):
        return self._variable.dims_axis_names

    @property
    def dims(self):
        return self._variable.dims

    @property
    def axis(self):
        return self._axis

    @property
    def units(self):
        return (
            self._variable.units
            if self._variable.units is not None
            else self.axis.atype.default_units
        )

    @property
    def data(self):
        return self._variable.data

    @property
    def values(self):
        return self._variable.values

    @property
    def properties(self):
        return self._variable.properties

    @property
    def cf_encoding(self):
        return self._variable.cf_encoding

    @property
    def ctype(self):
        # TODO: verify logic!
        if self.dims is None or self.dims == ():
            if self.data.ndim == 0:  # it doesn't depend on any dim
                return CoordinateType.SCALAR
        else:
            if (len(self.dims) == 1) and (self.dims[0].axis == self.axis):
                return CoordinateType.INDEPENDENT
            if (len(self.dims) > 1) or (self.dims[0].axis == self.axis):
                return CoordinateType.DEPENDENT
        raise ex.HCubeValueError("Couldn't infer coordinate type!", logger=self._LOG)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    @property
    def has_bounds(self) -> bool:
        return self._bounds is not None

    @property
    # TODO:  check! I think this works only if lat/lon are independent!
    def convention(self) -> Optional[Union[LatitudeConvention, LongitudeConvention]]:
        if self._axis.atype is AxisType.LATITUDE:
            return (
                LatitudeConvention.POSITIVE_TOP
                if self.first() > self.last()
                else LatitudeConvention.NEGATIVE_TOP
            )
        if self._axis.atype is AxisType.LONGITUDE:
            return (
                LongitudeConvention.POSITIVE_WEST
                if self.min() >= 0
                else LongitudeConvention.NEGATIVE_WEST
            )

    @classmethod
    @log_func_debug
    def from_xarray_dataarray(
        cls, da: xr.DataArray, mapping: Optional[Mapping[str, str]] = None
    ) -> "Coordinate":
        if not isinstance(da, xr.DataArray):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.DataArray` but provided `{type(da)}`",
                logger=cls._LOG,
            )
        if (bounds := da.encoding.get("bounds")) is not None:
            warnings.warn(
                f"Bound(s) `{bounds}`` defined for the provided xarray.DataArray but couldn't be found! Bound(s) will be skipped!"
            )

        axis = Axis.from_xarray_dataarray(da, mapping=mapping)
        var = Variable.from_xarray_dataarray(da, mapping=mapping)
        return Coordinate(variable=var, axis=axis, bounds=None, mapping=mapping)

    @classmethod
    @log_func_debug
    def from_xarray_dataset(
        cls,
        ds: xr.Dataset,
        coord_name: str,
        errors: Optional[str] = "raise",
        mapping: Optional[Mapping[str, str]] = None,
    ) -> "Coordinate":
        if not isinstance(ds, xr.Dataset):
            raise ex.HCubeTypeError(
                f"Expected type `xarray.Dataset` but provided `{type(ds)}`",
                logger=cls._LOG,
            )

        try:
            darr = ds[coord_name]
        except KeyError:
            # Checking `coord_name not in ds` doesn't work for integer values, like x = {0,1,2,3,....}(for example: nemo-ocean-16)
            if errors == "raise":
                raise ex.HCubeKeyError(
                    f"Requested coordinate `{coord_name}` does not exists in the provided dataset!",
                    logger=cls._LOG,
                )
            elif errors == "warn":
                warnings.warn(
                    f"Requested coordinate `{coord_name}` does not exists in the provided dataset!"
                )
            return

        axis = Axis.from_xarray_dataarray(darr, mapping=mapping)
        var = Variable.from_xarray_dataarray(darr, mapping=mapping)
        bounds = darr.encoding.get("bounds", darr.attrs.get("bounds"))
        if bounds:
            bounds = Variable.from_xarray_dataarray(ds[bounds])
            bounds._units = var.units
        return Coordinate(variable=var, axis=axis, bounds=bounds, mapping=mapping)

    @log_func_debug
    def to_xarray_dataarray(self):
        xr_var = self._variable.to_xarray_dataarray()

        coords = self.to_dict_of_tuple()

        if self.has_bounds:
            # TODO: maybe we can return time_bounds as DataArray and it wouold have associated time dimension? or just raise a warning
            warnings.warn(
                "Coordinate contains bounds but they cannot be stored with time as xarray.DataArray. Bounds will be skipped!"
            )
            coords.pop(self.bounds.name, None)
        xr_var = xr_var.assign_coords(coords)
        return xr_var

    @log_func_debug
    def to_xarray_dataset(self):
        coords = self.to_dict_of_tuple()
        if self.has_bounds:
            xr_bounds = self.bounds.to_xarray_dataset()
            xr_bounds = xr_bounds.assign_coords(coords)
            xr_bounds[self.name].encoding["bounds"] = self.bounds.name
            return xr_bounds
        xr_var = self._variable.to_xarray_dataarray()
        xr_var = xr_var.assign_coords(coords)
        return xr_var

    def to_dict_of_tuple(self) -> Mapping[Hashable, Tuple[Any]]:
        # return a mapping of coordinates dims_names, data and attributes
        coords = {}
        coords[self.name] = self.variable.to_tuple(return_variable=False)
        if self.has_bounds:
            coords[self.bounds.name] = self.bounds.to_tuple(return_variable=False)
        return coords
