from enum import Enum
from typing import Any, Hashable, Iterable, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
from geokube.utils import util_methods
import xarray as xr

import geokube.utils.exceptions as ex
from geokube.utils.decorators import log_func_debug
from geokube.utils.hcube_logger import HCubeLogger

from .axis import Axis, AxisType
from .dimension import Dimension
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

class Coordinate(Variable): 
    __slots__ = ("_axis", "_bounds")

    _LOG = HCubeLogger(name="Coordinate")

    def __init__(
        self,
        data: Union[np.ndarray, da.Array, Variable],
        axis: Optional[Union[str, Axis, Dimension]],
        dims: Optional[Tuple[Dimension]] = None,
        units: Optional[Union[Unit, str]] = None,
        bounds: Optional[Union[np.ndarray, da.Array, Variable]] = None,
        properties: Optional[Mapping[Any, Any]] = None,
        encoding: Optional[Mapping[Any, Any]] = None
    ):
        if data is not None:
            super().__init__(data=data, dims=dims, units=units,
                             properties=properties, encoding=encoding)
        else:
            raise ex.HCubeTypeError(
                f"Data is not provided",
                logger=self._LOG,
            )

        if isinstance(axis, Axis) or isinstance(axis, Dimension):
            self._axis = axis        
        elif isinstance(axis, str):
            self._axis = Axis(axis)
        else:
            raise ex.HCubeTypeError(
                f"Expected argument of one of the following types `Axis`, `Dimension`, or `String` but provided {type(data)}",
                logger=self._LOG,
            )

        if bounds is not None:
            if isinstance(bounds, dict):
                # we need to check each element of dict; in case build a variable
                self._bounds = bounds
            if isinstance(bounds, Variable):
                self._bounds = { f"{self.name}_bounds": bounds }
            else:
                # in this case when only a numpy array is passed
                # we assume 2-D numpy array with shape(coord.dim, 2)
                #
                if bounds.ndim != 2 or bounds.shape != (self.shape[0], 2):
                    raise ex.HCubeValueError(
                        f"Expected shape for bounds is `({len(self._variable)},2)` but provided shape is `{bounds.shape}`",
                        logger=self._LOG,
                    )
                
                self._bounds = { 
                    f"{self.name}_bounds":
                        Variable(data=bounds,
                                 units=self.units,
                                 dims=tuple(self.axis, Dimension("bounds", AxisType.GENERIC)),
                                )
                }

    @property
    def axis(self) -> str:
        return self._axis

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def ncvar(self) -> str:
        return self.axis.encoding.get('ncvar', self.name)

    @property
    def is_dimension(self) -> str:
        return isinstance(self.axis, Dimension)

    @property
    def is_independent(self) -> str:
        return self.is_dimension()

    @property
    def is_dependent(self) -> str:
        return not self.is_dimension()

    @property
    def type(self):
        if self.dims is None or self.dims == ():
            return CoordinateType.SCALAR
        else:
            if self.is_dimension():
                return CoordinateType.DEPENDENT
            else:
                return CoordinateType.INDEPENDENT

    @property
    def bounds(self):
        if self._bounds is not None:
            return self._bounds
        else:
            return {}

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

        print(f"coordinate {ncvar}")

        da = ds[ncvar]
        
        var = Variable.from_xarray(da, id_pattern=id_pattern, mapping=mapping)
        
        name = Variable._get_name(ds[ncvar], mapping, id_pattern)
        var.encoding.update(name=ncvar)

        axis = Axis(da.attrs.get("axis", Variable._get_name(da, mapping, id_pattern)))
        if ncvar in da.dims:
            axis = Dimension(name=axis.name, axistype = axis.type, encoding={'name': ncvar})
        bnds_ncvar = da.encoding.pop("bounds", da.attrs.pop("bounds", None))
        if bnds_ncvar:
            bnds_name = Variable._get_name(ds[bnds_ncvar], mapping, id_pattern)
            bounds = {
                bnds_name: Variable.from_xarray(ds[bnds_ncvar], id_pattern=id_pattern, copy=copy, mapping=mapping)
            }
            if 'units' not in ds[bnds_ncvar].attrs and 'units' not in ds[bnds_ncvar].encoding:
                bounds[bnds_name].units = var.units
        else:
            bounds = None
        return Coordinate(data=var, axis=axis, bounds=bounds)