from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pint

from .units import units


# pylint: disable=unused-argument


class Axis(str):
    # HACK: `Axis` inherits `str` to be capable of acting as a dimension in
    # `xarray` data structures. If `xarray` becomes capable of handling all
    # hashables as dimensions, the inheritance from `str` will not be
    # necessary any more. In that case, `Axis` might inherit from `Hashable`.

    _DEFAULT_UNITS_: pint.Unit
    _DEFAULT_DTYPE_: np.dtype

    __slots__ = ('_units', '_dtype', '_encoding')

    def __new__(
        cls,
        units: pint.Unit | str | None = None,
        dtype: npt.DTypeLike | None = None,
        encoding: Mapping[str, Any] | None = None
    ):
        return str.__new__(cls, cls.__name__)

    def __init__(
        self,
        units: pint.Unit | str | None = None,
        dtype: npt.DTypeLike | None = None,
        encoding: Mapping[str, Any] | None = None
    ) -> None:
        self.__dtype = np.dtype(dtype) or self._DEFAULT_DTYPE_
        self.__units = pint.Unit(units or self._DEFAULT_UNITS_)
        self.__encoding = {} if encoding is None else dict(encoding)

    def __repr__(self) -> str:
        cls_ = self.__class__.__name__
        units = f"units='{self.__units}'"
        dtype = f"dtype='{self.__dtype}'"
        encoding = f"encoding={self.__encoding}"
        return f"{cls_}({units}, {dtype}, {encoding})"

    @property
    def name(self) -> str:
        return str(self)

    @property
    def units(self) -> pint.Unit:
        return self.__units

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    @property
    def encoding(self) -> dict[str, Any]:
        return self.__encoding


class Spatial(Axis):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_DTYPE_ = np.dtype('float64')


class Horizontal(Spatial):
    pass


class Longitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees_north']


class Latitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees_east']


class GridLongitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees']


class GridLatitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees']


class X(Horizontal):
    _DEFAULT_UNITS_ = units['meter']


class Y(Horizontal):
    _DEFAULT_UNITS_ = units['meter']


class Elevation(Spatial):
    _DEFAULT_UNITS_ = units['meter']


class Vertical(Elevation):
    pass


class Height(Elevation):
    pass


class Depth(Elevation):
    pass


class Z(Elevation):
    pass


class Time(Axis):
    _DEFAULT_UNITS_ = units['']
    _DEFAULT_DTYPE_ = np.dtype('datetime64')


class UserDefined(Axis): # Hash cannot be the class name -> redefine hash with axis name
    _DEFAULT_UNITS_ = units['']
    _DEFAULT_DTYPE_ = np.dtype('float64')


x = X()
y = Y()
z = Z()
grid_latitude = GridLatitude()
grid_longitude = GridLongitude()
latitude = Latitude()
longitude = Longitude()
vertical = Vertical()
time = Time(dtype=np.datetime64)
timedelta = Time(dtype=np.timedelta64)


def create(
    name: str, units: pint.Unit | str, dtype: npt.DTypeLike
) -> None:
    # TODO: Implement this.
    pass

__predefined_axis__ = {
    'x': x,
    'y': y,
    'z': z,
    'grid_latitude': grid_latitude,
    'grid_longitude': grid_longitude,
    'latitude': latitude,
    'longitude': longitude,
    'vertical': vertical,
    'time': time,
    'timedelta': timedelta
}

def _from_string(name: str):
    if (name in __predefined_axis__):
        return __predefined_axis__[name]
