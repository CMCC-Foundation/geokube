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
    _DEFAULT_ENCODING_: { 'dtype': np.dtype }

    __slots__ = ('__units', '__encoding')

    def __new__(
        cls,
        units: pint.Unit | str | None = None,
        encoding: Mapping[str, Any] | None = None
    ):
        return str.__new__(cls, cls.__name__)

    def __init__(
        self,
        units: pint.Unit | str | None = None,
        encoding: Mapping[str, Any] | None = None
    ) -> None:
        self.__units = pint.Unit(units or self._DEFAULT_UNITS_)
        self.__encoding = self._DEFAULT_ENCODING_ if encoding is None else dict(encoding)

    def __repr__(self) -> str:
        cls_ = self.__class__.__name__
        units = f"units='{self.__units}'"
        dtype = f"dtype='{self.__encoding['dtype']}'"
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
        return np.dtype(self.__encoding['dtype'])

    @property
    def encoding(self) -> dict[str, Any]:
        return self.__encoding


class Spatial(Axis):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_DTYPE_ = 'float64'


class Horizontal(Spatial):
    pass


class Longitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees_east']
    _DEFAULT_ENCODING_ = { 'standard_name': 'longitude',
                           'dtype': 'float64'}

class Latitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees_north']
    _DEFAULT_ENCODING_ = { 'standard_name': 'latitude',
                           'dtype': 'float64'}

class GridLongitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees']
    _DEFAULT_ENCODING_ = { 'standard_name': 'grid_longitude',
                           'dtype': 'float64'}

class GridLatitude(Horizontal):
    _DEFAULT_UNITS_ = units['degrees']
    _DEFAULT_ENCODING_ = { 'standard_name': 'grid_latitude',
                           'dtype': 'float64'}

class X(Horizontal):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'axis': 'X',
                           'dtype': 'float64'}


class Y(Horizontal):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'axis': 'Y',
                           'dtype': 'float64'}

class Elevation(Spatial):
    _DEFAULT_UNITS_ = units['meter']

class Z(Elevation):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'axis': 'Z',
                           'dtype': 'float64'}

class Vertical(Elevation):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'standard_name': 'height',
                           'positive': 'up',
                           'dtype': 'float64'}

class Height(Elevation):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'standard_name': 'height',
                           'positive': 'up',
                           'dtype': 'float64'}

class Depth(Elevation):
    _DEFAULT_UNITS_ = units['meter']
    _DEFAULT_ENCODING_ = { 'standard_name': 'depth',
                           'positive': 'down',
                           'dtype': 'float64'}


class Time(Axis):
    _DEFAULT_UNITS_ = units['']
    _DEFAULT_ENCODING_ = { 'standard_name': 'time',
                           'dtype': 'datetime64'}


class UserDefined(Axis): # Hash cannot be the class name -> redefine hash with axis name
    _DEFAULT_UNITS_ = units['']
    _DEFAULT_DTYPE_ = 'str'


x = X()
y = Y()
z = Z()
grid_latitude = GridLatitude()
grid_longitude = GridLongitude()
latitude = Latitude()
longitude = Longitude()
vertical = Vertical()
time = Time()
timedelta = Time(encoding={'dtype': 'timedelta64'})

def create(
    name: str, units: pint.Unit | str, encoding: dict
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
