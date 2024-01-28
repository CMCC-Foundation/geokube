from collections.abc import Mapping
from typing import Any, TypeVar

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
        return (
            f"{self.__class__.__name__}"
            f"(units='{self.__units}', encoding={self.__encoding})"
        )

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
                           'axis': 'T',
                           'dtype': 'datetime64'}


class UserDefined(Axis):
    # FIXME: Hash cannot be the class name -> redefine hash with axis name.
    _DEFAULT_UNITS_ = units['']
    _DEFAULT_ENCODING_ = { 'dtype': 'str'}

x = X()
y = Y()
z = Z()
grid_latitude = GridLatitude()
grid_longitude = GridLongitude()
latitude = Latitude()
longitude = Longitude()
vertical = Vertical()
height = Height()
depth = Depth()
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
    'height': height,
    'depth': depth,
    'time': time,
    'timedelta': timedelta
}


def _from_string(name: str):
    return __predefined_axis__.get(name)


UserDefinedT_co = TypeVar(
    'UserDefinedT_co', bound=UserDefined, covariant=True
)


def custom(
    type_name: str,
    default_units: str | pint.Unit | None = None,
    default_encoding: Mapping[str, Any] | None = None,
) -> UserDefinedT_co:
    # NOTE: To be used in this manner: `ensemble = custom('Ensemble')`.
    name = str(type_name)
    base = (UserDefined,)
    dict_ = {
        '_DEFAULT_UNITS_': pint.Unit(default_units or ''),
        '_DEFAULT_ENCODING_': dict(default_encoding or UserDefined._DEFAULT_ENCODING_),
    }
    cls_ = type(name, base, dict_)
    obj = cls_()
    return obj
