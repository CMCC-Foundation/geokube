from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from enum import Enum, unique
import re
from typing import Self
import sys
from warnings import warn

import numpy as np
import numpy.typing as npt
import pint


# TODO: This should be used in the configuration file:
#     [mypy]
#     plugins = numpy.typing.mypy_plugin
# See: https://numpy.org/devdocs/reference/typing.html.


@dataclass(frozen=True, slots=True)
class _AxisIndexer(Iterable):
    axis: Axis
    indexer: slice | npt.ArrayLike | pint.Quantity

    # TODO: Consider removing `__iter__` and inheritance from `Iterable`. It is
    # not necessary but sometimes can be suitable for unpacking or converting
    # to `dict`.

    def __iter__(
        self
    ) -> Iterator[Axis | slice | npt.ArrayLike | pint.Quantity]:
        return (getattr(self, field.name) for field in fields(self))


class Axis(str):
    # HACK: `Axis` inherits `str` to be capable of acting as a dimension in
    # `xarray` data structures. If `xarray` becomes capable of handling all
    # hashables as dimensions, the inheritance from `str` will not be
    # necessary any more. In that case, `Axis` might inherit from `Hashable`
    # and `Iterable` if `__iter__` method is kept.

    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self}')"

    # TODO: Consider whether the desired behavior or (in)equality comparison
    # between instances of `Axis` and `str`. For example, it is necessary to
    # decide how `Axis('time') == 'time'` behaves. If the implementation of
    # the methods `__eq__` and `__ne__` is skipped, then
    # `Axis('time') == 'time'` returns `True` and `Axis('time') != 'time'`
    # returns `False`.

    # def __eq__(self, other) -> bool:
    #     if other.__class__ is self.__class__:
    #         return str(self) == str(other)
    #     # return NotImplemented
    #     return False

    # def __ne__(self, other) -> bool:
    #     # return not self == other
    #     if other.__class__ is self.__class__:
    #         return str(self) != str(other)
    #     return True

    # def __hash__(self) -> int:
    #     return hash(str(self))

    # TODO: Consider removing `__iter__` and inheritance from `Iterable`. It is
    # not necessary but sometimes can be suitable for unpacking or converting
    # to `dict`.

    def __iter__(self) -> Iterator[Self]:
        return iter((self,))

    def __getitem__(
        self, key: slice | npt.ArrayLike | pint.Quantity
    ) -> _AxisIndexer:
        match key:
            case slice():
                return _AxisIndexer(self, key)
            case pint.Quantity():
                return _AxisIndexer(
                    self, pint.Quantity(np.array(key.magnitude), key.units)
                )
            case _:
                return _AxisIndexer(self, np.array(key))


@unique
class _AxisKind(Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    GRID_LATITUDE = 'grid_latitude'
    GRID_LONGITUDE = 'grid_longitude'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'
    VERTICAL = 'vertical'
    TIME = 'time'
    TIMEDELTA = 'timedelta'


def create(name: str) -> None:
    module = sys.modules[__name__]
    names, values = [], []
    for axis_ in _AxisKind:
        names.append(axis_.name)
        values.append(axis_.value)
    if name.lower() in {value.lower() for value in values}:
        warn(f"'name' '{name}' not added because it already exists")
    else:
        names.append(name.upper())
        values.append(name)
        items = list(zip(names, values))
        setattr(module, '_AxisKind', Enum('_AxisKind', items))
        setattr(module, name, Axis(name))


# Dynamically adding predefined axes like: time, time delta, latitude,
# longitude, vertical, x, y, and points.
# NOTE: Linters cannot recognize that `axis` has these attributes.
# def _predefine_axes():
#     module = sys.modules[__name__]
#     for axis_ in _AxisKind:
#         axis_name = axis_.value
#         setattr(module, axis_name, Axis(axis_name))


# _predefine_axes()


# Statically adding predefined axes like: time, time delta, latitude,
# longitude, vertical, x, y, and points.
# NOTE: This is currently the preferred way to its dynamic alternative because
# of linting capabilities.
x = Axis('x')
y = Axis('y')
z = Axis('z')
grid_latitude = Axis('grid_latitude')
grid_longitude = Axis('grid_longitude')
latitude = Axis('latitude')
longitude = Axis('longitude')
vertical = Axis('vertical')
time = Axis('time')
timedelta = Axis('timedelta')


_PREDEFINED_AXIS_ENCODING = {
    x: {'axis': 'X', 'pattern': r"(x|projection_x*|rlon|grid_lon.*)"},
    y: {'axis': 'Y', 'pattern': r"(y|projection_y*|rlat|grid_lat.*)"},
    z: {'axis': 'Z', 'pattern': r"z"},
    grid_latitude: {
        # TODO: Check these attributes.
        'axis': 'Y', 'standard_name': 'grid_latitude', 'pattern': "grid_lat.*"
    },
    grid_longitude: {
        # TODO: Check these attributes.
        'axis': 'X', 'standard_name': 'grid_longitude', 'pattern': "grid_lon.*"
    },
    latitude: {
        'standard_name': 'latitude', 'pattern': r"(x?lat[a-z0-9]*|nav_lat)"
    },
    longitude: {
        'standard_name': 'longitude', 'pattern': r"(x?lon[a-z0-9]*|nav_lon)"
    },
    vertical: {
        'pattern': (
            r"(soil|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric"
            r"|pres|vertical|isotherm|model_level_number)[a-z_]*[0-9]*"
        )
    },
    timedelta: {'pattern': r"timedelta|time_delta"},
    time: {'standard_name': 'time', 'pattern': r"(time[0-9]*|T)"}
}

# TODO: Finish this once the units are managed properly.
_DEFAULT_UNITS = {
    x: pint.Unit(''),
    y: pint.Unit(''),
    z: pint.Unit(''),
    grid_latitude: pint.Unit('degree'),
    grid_longitude: pint.Unit('degree'),
    # latitude: units.degrees_north,
    # longitude: units.degrees_east,
    time: pint.Unit('')
}


# TODO: Consider making these methods internal.
def match_cfaxis(axis: str) -> Axis:
    for axis_, encoding in _PREDEFINED_AXIS_ENCODING.items():
        if (name := encoding.get('axis')) and name == axis:
            return axis_
    raise ValueError(f"'axis' '{axis}' do not match any of the existing axes")


def match_cfstdname(std_name: str) -> Axis:
    for axis_, encoding in _PREDEFINED_AXIS_ENCODING.items():
        if (name := encoding.get('standard_name')) and name == std_name:
            return axis_
    raise ValueError(
        f"'std_name' '{std_name}' do not match any of the existing axes"
    )


def match_pattern(name: str) -> Axis:
    for axis_, encoding in _PREDEFINED_AXIS_ENCODING.items():
        if pattern_ := encoding.get('pattern'):
            pattern = re.compile(pattern=pattern_, flags=re.IGNORECASE)
            if re.match(pattern=pattern, string=name) is not None:
                return axis_
    raise ValueError(
        f"'name' '{name}' do not match any of the existing patterns"
    )


def match(name: str) -> Axis:
    for match_func in (match_cfstdname, match_cfaxis, match_pattern):
        try:
            return match_func(name)
        except ValueError:
            pass
    raise ValueError(f"'name' '{name}' do not match any of the existing axes")
