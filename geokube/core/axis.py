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

    __slots__ = ('__default_units', '__dtype')

    def __new__(
        cls, name: str, default_units: pint.Unit | str, dtype: npt.DTypeLike
    ):
        return str.__new__(cls, name)

    def __init__(
        self, name: str, default_units: pint.Unit | str, dtype: npt.DTypeLike
    ) -> None:
        self.__dtype = np.dtype(dtype)
        self.__default_units = pint.Unit(default_units)

    def __repr__(self) -> str:
        name = f"name='{self}'"
        default_units = f"default_units='{self.__default_units}'"
        dtype = f"dtype='{self.__dtype}'"
        return f"{self.__class__.__name__}({name}, {default_units}, {dtype})"

    @property
    def name(self) -> str:
        return str(self)

    @property
    def default_units(self) -> pint.Unit:
        return self.__default_units

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype


class Spatial(Axis):
    pass


class Horizontal(Spatial):
    pass


class Elevation(Spatial):
    pass


class Time(Axis):
    pass


class UserDefined(Axis):
    pass


x = Horizontal('x', units['meter'], np.float64)
y = Horizontal('y', units['meter'], np.float64)
z = Elevation('z', units['meter'], np.float64)
grid_latitude = Horizontal('grid_latitude', units['degrees'], np.float64)
grid_longitude = Horizontal('grid_longitude', units['degrees'], np.float64)
latitude = Horizontal('latitude', units['degrees_north'], np.float64)
longitude = Horizontal('longitude', units['degrees_east'], np.float64)
vertical = Elevation('vertical', units['meter'], np.float64)
time = Time('time', units[''], 'datetime64')
timedelta = Time('timedelta', units[''], 'timedelta64')


def create(
    name: str, default_units: pint.Unit | str, dtype: npt.DTypeLike
) -> None:
    # TODO: Implement this.
    pass
