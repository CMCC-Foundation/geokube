from typing import Union


from ..utils.decorators import geokube_logging
from .axis import Axis, AxisType
from .coordinate import Coordinate, CoordinateType
from .enums import LatitudeConvention, LongitudeConvention


class DomainMixin:
    def __init__():
        pass

    @property
    def latitude(self):
        return self[AxisType.LATITUDE]

    @property
    def longitude(self):
        return self[AxisType.LONGITUDE]

    @property
    def vertical(self):
        return self[AxisType.VERTICAL]

    @property
    def time(self):
        return self[AxisType.TIME]

    @property
    def x(self):
        return self[AxisType.X]

    @property
    def y(self):
        return self[AxisType.Y]

    @property
    def longitude_convention(self) -> LongitudeConvention:
        if AxisType.LONGITUDE in self._Axis_to_name:
            return self[AxisType.LONGITUDE].convention

    @property
    def latitude_convention(self) -> LatitudeConvention:
        if AxisType.LATITUDE in self._axis_to_name:
            return self[AxisType.LATITUDE].convention

    @property
    def is_latitude_independent(self):
        return self[AxisType.LATITUDE].type is CoordinateType.INDEPENDENT

    @property
    def is_longitude_independent(self):
        return self[AxisType.LONGITUDE].type is CoordinateType.INDEPENDENT

    @geokube_logging
    def __getitem__(self, key: Union[AxisType, Axis, str]) -> Coordinate:
        if isinstance(key, str):
            return self.coords[key]
        elif isinstance(key, AxisType):
            if key not in self._axis_to_name:
                raise KeyError(
                    f"Axis of type `{key}` does not exist in the domain!",
                )
            return self.coords[self._axis_to_name.get(key)]
        elif isinstance(key, Axis):
            if key.type not in self._axis_to_name:
                raise KeyError(
                    f"Axis of type `{key}` does not exist in the domain!",
                )
            return self.coords[self._axis_to_name.get(key.type)]
        raise TypeError(
            f"Indexing coordinates for Domain is supported only for object of types [str, geokube.Axis, geokub.AxisType]. Provided type: {type(key)}",
        )
