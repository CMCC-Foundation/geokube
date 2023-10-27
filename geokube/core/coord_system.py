from collections.abc import Mapping, Sequence
from typing import Self, TYPE_CHECKING

import pint

from . import axis
from .crs import CRS


class SpatialCoordinateSystem:
    __slots__ = ('__crs', '__elevation', '__dim_axes', '__axes')

    if TYPE_CHECKING:
        __crs: CRS
        __elevation: axis.Elevation | None
        __dim_axes: tuple[axis.Spatial, ...]
        __axes: tuple[axis.Spatial, ...]

    def __init__(
        self, crs: CRS, elevation: axis.Elevation | None = None
    ) -> None:
        if not isinstance(crs, CRS):
            raise TypeError("'crs' must be an instance of 'CRS'")
        self.__crs = crs
        dim_axes: list[axis.Spatial] = list(crs.dim_axes)
        match elevation:
            case axis.Elevation():
                self.__elevation = elevation
                dim_axes.insert(0, elevation)
            case None:
                self.__elevation = None
            case _:
                raise TypeError(
                    "'elevation' must be an instance of 'Elevation' or 'None'"
                )
        self.__dim_axes = tuple(dim_axes)
        self.__axes = self.__dim_axes + self.aux_axes

    @property
    def crs(self) -> CRS:
        return self.__crs

    @property
    def elevation(self) -> axis.Elevation | None:
        return self.__elevation

    @property
    def dim_axes(self) -> tuple[axis.Spatial, ...]:
        return self.__dim_axes

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.__crs.aux_axes

    @property
    def axes(self) -> tuple[axis.Spatial, ...]:
        return self.__axes


class CoordinateSystem:
    __slots__ = (
        '__spatial',
        '__time',
        '__user_axes',
        '__all_axes',
        '__dim_axes',
        '__units'
    )

    if TYPE_CHECKING:
        __spatial: SpatialCoordinateSystem
        __time: axis.Time | None
        __user_axes: tuple[axis.UserDefined, ...]
        __all_axes: tuple[axis.Axis, ...]
        __dim_axes: tuple[axis.Axis, ...]
        __units: dict[axis.Axis, pint.Unit]

    def __init__(
        self,
        horizontal: CRS,
        elevation: axis.Elevation | None = None,
        time: axis.Time | None = None,
        user_axes: Sequence[axis.UserDefined] = (),
        units: Mapping[axis.Axis, pint.Unit] | None = None
    ) -> None:
        self.__spatial = SpatialCoordinateSystem(horizontal, elevation)
        axes: tuple[axis.Axis, ...]

        match time:
            case axis.Time():
                self.__time = time
                axes = (time,)
            case None:
                self.__time = None
                axes = ()
            case _:
                raise TypeError(
                    "'time' must be an instance of 'Time' or 'None'"
                )

        if user_axes:
            for user_axis in user_axes:
                if not isinstance(user_axis, axis.UserDefined):
                    raise TypeError(
                        "'user_axis' must be an instance of 'UserDefined'"
                    )
            self.__user_axes = tuple(user_axes)
            axes = self.__user_axes + axes
        else:
            self.__user_axes = ()

        self.__dim_axes = axes + self.__spatial.dim_axes
        self.__all_axes = axes + self.__spatial.axes

        self.__units = {axis: axis.units for axis in self.__all_axes}
        if units:
            self.__units |= units

    @property
    def spatial(self) -> SpatialCoordinateSystem:
        return self.__spatial

    @property
    def time(self) -> axis.Time | None:
        return self.__time

    @property
    def user_axes(self) -> tuple[axis.UserDefined, ...]:
        return self.__user_axes

    @property
    def axes(self) -> tuple[axis.Axis, ...]:
        return self.__all_axes

    @property
    def dim_axes(self) -> tuple[axis.Axis, ...]:
        return self.__dim_axes

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.spatial.crs.aux_axes

    @property
    def units(self) -> dict[axis.Axis, pint.Unit]:
        return self.__units

    def add_axis(self, new_axis: axis.UserDefined) -> Self:
        return type(self)(
            horizontal=self.__spatial.crs,
            elevation=self.__spatial.elevation,
            time=self.__time,
            user_axes=(*self.__user_axes, new_axis),
            units=self.__units.copy()
        )

    def delete_axis(self, existing_axis: axis.UserDefined) -> Self:
        new_axes = list(self.__user_axes)
        new_axes.remove(existing_axis)
        return type(self)(
            horizontal=self.__spatial.crs,
            elevation=self.__spatial.elevation,
            time=self.__time,
            user_axes=tuple(new_axes),
            units=self.__units.copy()
        )
