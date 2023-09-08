from collections.abc import Mapping, Sequence

import pint

from . import axis, crs


class CoordinateSystem:
    __slots__ = (
        '__horizontal_crs', '__elevation', '__time', '__user_axes', '__units'
    )

    _ELEVATION_AXES = frozenset({axis.vertical, axis.z})
    # TODO: Add reference time and climate time axes.
    _TIME_AXES = frozenset({axis.time, axis.timedelta})

    def __init__(
        self,
        horizontal_crs: crs.CRS | None = None,
        elevation: Sequence[axis.Axis] = (),
        time: Sequence[axis.Axis] = (),
        user_axes: Sequence[axis.Axis] = (),
        units: Mapping[axis.Axis, pint.Unit] | None = None
    ) -> None:
        # TODO: Solve the case with `None`.
        self.horizontal_crs = horizontal_crs

        if diff := set(elevation) - self._ELEVATION_AXES:
            raise ValueError(
                f"'elevation' contains axes that are not allowed: "
                f"{sorted({diff})}"
            )
        self.__elevation = tuple(elevation)

        if diff := set(time) - self._TIME_AXES:
            raise ValueError(
                f"'time' contains axes that are not allowed: {sorted({diff})}"
            )
        self.__time = tuple(time)

        # TODO: Reconsider the logic of checking the user axes.
        hor_axes = (
            set() if horizontal_crs is None else set(horizontal_crs.AXES)
        )
        predef_axes = hor_axes.union(self._ELEVATION_AXES, self._TIME_AXES)
        if intersect := set(user_axes) & predef_axes:
            raise ValueError(
                "'user_axes' contains axes that are not allowed: "
                f"{sorted({intersect})}"
            )
        self.__user_axes = tuple(user_axes)

        self.__units = axis._DEFAULT_UNITS.copy()
        if units:
            self.__units |= units

    @property
    def horizontal_crs(self) -> crs.CRS | None:
        return self.__horizontal_crs

    @horizontal_crs.setter
    def horizontal_crs(self, value: crs.CRS | None) -> None:
        if not (value is None or isinstance(value, crs.CRS)):
            raise TypeError("'horizontal_crs' must be an instance of 'CRS'")
        self.__horizontal_crs = value

    @property
    def elevation(self) -> tuple[axis.Axis, ...]:
        return self.__elevation

    @property
    def time(self) -> tuple[axis.Axis, ...]:
        return self.__time

    @property
    def user_axes(self) -> tuple[axis.Axis, ...]:
        return self.__user_axes

    @property
    def axes(self) -> tuple[axis.Axis, ...]:
        axes_ = (*self.__user_axes, *self.__time, *self.__elevation)
        if self.__horizontal_crs is not None:
            axes_ += self.__horizontal_crs.AXES
        return axes_

    @property
    def units(self) -> dict[axis.Axis, pint.Unit]:
        return self.__units

    def add_axis(self, new_axis: axis.Axis) -> None:
        if not isinstance(new_axis, axis.Axis):
            raise TypeError("'new_axis' must be an instance of 'Axis'")
        hor_axes = (
            set()
            if self.__horizontal_crs is None else
            set(self.__horizontal_crs.AXES)
        )
        all_axes = hor_axes.union(set(self.__elevation), set(self.__time))
        if new_axis in all_axes:
            raise ValueError(f"'new_axis' {new_axis} already exists")

        if new_axis in self._ELEVATION_AXES:
            self.__elevation += (new_axis,)
        elif new_axis in self._TIME_AXES:
            self.__time += (new_axis,)
        else:
            self.__user_axes += (new_axis,)

    def _delete_axis_from(
        self,
        axes: tuple[axis.Axis, ...],
        existing_axis: axis.Axis
    ) -> tuple[axis.Axis, ...]:
        tmp = list(axes)
        tmp.remove(existing_axis)
        return tuple(tmp)

    def delete_axis(self, existing_axis: axis.Axis) -> None:
        if (
            self.__horizontal_crs is not None
            and existing_axis in set(self.__horizontal_crs.AXES)
        ):
            raise ValueError("'existing_axis' cannot be removed from CRS")
        if existing_axis in set(self.__elevation):
            self.__elevation = self._delete_axis_from(
                self.__elevation, existing_axis
            )
        elif existing_axis in set(self.__time):
            self.__time = self._delete_axis_from(self.__time, existing_axis)
        elif existing_axis in set(self.__user_axes):
            self.__user_axes = self._delete_axis_from(
                self.__user_axes, existing_axis
            )
        else:
            raise ValueError(f"'existing_axis' {existing_axis} not found")
