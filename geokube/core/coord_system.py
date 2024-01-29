"""
Coordinate System
=================

The coordinate system of a domain that contains:

* Horizontal coordinate reference system with horizontal axes.  It can
  be either geodetic, rotated or a projection.
* Elevation axis.
* Temporal axis.
* User-defined axes.
* Units that correspond to the axes.

Classes
-------

SpatialCoordinateSystem
    The spatial part of the coordinate system of a domain.

CoordinateSystem
    The coordinate system of a domain.

"""

from collections.abc import Sequence
from typing import Self, TYPE_CHECKING

import pint

from . import axis
from .crs import CRS


class SpatialCoordinateSystem:
    """
    The spatial part of the coordinate system of a domain.

    Parameters
    ----------
    crs : :class:`.crs.CRS`
        Horizontal coordinate reference system with horizontal axes.  It
        can be either geodetic, rotated or a projection.
    elevation : :class:`.axis.Elevation`, default: None
        Elevation axis.

    Attributes
    ----------
    crs : :class:`.crs.CRS`
        Return the horizontal coordinate reference system.
    elevation : :class:`.axis.Elevation`, default: None
        Return the elevation axis.
    dim_axes : :class:`tuple` of :class:`.axis.Spatial`
        Return the tuple of spatial dimension axes.
    aux_axes : :class:`tuple` of :class:`.axis.Horizontal`
        Return the tuple of horizontal auxiliary axes.
    axes : :class:`tuple` of :class:`.axis.Spatial`
        Return the tuple of spatial dimension and auxiliary axes.

    """

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
        """
        Return the horizontal coordinate reference system.

        Horizontal coordinate reference system with horizontal axes.  It
        can be either geodetic, rotated or a projection.

        """
        return self.__crs

    @property
    def elevation(self) -> axis.Elevation | None:
        """Return the elevation axis."""
        return self.__elevation

    @property
    def dim_axes(self) -> tuple[axis.Spatial, ...]:
        """Return the tuple of spatial dimension axes."""
        return self.__dim_axes

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        """Return the tuple of horizontal auxiliary axes."""
        return self.__crs.aux_axes

    @property
    def axes(self) -> tuple[axis.Spatial, ...]:
        """Return the tuple of spatial dimension and auxiliary axes."""
        return self.__axes


class CoordinateSystem:
    """
    The coordinate system of a domain.

    Parameters
    ----------
    horizontal : :class:`.crs.CRS`
        Horizontal coordinate reference system with horizontal axes.  It
        can be either geodetic, rotated or a projection.
    elevation : :class:`.axis.Elevation`, default: None
        Elevation axis.
    time : :class:`.axis.Time`, default: None
        Time axis.
    user_axes : sequence of :class:`.axis.UserDefined`
        User-defined axes.

    Attributes
    ----------
    spatial : :class:`.SpatialCoordinateSystem`
        Return the spatial part of the coordinate system.
    crs : :class:`.crs.CRS`
        Return the horizontal coordinate reference system.
    elevation : :class:`.axis.Elevation`, default: None
        Return the elevation axis.
    time : :class:`.axis.Time`, default: None
        Return the time axis.
    user_axes : :class:`tuple` of :class:`.axis.UserDefined`
        Return the tuple of user-defined axes.
    dim_axes : :class:`tuple` of :class:`.axis.Spatial`
        Return the tuple of spatial dimension axes.
    aux_axes : :class:`tuple` of :class:`.axis.Horizontal`
        Return the tuple of horizontal auxiliary axes.
    axes : :class:`tuple` of :class:`.axis.Spatial`
        Return the tuple of spatial dimension and auxiliary axes.
    units : :class:`dict` of :class:`pint.Unit` and :class:`.axis.Axis`
        Return the units that correspond to the axes.

    Methods
    -------
    add_axis(new_axis)
        Return a new coordinate system with axis added to user axes.
    delete_axis(existing_axis)
        Return a new coordinate system without the specified axis.

    See Also
    --------
    SpatialCoordinateSystem :
        The spatial part of the coordinate system of a domain.

    Examples
    --------

    Coordinate system with geodetic CRS:

    >>> coord_system = CoordinateSystem(
    ...     horizontal=Geodetic(),
    ...     elevation=axis.vertical,
    ...     time=axis.time
    ... )

    Coordinate system with rotated geodetic CRS:

    >>> rotated_crs = RotatedGeodetic(
    ...     grid_north_pole_latitude=47.0,
    ...     grid_north_pole_longitude=-168.0
    ... )
    >>> coord_system = CoordinateSystem(
    ...     horizontal=rotated_crs,
    ...     elevation=axis.vertical,
    ...     time=axis.time
    ... )

    """

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

    def __init__(
        self,
        horizontal: CRS,
        elevation: axis.Elevation | None = None,
        time: axis.Time | None = None,
        user_axes: Sequence[axis.UserDefined] = ()
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

    @property
    def spatial(self) -> SpatialCoordinateSystem:
        """Return the spatial part of the coordinate system."""
        return self.__spatial

    @property
    def crs(self) -> CRS:
        """
        Return the horizontal coordinate reference system.

        Horizontal coordinate reference system with horizontal axes.  It
        can be either geodetic, rotated or a projection.

        """
        return self.spatial.crs

    @property
    def elevation(self) -> axis.Elevation | None:
        """Return the elevation axis."""
        return self.spatial.elevation

    @property
    def time(self) -> axis.Time | None:
        """Return the time axis."""
        return self.__time

    @property
    def user_axes(self) -> tuple[axis.UserDefined, ...]:
        """Return the tuple of user-defined axes."""
        return self.__user_axes

    @property
    def axes(self) -> tuple[axis.Axis, ...]:
        """Return the tuple of spatial dimension and auxiliary axes."""
        return self.__all_axes

    @property
    def dim_axes(self) -> tuple[axis.Axis, ...]:
        """Return the tuple of spatial dimension axes."""
        return self.__dim_axes

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        """Return the tuple of horizontal auxiliary axes."""
        return self.spatial.crs.aux_axes

    @property
    def units(self) -> dict[axis.Axis, pint.Unit]:
        """Return the units that correspond to the axes."""
        # TODO: Consider providing units in the initializer.
        return {axis: axis.units for axis in self.__all_axes}

    def add_axis(self, new_axis: axis.UserDefined) -> Self:
        """
        Return a new coordinate system with axis added to user axes.

        This method appends `new_axis` to the end of the sequence
        :attr:`.user_axes`.  The other axes and the coordinate reference
        system remain unchanged.

        Parameters
        ----------
        new_axis : :class:`.axis.UserDefined`
            The axis to add.

        Returns
        -------
        CoordinateSystem
            New coordinate system with axis added to the end of the
            sequence of user axes.

        """
        return type(self)(
            horizontal=self.__spatial.crs,
            elevation=self.__spatial.elevation,
            time=self.__time,
            user_axes=(*self.__user_axes, new_axis)
        )

    def delete_axis(self, existing_axis: axis.UserDefined) -> Self:
        """
        Return a new coordinate system without the specified axis.

        This method removes `existing_axis` from the sequence
        :attr:`.user_axes`.  The other axes and the coordinate reference
        system remain unchanged.

        Parameters
        ----------
        new_axis : :class:`.axis.UserDefined`
            The axis to add.

        Returns
        -------
        CoordinateSystem
            New coordinate system with axis removed from the sequence of
            user axes.

        """
        new_axes = list(self.__user_axes)
        new_axes.remove(existing_axis)
        return type(self)(
            horizontal=self.__spatial.crs,
            elevation=self.__spatial.elevation,
            time=self.__time,
            user_axes=tuple(new_axes)
        )
