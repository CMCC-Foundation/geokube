"""
Domain
======

A domain construct that contains dimension and auxiliary axes,
coordinate values and related units, coordinate reference system, etc.


Classes
-------

:class:`geokube.core.domain.Domain`
    Base class for domain constructs.

:class:`geokube.core.domain.Points`
    Point domain defined at scattered locations and times.

:class:`geokube.core.domain.Profiles`
    Profile domain defined along vertical lines at fixed locations.

:class:`geokube.core.domain.Grid`
    Gridded domain defined at a spatial and temporal grid.

"""

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
# pylint: disable=unused-import
import pint_xarray  # noqa: F401
from pyproj import Transformer
import xarray as xr

from . import axis
from .coord_system import CoordinateSystem
from .feature import (
    GridFeature, PointsFeature, ProfilesFeature, _as_points_dataset
)
from .quantity import get_magnitude, create_quantity
from .crs import CRS, Geodetic
from .units import units


# TODO: Consider making this class internal.
# TODO: Consider renaming this class to `DomainMixin`.
# NOTE: maybe we don't need this class
class Domain:
    """
    Base class for domain constructs.

    Methods
    -------
    as_xarray_dataset(coords, coord_system)
        Return xarray dataset representation of the caller.

    See Also
    --------
    :class:`geokube.core.domain.Points` :
        Point domain defined at scattered locations and times.
    :class:`geokube.core.domain.Profiles` :
        Profile domain defined along a vertical line at fixed locations.
    :class:`geokube.core.domain.Grid` :
        Gridded domain defined at a spatial and temporal grid.

    """

    __slots__ = ()

    # NOTE: `Domain` and `Field` have exactly the same method.
    @classmethod
    def _from_xarray_dataset(
        cls,
        ds: xr.Dataset, # This should be CF-compliant or use cf_mapping to be a CF-compliant
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> Self:
        obj = object.__new__(cls)
        # TODO: Make sure that `cls.__mro__[2]` returns the correct `Feature`
        # class from the inheritance hierarchy.
        feature_cls = cls.__mro__[2]
        # pylint: disable=unnecessary-dunder-call
        feature_cls.__init__(obj, ds, cf_mappings)
        return obj

    @classmethod
    def _from_xrdset(
        cls, dset: xr.Dataset, coord_system: CoordinateSystem
    ) -> Self:
        return cls(coords=dset.coords, coord_system=coord_system)

    @staticmethod
    def as_xarray_dataset(
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity | xr.DataArray],
        coord_system: CoordinateSystem
    ) -> xr.Dataset:
        """
        Return xarray dataset representation of the caller.

        Parameters
        ----------
        coords : dict_like
            Mapping of axes and the corresponding coordinates. The keys
            are the instances of the subtypes of
            :class:`geokube.core.axis.Axis` and the values are
            array-like objects, pint quantities, or xarray data arrays.
        coord_system : CoordinateSystem
            Coordinate system.

        Returns
        -------
        ds : xarray dataset
            Dataset representation of the caller.
        """
        da = coord_system.crs.as_xarray()
        r_coords = dict(coords)
        r_coords[da.name] = da
        ds = xr.Dataset(coords=r_coords)
        ds.attrs['grid_mapping'] = da.name
        return ds


class Points(Domain, PointsFeature):
    """
    Point domain defined at scattered locations and times.

    Parameters
    ----------
    coords : dict_like or sequence
        Mapping of axes and the corresponding coordinates. The keys
        are the instances of the subtypes of
        :class:`geokube.core.axis.Axis` and the values are
        array-like objects or pint quantities.  It can also be a
        two-dimensional sequence of point coordinates.
    coord_system : CoordinateSystem
        Coordinate system.

    Attributes
    ----------
    coords : dict
        Return the coordinates of a domain, with their units.
    coord_system : CoordinateSystem
        Return the coordinate system of a domain.
    crs : CRS
        Return the coordinate reference system of a domain.
    dim_axes : tuple
        Return the dimension axes of a domain.
    dim_coords : dict
        Return the dimension coordinates of a domain.
    aux_axes : tuple
        Return the auxiliary axes of a domain.
    aux_coords : dict
        Return the auxiliary coordinates of a domain.
    number_of_points : int
        The number of points.

    Methods
    -------
    sel(indexers, **xarray_kwargs)
        Return a new feature, domain, or field selected by labels.
    isel(indexers, **xarray_kwargs)
        Return a new feature, domain, or field selected by indexes.
    bounding_box(south, north, west, east, bottom, top)
        Return a subset defined with a bounding box.
    nearest_horizontal(latitude, longitude)
        Return the nearest horizontal locations from the domain.
    nearest_vertical(elevation)
        Return the nearest vertical locations from the domain.
    time_range(start, end)
        Return a subset defined within the time bounds.
    nearest_time(time)
        Return a subset with the nearest time values.
    latest()
        Return a subset with just the latest (largest) time value.
    to_netcdf(path, **xarray_kwargs)
        Write the contents to a netCDF file.
    as_xarray_dataset(coords, coord_system)
        Return xarray dataset representation of the caller.

    See Also
    --------
    :class:`geokube.core.domain.Profiles` :
        Profile domain defined along a vertical line at fixed locations.
    :class:`geokube.core.domain.Grid` :
        Gridded domain defined at a spatial and temporal grid.

    Examples
    --------

    Creating a points domain assumes instantiating a coordinate system
    and coordinates.  One way to pass the coordinates is with a mapping
    of axis-coordinate pairs:

    >>> coord_system = CoordinateSystem(
    ...     horizontal=Geodetic(),
    ...     elevation=axis.vertical,
    ...     time=axis.time
    ... )
    >>> pts_domain = Points(
    ...     coords={
    ...         axis.latitude: [35, 25],
    ...         axis.longitude: [10, 12] * units.degree_E,
    ...         axis.vertical: [0, 10] * units.meter,
    ...         axis.time: ['2023-07-12T11', '2023-08-10T15']
    ...     },
    ...     coord_system=coord_system
    ... )

    Alternatively, the coodinates can be created by passing a sequence
    of point coordinates:

    >>> pts_domain = Points(
    ...     coords=[
    ...         ('2023-07-12T11', 0, 35, 10),
    ...         ('2023-08-10T15', 10, 25, 12)
    ...     ],
    ...     coord_system=coord_system
    ... )

    """

    __slots__ = ()

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, npt.ArrayLike | pint.Quantity]
            | Sequence[Sequence]
        ),
        coord_system: CoordinateSystem
    ) -> None:
        units = coord_system.units

        match coords:
            case Mapping():
                result_coords = {}
                for axis_, coord in coords.items():
                    attrs = axis_.encoding.copy()
                    coord_units = units.get(axis_)
                    coord_dtype = attrs.pop('dtype', None)
                    coord_ = create_quantity(coord, coord_units, coord_dtype)
                    attrs['units'] = coord_.units
                    result_coords[axis_] = xr.DataArray(
                        data=coord_.magnitude, dims=self._DIMS_, attrs=attrs
                    )
                    if coord_.ndim != 1:
                        raise ValueError(
                            f"'coords' have axis {axis_} that does not have "
                            "one-dimensional values"
                        )
                if not set(coord_system.axes) <= result_coords.keys():
                    raise ValueError(
                        "'coords' must have all axes contained in the "
                        "coordinate system"
                    )
            case Sequence():
                # NOTE: This approach currently does not allow providing units.
                n_dims = {len(point) for point in coords}
                if len(n_dims) != 1:
                    raise ValueError(
                        "'coords' must have points of equal number of "
                        "dimensions"
                    )
                data = pd.DataFrame(data=coords, columns=coord_system.axes)
                result_coords = {}
                for axis_, vals in data.items():
                    attrs = axis_.encoding.copy()
                    coord_units = units.get(axis_)
                    coord_dtype = attrs.pop('dtype', None)
                    coord_ = vals.to_numpy(dtype=coord_dtype)
                    attrs['units'] = coord_units
                    result_coords[axis_] = xr.DataArray(
                        data=coord_, dims=self._DIMS_, attrs=attrs
                    )
            case _:
                raise TypeError("'coords' must be a sequence or mapping")

        ds = Domain.as_xarray_dataset(result_coords, coord_system)

        super().__init__(ds=ds)


class Profiles(Domain, ProfilesFeature):
    """
    Profile domain defined along vertical lines at fixed locations.

    Parameters
    ----------
    coords : dict_like or sequence
        Mapping of axes and the corresponding coordinates. The keys
        are the instances of the subtypes of
        :class:`geokube.core.axis.Axis` and the values are
        array-like objects or pint quantities.
    coord_system : CoordinateSystem
        Coordinate system.

    Attributes
    ----------
    coords : dict
        Return the coordinates of a domain, with their units.
    coord_system : CoordinateSystem
        Return the coordinate system of a domain.
    crs : CRS
        Return the coordinate reference system of a domain.
    dim_axes : tuple
        Return the dimension axes of a domain.
    dim_coords : dict
        Return the dimension coordinates of a domain.
    aux_axes : tuple
        Return the auxiliary axes of a domain.
    aux_coords : dict
        Return the auxiliary coordinates of a domain.
    number_of_profiles : int
        The number of profiles in a domain or field.
    number_of_levels : int
        The number of levels in a domain or field.

    Methods
    -------
    sel(indexers, **xarray_kwargs)
        Return a new feature, domain, or field selected by labels.
    isel(indexers, **xarray_kwargs)
        Return a new feature, domain, or field selected by indexes.
    bounding_box(south, north, west, east, bottom, top)
        Return a subset defined with a bounding box.
    nearest_horizontal(latitude, longitude)
        Return the nearest horizontal locations from the domain.
    nearest_vertical(elevation)
        Return the nearest vertical locations from the domain.
    time_range(start, end)
        Return a subset defined within the time bounds.
    nearest_time(time)
        Return a subset with the nearest time values.
    latest()
        Return a subset with just the latest (largest) time value.
    as_points()
        Return points representation of the data and coordinates.
    to_netcdf(path, **xarray_kwargs)
        Write the contents to a netCDF file.
    as_xarray_dataset(coords, coord_system)
        Return xarray dataset representation of the caller.

    See Also
    --------
    :class:`geokube.core.domain.Points` :
        Point domain defined at scattered locations and times.
    :class:`geokube.core.domain.Grid` :
        Gridded domain defined at a spatial and temporal grid.

    Examples
    --------

    Creating a profiles domain assumes instantiating a coordinate system
    and coordinates:

    >>> coord_system = CoordinateSystem(
    ...     horizontal=Geodetic(),
    ...     elevation=axis.vertical,
    ...     time=axis.time
    ... )
    >>> time = ['2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
    >>> vertical = (
    ...     10
    ...     + np.random.default_rng(seed=0).random(size=(5, 12))
    ...     + np.tile(np.arange(12), 5).reshape(5, 12)
    ... )
    >>> prof_domain = Profiles(
    ...     coords={
    ...         axis.latitude: [30, 35, 40, 45, 50],
    ...         axis.longitude: [10, 15, 20, 25, 30],
    ...         axis.time: time,
    ...         axis.vertical: vertical * units.bar
    ...     },
    ...     coord_system=coord_system
    ... )

    """

    __slots__ = ()

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        interm_coords = dict(coords)
        result_coords: dict[axis.Axis, xr.DataArray] = {}
        prof = (self._DIMS_[0],)
        n_prof = set()
        # FIXME: The purpose of this code is to get the actual axis passed by
        # `coords`, but it is not working as intended.
        vert_axis = ({axis.vertical} & interm_coords.keys()).pop()
        vert = interm_coords.pop(vert_axis)
        vert_attrs = vert_axis.encoding.copy()

        # Vertical.
        if isinstance(vert, Sequence):
            n_lev = [len(vert_val) for vert_val in vert]
            n_lev_tot = max(n_lev)
            n_prof_tot = len(vert)
            vert_vals = np.empty(
                shape=(n_prof_tot, n_lev_tot), dtype=vert_attrs.pop('dtype')
            )
            for i, (stop_idx, vals) in enumerate(zip(n_lev, vert)):
                if stop_idx == n_lev_tot:
                    vert_vals[i, :] = vals
                else:
                    vert_vals[i, :stop_idx] = vals
                    vert_vals[i, stop_idx:] = np.nan
            vert_qty = pint.Quantity(vert_vals, coord_system.units[vert_axis])
        else:
            vert_qty = create_quantity(
                vert,
                coord_system.units.get(vert_axis),
                vert_attrs.pop('dtype')
            )
            vert_shape = vert_qty.shape
            if len(vert_shape) != 2:
                raise ValueError(
                    "'coords' must have vertical as a two-dimensional data "
                    "structure"
                )
            n_prof_tot, n_lev_tot = vert_shape

        vert_attrs['units'] = vert_qty.units
        result_coords[vert_axis] = xr.DataArray(
            data=vert_qty.magnitude, dims=self._DIMS_, attrs=vert_attrs
        )

        # All coordinates except the vertical.
        for axis_, vals in interm_coords.items():
            attrs = axis_.encoding.copy()
            qty = create_quantity(
                vals, coord_system.units.get(axis_), attrs.pop('dtype')
            )
            attrs['units'] = qty.units
            if qty.ndim != 1:
                raise ValueError(
                    f"'coords' have axis {axis_} that does not have "
                    "one-dimensional values"
                )
            result_coords[axis_] = xr.DataArray(
                data=qty.magnitude, dims=prof, attrs=attrs
            )
            n_prof.add(qty.size)
        if len(n_prof) != 1:
            raise ValueError(
                "'coords' with the exception of vertical must have values of "
                "equal sizes"
            )
        if n_prof_tot != n_prof.pop():
            raise ValueError("'coords' have items of inappropriate sizes")
        if not set(coord_system.axes) <= result_coords.keys():
            raise ValueError(
                "'coords' must have all axes from the coordinate system"
            )

        ds = Domain.as_xarray_dataset(result_coords, coord_system)

        super().__init__(ds=ds)

    def as_points(self) -> Points:
        """
        Return points representation of the data and coordinates.

        Returns
        -------
        Points
            Points domain with the same data and coordinates.
        """
        return Points._from_xarray_dataset(_as_points_dataset(self))


class Grid(Domain, GridFeature):
    """
    Gridded domain defined at a spatial and temporal grid.

    Parameters
    ----------
    coords : dict_like or sequence
        Mapping of axes and the corresponding coordinates. The keys
        are the instances of the subtypes of
        :class:`geokube.core.axis.Axis` and the values are
        array-like objects or pint quantities.
    coord_system : CoordinateSystem
        Coordinate system.

    Attributes
    ----------
    coords : dict
        Return the coordinates of a domain, with their units.
    coord_system : CoordinateSystem
        Return the coordinate system of a domain.
    crs : CRS
        Return the coordinate reference system of a domain.
    dim_axes : tuple
        Return the dimension axes of a domain.
    dim_coords : dict
        Return the dimension coordinates of a domain.
    aux_axes : tuple
        Return the auxiliary axes of a domain.
    aux_coords : dict
        Return the auxiliary coordinates of a domain.

    Methods
    -------
    sel(indexers, **xarray_kwargs)
        Return a new feature, domain, or field selected by labels.
    isel(indexers, **xarray_kwargs)
        Return a new feature, domain, or field selected by indexes.
    bounding_box(south, north, west, east, bottom, top)
        Return a subset defined with a bounding box.
    nearest_horizontal(latitude, longitude)
        Return the nearest horizontal locations from the domain.
    nearest_vertical(elevation)
        Return the nearest vertical locations from the domain.
    infer_resolution(axis)
        Return inferred resolution for a given axis.
    as_geodetic()
        Return the transformed gridded domain with the geodetic CRS.
    spatial_transform_to(crs)
        Return the values of horizontal coordinates in a given CRS.
    time_range(start, end)
        Return a subset defined within the time bounds.
    nearest_time(time)
        Return a subset with the nearest time values.
    latest()
        Return a subset with just the latest (largest) time value.
    as_points()
        Return points representation of the data and coordinates.
    to_netcdf(path, **xarray_kwargs)
        Write the contents to a netCDF file.
    as_xarray_dataset(coords, coord_system)
        Return xarray dataset representation of the caller.

    See Also
    --------
    :class:`geokube.core.domain.Points` :
        Point domain defined at scattered locations and times.
    :class:`geokube.core.domain.Profiles` :
        Profile domain defined along a vertical line at fixed locations.

    Examples
    --------
    >>> coord_system = CoordinateSystem(
    ...     horizontal=Geodetic(),
    ...     elevation=axis.vertical,
    ...     time=axis.time
    ... )
    >>> grid_domain = Grid(
    ...     coords={
    ...         axis.time: [
    ...             '2023-10-01',
    ...             '2023-10-02',
    ...             '2023-10-03',
    ...             '2023-10-04',
    ...             '2023-10-05'
    ...         ],
    ...         axis.vertical: [0, 10, 20, 30] * units.cm,
    ...         axis.latitude: [30, 35, 40],
    ...         axis.longitude: [10, 15]
    ...     },
    ...     coord_system=coord_system
    ... )

    """

    # TODO: Consider auxiliary coordinates other than the
    # latitude and longitude. Especially consider how to represent them in the
    # API.
    # NOTE: The assumption is that the latitude and longitude as auxiliary
    # coordinates must have the dimensions either
    # `(axis.grid_latitude, axis.grid_longitude)` or `(axis.y, axis.x)`.
    __slots__ = ()

    def __init__(
        self,
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity],
        coord_system: CoordinateSystem
    ) -> None:
        if not isinstance(coords, Mapping):
            raise TypeError("'coords' must be a mapping")

        result_coords = {}
        dims: tuple[axis.Axis, ...]
        dim_axes = set(coord_system.dim_axes)
        for axis_, coord in coords.items():
            attrs = axis_.encoding.copy()
            if isinstance(coord, pd.IntervalIndex):
                if coord.closed != 'both':
                    raise NotImplementedError(
                        "'coords' contain an open interval index, which is "
                        "currently not supported"
                    )
                attrs['units'] = 'dimensionless'
                del attrs['dtype']
                coord_vals = coord.to_numpy()
                dims = (axis_,)
            else:
                coord_units = coord_system.units.get(axis_)
                coord_dtype = attrs.pop('dtype', None)
                coord_qty = create_quantity(coord, coord_units, coord_dtype)
                attrs['units'] = coord_qty.units
                coord_vals = coord_qty.magnitude
                if (
                    coord_vals.dtype is np.dtype(object)
                    and isinstance(coord_vals[0], pd.Interval)
                ):
                    coord_vals = pd.IntervalIndex(coord_vals, closed='both')

                if axis_ in dim_axes:
                    # Dimension coordinates.
                    match coord_vals.ndim:
                        case 0:
                            # Constant.
                            dims = ()
                        case 1:
                            # Oridinary dimension coordinate.
                            dims = (axis_,)
                        case _:
                            # Anything else (not allowed).
                            raise ValueError(
                                f"'coords' have a dimension axis {axis_} that "
                                "has multi-dimensional values"
                            )
                else:
                    # Auxiliary coordinates.
                    dims = coord_system.crs.dim_axes if coord_vals.ndim else ()

            result_coords[axis_] = xr.DataArray(
                data=coord_vals, dims=dims, attrs=attrs
            )

        dset = Domain.as_xarray_dataset(result_coords, coord_system)
        super().__init__(dset)

    def infer_resolution(self, axis: axis.Axis) -> float:
        """
        Return inferred resolution for a given axis.

        Parameters
        ----------
        axis : axis.Axis
            The axis to use for inferring the resolution.

        Returns
        -------
        float
            The inferred resolution for a given axis.

        """
        return self.coords[axis].ptp() / (self.coords[axis].size - 1)

    def as_geodetic(self) -> Self:
        """
        Return the transformed gridded domain with the geodetic CRS.

        Returns
        -------
        Grid
            New gridded domain with the geodetic CRS.

        """
        coord_system = CoordinateSystem(
            horizontal=Geodetic(),
            elevation = self.coord_system.elevation,
            time = self.coord_system.time,
            user_axes = self.coord_system.user_axes
        )

        # Infering latitude and longitude steps from the x and y coordinates.
        # this works only for Geodetic and Rotated Pole
        # It should be generalized also for projections
        # TODO: once we get the resolution for the horizontal we should transform
        # in a value in lat/lon 
        lat_step = self.infer_resolution(self.crs.dim_Y_axis)
        lon_step = self.infer_resolution(self.crs.dim_X_axis)

        # Building regular latitude-longitude coordinates.
        lat_vals = self.coords[axis.latitude]
        lon_vals = self.coords[axis.longitude]
        south, north = lat_vals.min().magnitude, lat_vals.max().magnitude
        west, east = lon_vals.min().magnitude, lon_vals.max().magnitude
#        lat = np.arange(south, north + lat_step / 2, lat_step)
#        lon = np.arange(west, east + lon_step / 2, lon_step)
        lat = np.arange(south, north, lat_step)
        lon = np.arange(west, east, lon_step)
        
        coords = self.coords  # Or `self.coords.copy()`.
        for axis_ in self.crs.axes:
            del coords[axis_]
        hor_coords = {
            coord_system.crs.dim_Y_axis: lat,
            coord_system.crs.dim_X_axis: lon
        }
        coords |= hor_coords

        return type(self)(coords=coords, coord_system=coord_system)

    def spatial_transform_to(self, crs: CRS) -> tuple:
        """
        Return the values of horizontal coordinates in a given CRS.

        Parameters
        ----------
        crs : CRS
            Target coordinate reference system.

        Returns
        -------
        tuple
            The values of horizontal coordinates after the
            transformation.

        """
        # TODO: we assume that they have the same datum. We need to change
        # the code when datum are different!!
        # 
        lat = get_magnitude(self.coords[axis.latitude], units['degrees_N'])
        lon = get_magnitude(self.coords[axis.longitude], units['degrees_E'])
        if lat.ndim == lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        transformer = Transformer.from_crs(
            crs_from=self.crs._crs,
            crs_to=crs._crs,
            always_xy=True
        )
        # x, y = transformer.transform(lon, lat)
        return transformer.transform(lon, lat)

    def as_points(self) -> Points:
        """
        Return points representation of the data and coordinates.

        Returns
        -------
        Points
            Points domain with the same data and coordinates.
        """
        return Points._from_xarray_dataset(_as_points_dataset(self))
