"""
Feature
=======

A feature construct that serves as a base class for domains and fields.


Classes
-------

:class:`geokube.core.feature.FeatureMixin`
    Mixin class for common domain and field properties and methods.

:class:`geokube.core.feature.Feature`
    Base class for feature, domain, and field constructs.

:class:`geokube.core.feature.PointsFeature`
    Feature defined on a point domain.

:class:`geokube.core.feature.ProfilesFeature`
    Feature defined on a profile domain.

:class:`geokube.core.feature.GridFeature`
    Feature defined on a gridded domain.

"""

# NOTE: There is probably no need to use the full path to the class from this
# module in the docstrings. 

from collections.abc import Mapping
from datetime import date, datetime
from numbers import Number
from typing import Self, Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
# pylint: disable=unused-import
import pint_xarray  # noqa: F401
import xarray as xr

from . import axis, indexes
from .coord_system import CoordinateSystem
from .crs import CRS, Geodetic, RotatedGeodetic
from .indexers import get_array_indexer, get_indexer
from .quantity import create_quantity, get_magnitude
from .units import units


# NOTE:
# inherit from xr Dataset do not work since it is not possible to create 
# a new dataset with our own index!!!
# let's use compose for the moment
#
# class Feature(xr.Dataset):
#
# This is a wrapper for an xarray Dataset CF-compliant with indexes 
# which allow to perform spatial operations like bbox, nearest, ...
# enhanced with the coordinate system class.
# 


class FeatureMixin:
    """
    Mixin class for common domain and field properties and methods.

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
    time_range(start, end)
        Return a subset defined within the time bounds.
    nearest_time(time)
        Return a subset with the nearest time values.
    latest()
        Return a subset with just the latest (largest) time value.
    to_netcdf(path, **xarray_kwargs)
        Write the contents to a netCDF file.

    """

    @property
    def coords(self) -> dict[axis.Axis, pint.Quantity]:
        """
        Return the coordinates of a domain, with their units.

        The coordinates contain the axes, coordinate values, and
        coordinate units.  They are represented as a `dict` with the
        :class:`geokube.core.axis.Axis` instance keys and
        :class:`pint.Quantity` values.

        """
        return self._coords

    @property
    def coord_system(self) -> CoordinateSystem:
        """
        Return the coordinate system of a domain.

        The coordinate system is an instance of the class
        :class:`geokube.core.coord_system.CoordinateSystem`.  It
        contains the information on the spatial, temporal, and other
        axes and related units, as well as about the horizontal
        coordinate reference system.

        """
        return self._coord_system

    @property
    def crs(self) -> CRS:
        """
        Return the coordinate reference system of a domain.

        The horizontal coordinate reference system is an instance of a
        subclass of the class :class:`geokube.core.crs.CRS`.  It
        contains dimension and auxiliary horizontal axes, and can be
        used for transformations.

        """
        return self.coord_system.spatial.crs

    @property
    def dim_axes(self) -> tuple[axis.Axis, ...]:
        """
        Return the dimension axes of a domain.

        There can be any number of kind of axes.  They are represented
        as a `tuple` of :class:`geokube.core.axis.Axis` instances.

        """
        return self.coord_system.dim_axes

    @property
    def dim_coords(self) -> dict[axis.Axis, pint.Quantity]:
        """
        Return the dimension coordinates of a domain.

        The coordinates contain the axes, coordinate values, and
        coordinate units.  They are represented as a dict with the
        :class:`geokube.core.axis.Axis` instance keys and
        :class:`pint.Quantity` values.

        """
        return self._dim_coords

    @property
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        """
        Return the auxiliary axes of a domain.

        Auxiliary axes can be only horizontal.  They are represented
        as a tuple of :class:`geokube.core.axis.Horizontal` instances.

        """
        return self.coord_system.aux_axes

    @property
    def aux_coords(self) -> dict[axis.Axis, pint.Quantity]:
        """
        Return the auxiliary coordinates of a domain.

        The coordinates contain the axes, coordinate values, and
        coordinate units.  They are represented as a dict with the
        :class:`geokube.core.axis.Axis` instance keys and
        :class:`pint.Quantity` values.

        """
        return self._aux_coords

    def sel(
        self, indexers: Mapping[axis.Axis, Any] | None = None,
        **xarray_kwargs
    ) -> Self:
        """
        Return a new feature, domain, or field selected by labels.

        The subsetting operation can be done only along the dimensions.
        Subsetting is unit-aware, which means that unit conversion can
        be performed on the labels in the background if necesssary.
        The coordinates and data variables are reduced.  The type of
        the caller and other data are preserved.

        Parameters
        ----------
        indexers : dict-like, optional
            A mapping with the keys that must be the dimensional axes
            and the values that represent the corresponding labels.
            The values can be scalars, arrays, or slices. They can have
            the units (i.e. can be instances of the class
            :class:`pint.Quantity`).  If a label does not have a unit,
            then it is assumed that its unit is the default unit for
            the corresponding axis (key).

        xarray_kwargs : dict, optional
            Keyword arguments passed to :meth:`xarray.Dataset.sel`.

        Returns
        -------
        out : caller type
            New feature, domain, or field created as a result of
            subsetting.  The type of the subsetted object is preserved.

        """
        dset = self._dset.sel(indexers=indexers, **xarray_kwargs)
        out = self._from_xarray_dataset(dset)
        return out

    def isel(
        self, indexers: Mapping[axis.Axis, Any] | None = None,
        **xarray_kwargs
    ) -> Self:
        """
        Return a new feature, domain, or field selected by indexes.

        The subsetting operation can be done only along the dimensions.
        The coordinates and data variables are reduced.  The type of
        the caller and other data are preserved.

        Parameters
        ----------
        indexers : dict-like, optional
            A mapping with the keys that must be the dimensional axes
            and the values that represent the corresponding integer
            indexes.

        xarray_kwargs : dict, optional
            Keyword arguments passed to :meth:`xarray.Dataset.isel`.

        Returns
        -------
        out : caller type
            New feature, domain, or field created as a result of
            subsetting.  The type of the subsetted object is preserved.

        """
        dset = self._dset.isel(indexers, **xarray_kwargs)
        out = self._from_xarray_dataset(dset)
        return out

    # Spatial subsetting ------------------------------------------------------

    def bounding_box(
        self,
        south: Number | pint.Quantity | None = None,
        north: Number | pint.Quantity | None = None,
        west: Number | pint.Quantity | None = None,
        east: Number | pint.Quantity | None = None,
        bottom: Number | pint.Quantity | None = None,
        top: Number | pint.Quantity | None = None
    ) -> Self:
        """
        Return a subset defined with a bounding box.

        This operation extracts a subset of values within a bounding
        box specified with the parameters north, south, east, and west.
        It is unit-aware, which means that the units do not need to be
        converted before passing to the method.  If a quantity is
        passed, the units are automatically adjusted.  If just a number
        is provided, it is assumed that the units are the same as the
        corresponding coordinate.

        Parameters
        ----------
        south, north, west, east : number or quantity, optional
            Horizontal bounds.
        bottom, top : number or quantity, optional
            Vertical bounds.

        Returns
        -------
        feature : caller type
            A feature, domain, or field obtained after subsetting the
            caller.

        """
        # TODO: manage when north, south, west and east are None
        # we need to consider min/max for lat/lon
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        feature = self.sel(h_idx)
        if not (bottom is None and top is None):
            feature = feature.sel({axis.vertical: slice(bottom, top)})
        return feature

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> Self:
        """
        Return the nearest horizontal locations from the domain.

        Parameters
        ----------
        latitude, longitude : array_like or quantity
            Horizontal coordinates for subsetting.

        Returns
        -------
        caller type
            A feature, domain, or field with the the nearest horizontal
            locations from the domain to all latitude-longitude pairs.

        """
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        return self.sel(idx, method='nearest', tolerance=np.inf)

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        """
        Return the nearest vertical locations from the domain.

        This operation is unit-aware, which means that the units do not
        need to be converted before passing to the method.  If a
        quantity is passed, the units are automatically adjusted.  If
        just a number or array is provided, it is assumed that the units
        are the same as the corresponding coordinate.

        Parameters
        ----------
        elevation : array_like or quantity
            Vertical coordinates for subsetting.

        Returns
        -------
        caller type
            A feature, domain, or field with the the nearest vertical
            locations to the elevation.

        """
        idx = {axis.vertical: elevation}
        return self.sel(idx, method='nearest', tolerance=np.inf)

    # Temporal subsetting -----------------------------------------------------

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        """
        Return a subset defined within the time bounds.

        This operation extracts a subset of values along the time axis,
        within the time interval between `start` and `end`.

        Parameters
        ----------
        start, end : date, datetime, or str, optional
            Time bounds.

        Returns
        -------
        caller type
            A feature, domain, or field obtained after subsetting the
            caller along the time axis.

        """
        idx = {axis.time: slice(start, end)}
        return self.sel(idx)

    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        """
        Return a subset with the nearest time values.

        This operation extracts a subset of values along the time axis,
        which are nearest to the values specified with `time`.

        Parameters
        ----------
        time : array_like, optional
            Time values to search for.

        Returns
        -------
        caller type
            A feature, domain, or field obtained after subsetting the
            caller along the time axis.

        """
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        return self.sel(idx, method='nearest', tolerance=None)

    def latest(self) -> Self:
        """
        Return a subset with just the latest (largest) time value.

        Returns
        -------
        caller type
            A feature, domain, or field obtained after subsetting the
            caller along the time axis.

        Raises
        ------
        NotImplementedError
            If the time axis is not present.

        """
        if axis.time not in self._dset.coords:
            raise NotImplementedError()
        latest = self._dset[axis.time].max().astype(str).item()
        idx = {axis.time: slice(latest, latest)}
        return self.sel(idx)

    # Writing to file ---------------------------------------------------------

    def to_netcdf(self, path: str, **xarray_kwargs) -> None:
        """
        Write the contents to a netCDF file.

        This operation extracts a subset of values along the time axis,
        which are nearest to the values specified with `time`.

        Parameters
        ----------
        path : str
            The path of the target netCDF file.
        **xarray_kwargs : dict, optional
            Extra arguments to :meth:`xarray.Dataset.to_netcdf`.

        """
        dset = self._dset
        dset = dset.drop_indexes(dset.xindexes)

        coords = dict(dset.coords)
        axes = {}
        for axis_, coord in coords.items():
            if isinstance(axis_, axis.Axis):
                axes[axis_] = (
                    coord.attrs.get('standard_name')
                    or axis_.encoding['standard_name']
                )
        dims = {}
        for dim in dset.dims:
            names = set(dims.values())
            if (
                isinstance(dim, axis.Axis)
                and (dim_name := dim.encoding['standard_name']) not in names
            ):
                dims[dim] = dim_name
        dset = dset.rename_vars(axes).swap_dims(dims)

        if (
            (time_coord := coords.get(axis.time)) is not None
            and (
                (bnds := time_coord.encoding.get('bounds'))
                in {'time_bnds', 'time_bounds'}
            )
            and isinstance(time_coord.data[0], pd.Interval)
        ):
            inter = pd.IntervalIndex(time_coord.to_numpy())
            left, right = inter.left.to_numpy(), inter.right.to_numpy()
            time_vals = np.empty(shape=(time_coord.size, 2), dtype=left.dtype)
            time_vals[:, 0], time_vals[:, 1] = left, right
            dset = dset.assign_coords(
                coords={
                    bnds: xr.Variable(dims=('time', 'bnds'), data=time_vals),
                    'time': xr.Variable(
                        dims=('time',),
                        data=left,
                        attrs=time_coord.attrs,
                        encoding=time_coord.encoding
                    )
                }
            )

        # TODO: Use `itertools.chain` to create a single loop.
        for coord in dset.coords.values():
            attrs = coord.attrs
            if 'units' in attrs and units[attrs['units']] == units[None]:
                del attrs['units']
        for var in dset.data_vars.values():
            attrs = var.attrs
            if 'units' in attrs and units[attrs['units']] == units[None]:
                del attrs['units']

        dset.attrs.pop('grid_mapping', None)
        if gmn := dset.cf.grid_mapping_names:
            crs_var_name = next(iter(gmn.values()))[0]
            for var in dset.data_vars.values():
                var.attrs.pop('grid_mapping', None)
                var.encoding.setdefault('grid_mapping', crs_var_name)

        # TODO: Improve squeezing.
        dset = dset.squeeze()

        dset.to_netcdf(path, **xarray_kwargs)


class Feature(FeatureMixin):
    """
    Base class for feature constructs.

    Parameters
    ----------
    ds : xarray dataset
        CF-compliant dataset with all the inputs.
    cf_mappings : dict_like
        CF-compliant hints

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
    time_range(start, end)
        Return a subset defined within the time bounds.
    nearest_time(time)
        Return a subset with the nearest time values.
    latest()
        Return a subset with just the latest (largest) time value.
    to_netcdf(path, **xarray_kwargs)
        Write the contents to a netCDF file.

    See Also
    --------
    :class:`geokube.core.feature.PointsFeature` :
        Feature defined on a point domain.
    :class:`geokube.core.feature.ProfilesFeature` :
        Feature defined on a profile domain.
    :class:`geokube.core.feature.GridFeature` :
        Feature defined on a gridded domain.

    """

    __slots__ = (
        '_dset', '_coord_system', '_coords', '_aux_coords', '_dim_coords'
    )

    def __init__(
        self,
        ds: xr.Dataset, # This should be CF-compliant or use cf_mapping to be a CF-compliant
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> None:
        # TODO: check if xarray dataset is CF compliant (otherwise raise an error)       
        # Horizontal coordinate system:
        # TODO: manage cf_mappings

        self._dset = ds
        ds_coords = dict(ds.coords)
        if gmn := ds.cf.grid_mapping_names:
            crs_var_name = next(iter(gmn.values()))[0]
            hor_crs = CRS.from_cf(ds[crs_var_name].attrs)
            ds_coords.pop(crs_var_name)
        else:
            # TODO: implement a function to guess the CRS
            hor_crs = Geodetic()

        # Coordinates.
        coords = {}
        for cf_coord, cf_coord_names in ds.cf.coordinates.items():
            assert len(cf_coord_names) == 1
            cf_coord_name = cf_coord_names[0]
            coord = ds_coords.pop(cf_coord_name)
            axis_ = axis._from_string(cf_coord)
            coords[axis_] = pint.Quantity(
                coord.to_numpy(), coord.attrs.get('units')
            )

        for cf_axis, cf_axis_names in ds.cf.axes.items():
            assert len(cf_axis_names) == 1
            cf_axis_name = cf_axis_names[0]
            if cf_axis_name in ds_coords:
                coord = ds_coords.pop(cf_axis_name)
                axis_ = axis._from_string(cf_axis.lower())
                if isinstance(hor_crs, RotatedGeodetic):
                    if axis_ is axis.x:
                        axis_ = axis.grid_longitude
                    elif axis_ is axis.y:
                        axis_ = axis.grid_latitude
                coords[axis_] = pint.Quantity(
                    coord.to_numpy(), coord.attrs.get('units')
                )

        # Coordinate system.
        time = {
            axis_
            for axis_ in coords
            if isinstance(axis_, axis.Time) and coords[axis_].ndim
        }
        assert len(time) <= 1
        elev = {
            axis_
            for axis_ in coords
            if isinstance(axis_, axis.Elevation) and coords[axis_].ndim
        }
        assert len(elev) <= 1
        # TODO: Add user axes.
        coord_system = CoordinateSystem(
            horizontal=hor_crs,
            elevation=elev.pop() if elev else None,
            time=time.pop() if time else None
        )

        self._coord_system = coord_system

        # self._coords = {
        #     axis_: ds[axis_].pint.quantify().data
        #     for axis_ in coord_system.axes
        # }
        coords = {}
        for axis_ in coord_system.axes:
            darr = ds[axis_]
            coords[axis_] = create_quantity(
                darr, darr.attrs.get('units'), darr.dtype
            )
        self._coords = coords

        self._dim_coords = {ax: ds[ax].data for ax in coord_system.dim_axes}

        self._aux_coords = {ax: ds[ax].data for ax in coord_system.aux_axes}

    def __eq__(self, other, /) -> bool:
        """Return ``self == other``."""
        if type(self) is not type(other):
            return False
        return self._dset.identical(other._dset)

    def __ne__(self, other, /) -> bool:
        """Return ``self != other``."""
        return not self == other

    def _get_var_names(self) -> set[str]:
        all_vars = set()
        anc_vars = set()
        for var_name, var in self._dset.data_vars.items():
            all_vars.add(var_name)
            if (anc_attr := var.attrs.get('ancillary_variables')) is not None:
                anc_vars |= set(anc_attr.split(' '))
        return all_vars - anc_vars

    @classmethod
    def _from_xarray_dataset(
        cls,
        ds: xr.Dataset,
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> Self:
        return cls(ds, cf_mappings)

    #TODO: Implement __getitem__ ??


class PointsFeature(Feature):
    """
    Feature defined on a point domain.

    Parameters
    ----------
    ds : xarray dataset
        CF-compliant dataset with all the inputs.
    cf_mappings : dict_like
        CF-compliant hints

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

    See Also
    --------
    :class:`geokube.core.feature.ProfilesFeature` :
        Feature defined on a profile domain.
    :class:`geokube.core.feature.GridFeature` :
        Feature defined on a gridded domain.

    """

    __slots__ = ('_n_points',)
    _DIMS_ = ('_points',)

    def __init__(
        self,
        ds: xr.Dataset, # This dataset should check for _DIMS_ that is points
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> None:

        # TODO: check if ds is a Points Features -> _points dim should exist

        super().__init__(ds=ds, cf_mappings=cf_mappings)

        hor_axes = set(self.crs.axes)
        for axis_ in self.coord_system.axes:
            if axis_ not in hor_axes:
                self._dset = self._dset.set_xindex(axis_, indexes.OneDimIndex)
        self._dset = self._dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )

    @property
    def number_of_points(self) -> int:
        """Returns the number of points."""
        return self._dset['_points'].size


class ProfilesFeature(Feature):
    """
    Feature defined on a profile domain.

    Parameters
    ----------
    ds : xarray dataset
        CF-compliant dataset with all the inputs.
    cf_mappings : dict_like
        CF-compliant hints

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

    See Also
    --------
    :class:`geokube.core.feature.PointsFeature` :
        Feature defined on a point domain.
    :class:`geokube.core.feature.GridFeature` :
        Feature defined on a gridded domain.

    """

    __slots__ = ('_n_profiles', '_n_levels')
    _DIMS_ = ('_profiles', '_levels')

    def __init__(
        self,
        ds: xr.Dataset,
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> None:

        # TODO: check if it is a profile features (_profiles and _levels dims should exist)

        super().__init__(
            ds=ds,
            cf_mappings=cf_mappings
        )

        for axis_ in self.coord_system.axes:
            if axis_ not in set(self.coord_system.spatial.axes):
                self._dset = self._dset.set_xindex(axis_, indexes.OneDimIndex)

        self._dset = self._dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )
        self._dset = self._dset.set_xindex(
            axis.vertical, indexes.TwoDimVertProfileIndex
        )

    @property
    def number_of_profiles(self) -> int:
        """Returns the number of profiles in a domain or field."""
        return self._dset['_profiles'].size

    @property
    def number_of_levels(self) -> int:
        """Returns the number of levels in a domain or field."""
        return self._dset['_levels'].size

    def bounding_box(
        self,
        south: Number | None = None,
        north: Number | None = None,
        west: Number | None = None,
        east: Number | None = None,
        bottom: Number | None = None,
        top: Number | None = None
    ) -> Self:
        """
        Return a subset defined with a bounding box.

        This operation extracts a subset of values within a bounding
        box specified with the parameters north, south, east, and west.
        It is unit-aware, which means that the units do not need to be
        converted before passing to the method.  If a quantity is
        passed, the units are automatically adjusted.  If just a number
        is provided, it is assumed that the units are the same as the
        corresponding coordinate.

        Parameters
        ----------
        south, north, west, east : number or quantity, optional
            Horizontal bounds.
        bottom, top : number or quantity, optional
            Vertical bounds.

        Returns
        -------
        feature : caller type
            A feature, domain, or field obtained after subsetting the
            caller.

        """
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        feature = self.sel(h_idx)

        if not (bottom is None and top is None):
            # TODO: Try to move this functionality to
            # `indexes.TwoDimHorPointsIndex.sel`.
            warn(
                "'bounding_box' loads in memory and makes a copy of the data "
                "and vertical coordinate when 'bottom' or 'top' is not 'None'"
            )
            # TODO: Make `vert_axis` the actual vertical axis from the
            # coordinates.
            v_slice = slice(bottom, top)
            v_idx = {axis.vertical: v_slice}
            new_data = feature.sel(v_idx)._dset
            new_data = new_data.drop_indexes(
                coord_names=list(new_data.xindexes.keys())
            )
            vert = new_data[axis.vertical]
            vert_mag, vert_units = vert.to_numpy(), vert.attrs['units']
            mask = get_indexer(
                [vert_mag], [get_magnitude(v_slice, vert_units)]
            )[0]
            masked_vert = np.where(mask, vert_mag, np.nan)
            new_data[axis.vertical].to_numpy()[:] = masked_vert
            for darr in new_data.data_vars.values():
                mag = darr.to_numpy()
                masked_data = np.where(mask, mag, np.nan)
                darr.to_numpy()[:] = masked_data
            feature = self._from_xarray_dataset(new_data)
        return feature

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        """
        Return the nearest vertical locations from the domain.

        This operation is unit-aware, which means that the units do not
        need to be converted before passing to the method.  If a
        quantity is passed, the units are automatically adjusted.  If
        just a number or array is provided, it is assumed that the units
        are the same as the corresponding coordinate.

        Parameters
        ----------
        elevation : array_like or quantity
            Vertical coordinates for subsetting.

        Returns
        -------
        caller type
            A feature, domain, or field with the the nearest vertical
            locations to the elevation.

        """
        # TODO: Try to move this functionality to
        # `indexes.TwoDimHorPointsIndex.sel`.
        dset = self._dset
        new_coords = {
            axis_: coord.variable for axis_, coord in dset.coords.items()
        }
        # NOTE: The purpose of this code is to get the actual vertical axis
        # passed by `coords`. It can differ from the default `axis.vertical` in
        # `encoding`. The set intersection cannot be used.
        vert_axis: axis.vertical
        for axis_ in new_coords:
            if axis_ == axis.vertical:
                vert_axis = axis_
                break
        vert = dset[vert_axis]
        vert_mag, vert_units = vert.to_numpy(), vert.attrs['units']
        n_profiles = vert_mag.shape[0]
        shape = (n_profiles, len(elevation))
        new_vert_mag = np.empty(shape=shape, dtype=vert_mag.dtype)
        # TODO: Try to implement this in a more efficient way.
        level_indices = []
        for profile_idx in range(n_profiles):
            level_idx = get_indexer(
                [vert_mag[profile_idx, :]],
                [get_magnitude(elevation, vert_units)],
                return_all=False,
                method='nearest',
                tolerance=np.inf
            )
            level_indices.append(level_idx)
            new_vert_mag[profile_idx, :] = vert_mag[profile_idx, level_idx]
        new_coords[vert_axis] = xr.Variable(
            dims=vert.dims, data=new_vert_mag, attrs=vert.attrs
        )

        new_data_vars = {}
        for name, darr in dset.data_vars.items():
            data_mag = darr.to_numpy()
            new_data_mag = np.empty(shape=shape, dtype=data_mag.dtype)
            # TODO: Try to implement this in a more efficient way.
            for profile_idx in range(n_profiles):
                level_idx = level_indices[profile_idx]
                new_data_mag[profile_idx, :] = data_mag[profile_idx, level_idx]
            new_data_vars[name] = xr.DataArray(
                data=new_data_mag,
                dims=self._DIMS_,
                coords=new_coords,
                attrs=darr.attrs
            )

        new_dset = xr.Dataset(
            data_vars=new_data_vars, coords=new_coords, attrs=dset.attrs
        )
        feature = self._from_xarray_dataset(new_dset)
        return feature

    def as_points(self) -> PointsFeature:
        """
        Return points representation of the data and coordinates.

        Returns
        -------
        PointsFeature
            Points feature with the same data and coordinates.
        """


class GridFeature(Feature):
    """
    Feature defined on a gridded domain.

    Parameters
    ----------
    ds : xarray dataset
        CF-compliant dataset with all the inputs.
    cf_mappings : dict_like
        CF-compliant hints

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

    See Also
    --------
    :class:`geokube.core.feature.PointsFeature` :
        Feature defined on a point domain.
    :class:`geokube.core.feature.ProfilesFeature` :
        Feature defined on a profile domain.

    """

    __slots__ = ('_DIMS_',)

    def __init__(
        self,
        ds: xr.Dataset,
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> None:
        super().__init__(ds=ds, cf_mappings=cf_mappings)        

        # TODO: Check if it is a Grid Feature ???

        # self._dims = 
        self._DIMS_ = self.coord_system.dim_axes # this depends on the Coordinate System

        # for axis_ in self._DIMS_:
        #     if axis_ not in self._dset.xindexes:
        #         self._dset = self._dset.set_xindex(axis_)
        self._dset = self._dset.drop_indexes(
            coord_names=list(self._dset.xindexes.keys())
        )
        for axis_ in self._DIMS_:
            self._dset = self._dset.set_xindex(
                axis_, indexes.OneDimPandasIndex
            )
        if {axis.latitude, axis.longitude} == set(self.aux_axes):
            self._dset = self._dset.set_xindex(
                self.aux_axes, indexes.TwoDimHorGridIndex
            )

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity,
        as_points: bool = True
    ) -> Self | PointsFeature:
        """
        Return the nearest horizontal locations from the domain.

        Parameters
        ----------
        latitude, longitude : array_like or quantity
            Horizontal coordinates for subsetting.

        Returns
        -------
        caller type or PointsFeature
            A feature, domain, or field with the the nearest horizontal
            locations from the domain to all latitude-longitude pairs.

        """
        lat, lon = self._dset[axis.latitude], self._dset[axis.longitude]
        lat_vals, lon_vals = lat.to_numpy(), lon.to_numpy()
        if isinstance(self.crs, Geodetic):
            lat_vals, lon_vals = np.meshgrid(lat_vals, lon_vals, indexing='ij')
        dims = (self.crs.dim_Y_axis, self.crs.dim_X_axis)

        lat_labels = get_magnitude(latitude, lat.attrs['units'])
        lon_labels = get_magnitude(longitude, lon.attrs['units'])

        idx = get_array_indexer(
            [lat_vals, lon_vals],
            [lat_labels, lon_labels],
            method='nearest',
            tolerance=np.inf,
            return_all=False
        )
        result_idx = dict(zip(dims, idx))

        # Example:
        # lat_vals = np.array([[30., 31.], [35., 36.], [40., 41.]])
        # lon_vals = np.array([[10., 11.], [15., 16.], [20., 21.]])
        # lat_labels = [29, 42]
        # lon_labels = [5, 18]
        # idx:
        # (array([0, 2]), array([0, 0]))
        # result_idx:
        # {axis.grid_latitude: array([0, 2]),
        #  axis.grid_longitude: array([0, 0])}

        dset = self._dset.isel(indexers=result_idx)
        result = self._from_xarray_dataset(dset)
        if as_points:
            return result.as_points()
        return result

    def as_points(self) -> PointsFeature:
        """
        Return points representation of the data and coordinates.

        Returns
        -------
        PointsFeature
            Points feature with the same data and coordinates.
        """
        return PointsFeature._from_xarray_dataset(_as_points_dataset(self))


def _to_points_dict(feature: Feature) -> dict[axis.Axis, pint.Quantity]:
    dset = feature._dset.drop_vars(names=[feature._dset.attrs['grid_mapping']])
    coords_copy = dict(dset.coords)
    internal_dims = set()

    n_vals = 1
    for dim_axis in dset.dims:
        if dim_axis not in coords_copy:
            coords_copy[dim_axis] = dset[dim_axis]
            internal_dims.add(dim_axis)
        n_vals *= coords_copy[dim_axis].size
    n_reps = n_vals
    idx, data = {}, {}

    # Dimension coordinates.
    for dim_axis in dset.dims:
        coord = coords_copy.pop(dim_axis)
        n_coord_vals = coord.size
        n_tiles = n_vals // n_reps
        n_reps //= n_coord_vals
        coord_idx = np.arange(n_coord_vals)
        coord_idx = np.tile(np.repeat(coord_idx, n_reps), n_tiles)
        idx[dim_axis] = coord_idx
        data[dim_axis] = pint.Quantity(
            coord.data[coord_idx], coord.attrs.get('units')
        )
    if internal_dims:
        for dim_axis in internal_dims:
            del data[dim_axis]

    # Auxiliary coordinates.
    for aux_axis, coord in coords_copy.items():
        coord_idx = tuple(idx[dim] for dim in coord.dims)
        data[aux_axis] = pint.Quantity(
            coord.data[coord_idx], coord.attrs['units']
        )

    names = feature._get_var_names()

    if not names:
        return data

    # Data.
    assert len(names) == 1
    name = names.pop()
    darr = dset[name]
    vals = darr.to_numpy().reshape(n_vals)
    data[name] = pint.Quantity(vals, darr.attrs.get('units'))

    return data


def _as_points_dataset(feature: Feature) -> xr.Dataset:
    # Creating the resulting points field.
    new_coords = {
        axis_: xr.DataArray(
            data=coord.magnitude,
            dims=('_points',),
            attrs=feature._dset[axis_].attrs
        )
        for axis_, coord in _to_points_dict(feature).items()
    }

    # Creating the resulting data.
    if names := feature._get_var_names():
        assert len(names) == 1
        name = names.pop()
        new_data = {name: new_coords.pop(name)}
    else:
        new_data = None

    # Creating the resulting dataset and corresponding result.
    new_dset = xr.Dataset(
        data_vars=new_data, coords=new_coords, attrs=feature._dset.attrs
    )
    return new_dset
