"""
Field
=====

A field construct that contains a data variable with related units,
domain, ancillary constructs, properties, etc.

Classes
-------

:class:`geokube.core.field.Field`
    Base class for field constructs.

:class:`geokube.core.field.PointsField`
    Field defined on a point domain.

:class:`geokube.core.field.ProfilesField`
    Field defined on a profile domain.

:class:`geokube.core.field.GridField`
    Field defined on a gridded domain.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable, Self, TYPE_CHECKING

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import pyarrow as pa
from pyproj import Transformer
import xarray as xr

from . import axis
from .cell_method import CellMethod
from .coord_system import CoordinateSystem
from .crs import Geodetic
from .domain import Domain, Grid, Points, Profiles
from .feature import (
    PointsFeature, ProfilesFeature, GridFeature, _as_points_dataset
)
from .quantity import create_quantity

_ARRAY_TYPES = (np.ndarray, da.Array)

# TODO: Consider making this class internal.
class Field:
    """
    Base class for field constructs.

    Attributes
    ----------
    domain : Domain
        The domain construct over which a field is defined.
    ancillary : dict or quantity
        Ancillary variables and data of a field.
    name : str
        The name of a represented quantity.
    data : array_like or quantity
        The values of a single physical variable.
    properties : dict
        The static metadata of the field
    encoding : dict
        The encoding of a data variable.
    cell_method : CellMethod or None
        The cell method of a field if any, otherwise None.

    See Also
    --------
    :class:`geokube.core.field.PointsField` :
        Field defined on a point domain.
    :class:`geokube.core.field.ProfilesField` :
        Field defined on a profile domain
    :class:`geokube.core.field.GridField` :
        Field defined on a gridded domain.

    """

    __slots__ = ()

    if TYPE_CHECKING:
        _DOMAIN_CLS_: type[Domain]
        _dset: xr.Dataset
        _name: str

    # TODO: Add cell methods
    # __slots__ = ('_cell_method_',)

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
        names = feature_cls._get_var_names(obj)
        obj._name = names.pop()
        return obj

    @classmethod
    def as_xarray_dataset(
        cls,
        data_vars: Mapping[str, npt.ArrayLike | pint.Quantity | xr.DataArray],
        coords: Mapping[axis.Axis, npt.ArrayLike | pint.Quantity | xr.DataArray],
        coord_system: CoordinateSystem
    ) -> xr.Dataset:
        da = coord_system.crs.as_xarray()
        r_coords = coords
        r_coords[da.name] = da
        ds = xr.Dataset(
            coords=r_coords
        )
        ds.attrs['grid_mapping'] = da.name
        return ds

    @property # TODO: define setter method
    def domain(self) -> Domain:
        """
        Return the domain construct over which a field is defined.

        A domain construct that contains dimension and auxiliary axes,
        coordinate values and related units, coordinate reference
        system, etc.  The type of a domain corresponts to the type of a
        field.

        See Also
        --------
        :class:`geokube.core.domain.Points` :
            Point domain defined at scattered locations and times

        :class:`geokube.core.domain.Profiles` :
            Profile domain defined along a vertical line at fixed
            locations

        :class:`geokube.core.domain.Grid` :
            Gridded domain defined at a spatial and temporal grid

        """
        coords = {}
        for ax in self.coord_system.axes:
            coords[ax] = self.coords[ax]
        return self._DOMAIN_CLS_(
                coords=coords, 
                coord_system=self.coord_system)    

    @property
    def ancillary(self, name: str | None = None) -> dict | pint.Quantity:
        """
        Return ancillary variables and data of a field.

        The ancillary variables are stored in a dictionary of
        multidimensional arrays that contains ancillary metadata which
        vary within the domain.

        """
        # TODO: Check this!
        if name is not None:
            # TODO: Quantify the data.
            return self[name].data
        ancillary_ = {}
        for c in self._dset.data_vars:
            if c != self.name:
                ancillary_[c] = self._dset[c].data
        return ancillary_

    @property # return field name
    def name(self) -> str:
        """Return the name of a represented quantity."""
        return self._name

    @property # define data method to return field data
    def data(self) -> pint.Quantity:
        """
        Return the values of a single physical variable.

        Data are stored in a multidimensional array defined over the
        domain.  The order of the array dimensions should correspond to
        the order of the axes in the coordinate system of the domain.
        This array can be an instance of :class:`numpy.ndarray`,
        :class:`dask.array.core.Array`, or :class:`pint.Quantity`.

        """
        darr = self._dset[self.name]
        qty = create_quantity(darr.data, darr.attrs.get('units'), darr.dtype)
        return qty

    @property
    def properties(self) -> dict:
        """Return the static metadata of the field."""
        return self._dset[self.name].attrs

    @property
    def encoding(self) -> dict:
        """Return the The encoding of a data variable."""
        return self._dset[self.name].encoding

    @property
    def cell_method(self) -> CellMethod | None:
        """
        Return the cell method of a field if any, otherwise None.

        A CellMethod object describes how data represent variation
        within the cells of a field.  If a field does not have a cell
        method, it is None.

        """
        darr = self._dset[self.name]
        if (cmethod := darr.attrs.get('cell_methods')) is not None:
            return CellMethod.parse(cmethod)
        return None

    # return an xarray dataset CF-Compliant
    def to_xarray(self) -> xr.Dataset:
        ds = self._dset  # we need to copy the ds metadata (and not the data) maybe .copy()
        return ds


class PointsField(Field, PointsFeature):
    """
    Field defined on a point domain.

    Parameters
    ----------
    name : str
        The name of a represented quantity.
    domain : points domain
        The domain construct over which a field is defined.
    data : array_like or quantity
        The values of a single physical variable.
    ancillary : dict_like, optional
        Ancillary variables and data of a field.
    properties : dict_like, optional
        The static metadata of the field
    encoding : dict_like, optional
        The encoding of a data variable.
    cell_method : str, default ''
        The cell method of a field if any.

    Attributes
    ----------
    domain : points domain
        The domain construct over which a field is defined.
    ancillary : dict or quantity
        Ancillary variables and data of a field.
    name : str
        The name of a represented quantity.
    data : array_like or quantity
        The values of a single physical variable.
    properties : dict
        The static metadata of the field
    encoding : dict
        The encoding of a data variable.
    cell_method : CellMethod or None
        The cell method of a field if any, otherwise None.
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
    :class:`geokube.core.domain.Points` :
        Point domain defined at scattered locations and times.
    :class:`geokube.core.field.ProfilesField` :
        Field defined on a profile domain
    :class:`geokube.core.field.GridField` :
        Field defined on a gridded domain.

    """

    __slots__ = ('_name',)
    _DOMAIN_CLS_ = Points

    def __init__(
        self,
        name: str,
        domain: Points,
        data: npt.ArrayLike | pint.Quantity | None = None,
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None,
        cell_method: str = ''
    ) -> None:
        n_pts = domain.number_of_points
        match data:
            case pint.Quantity():
                data_ = (
                    data
                    if isinstance(data.magnitude, _ARRAY_TYPES) else
                    pint.Quantity(np.asarray(data.magnitude), data.units)
                )
            case np.ndarray() | da.Array():
                # NOTE: The pattern arr * unit does not work when arr has
                # strings.
                data_ = pint.Quantity(data)
            case None:
                data_ = pint.Quantity(
                    np.full(shape=n_pts, fill_value=np.nan, dtype=np.float32)
                )
            case _:
                data_ = pint.Quantity(np.asarray(data))
        if data_.shape != (n_pts,):
            raise ValueError(
                "'data' must have one-dimensional values and the same size as "
                "the coordinates"
            )

        # TODO: Move the common part for all field types to `Field.__init__`,
        # together with the `_name` field.
        data_vars = {}
        attrs = properties if properties is not None else {}
        attrs['units'] = str(data_.units)
        # TODO: Check if domain/axis time has intervals when the cell
        # method exists and is different than points.
        # NOTE: Cell methods and intervals go together.
        if cell_method:
            attrs['cell_methods'] = cell_method
        data_vars[name] = xr.DataArray(
            data=data_.magnitude, dims=self._DIMS_, attrs=attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(
                    data=anc_data, dims=self._DIMS_
                )

        dset = domain._dset
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(data_vars)

        super().__init__(dset)
        names = self._get_var_names()
        assert len(names) == 1
        self._name = names.pop()


class ProfilesField(Field, ProfilesFeature):
    """
    Field defined on a profile domain.

    Parameters
    ----------
    name : str
        The name of a represented quantity.
    domain : points domain
        The domain construct over which a field is defined.
    data : array_like or quantity
        The values of a single physical variable.
    ancillary : dict_like, optional
        Ancillary variables and data of a field.
    properties : dict_like, optional
        The static metadata of the field
    encoding : dict_like, optional
        The encoding of a data variable.
    cell_method : str, default ''
        The cell method of a field if any.

    Attributes
    ----------
    domain : profiles domain
        The domain construct over which a field is defined.
    ancillary : dict or quantity
        Ancillary variables and data of a field.
    name : str
        The name of a represented quantity.
    data : array_like or quantity
        The values of a single physical variable.
    properties : dict
        The static metadata of the field
    encoding : dict
        The encoding of a data variable.
    cell_method : CellMethod or None
        The cell method of a field if any, otherwise None.
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
    :class:`geokube.core.domain.Profiles` :
        Profile domain defined along a vertical line at fixed locations.
    :class:`geokube.core.field.PointsField` :
        Field defined on a point domain.
    :class:`geokube.core.field.GridField` :
        Field defined on a gridded domain.

    Examples
    --------

    Creating a profiles field assumes instantiating at least a
    coordinate system, domain, and data.  It is sufficient to specify
    the data units for one profile:

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
    >>> prof_field = ProfilesField(
    ...     name='temperature',
    ...     data=[
    ...         [300, 350] * units.kelvin,
    ...         [280, 300, 320, 340, 360, 380],
    ...         [289, 295],
    ...         [295, 300, 305, 310, 315],
    ...         [298, 299, 300, 301, 302]
    ...     ],
    ...     domain=prof_domain
    ... )

    """

    __slots__ = ('_name',)
    _DOMAIN_CLS_ = Profiles

    def __init__(
        self,
        name: str,
        domain: Profiles,
        data: npt.ArrayLike | pint.Quantity | None = None,
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None,
        cell_method: str = ''
    ) -> None:
        n_prof, n_lev = domain.number_of_profiles, domain.number_of_levels
        data_shape = (n_prof, n_lev)
        if isinstance(data, Sequence):
            if len(data) != n_prof:
                raise ValueError(
                    "'data' does not contain the same number of profiles as "
                    "the coordinates do"
                )
            all_sizes, all_units, all_data = [], set(), []
            for data_item in data:
                all_sizes.append(len(data_item))
                if isinstance(data_item, pint.Quantity):
                    all_units.add(data_item.units)
                    all_data.append(data_item.magnitude)
                else:
                    all_data.append(data_item)
            if max(all_sizes) > n_lev:
                raise ValueError(
                    "'data' contains more levels than the coordinates do"
                )
            match len(all_units):
                case 0:
                    unit = None
                case 1:
                    unit = all_units.pop()
                case _:
                    # TODO: Consider supporting unit conversion in such cases.
                    raise ValueError("'data' has items with different units")
            data_vals = np.empty(shape=data_shape, dtype=np.float32)
            for i, (stop_idx, vals) in enumerate(zip(all_sizes, all_data)):
                if stop_idx == n_lev:
                    data_vals[i, :] = vals
                else:
                    data_vals[i, :stop_idx] = vals
                    data_vals[i, stop_idx:] = np.nan
            data_ = pint.Quantity(data_vals, unit)
        else:
            match data:
                case pint.Quantity():
                    data_ = (
                        data
                        if isinstance(data.magnitude, _ARRAY_TYPES) else
                        pint.Quantity(np.asarray(data.magnitude), data.units)
                    )
                case np.ndarray() | da.Array():
                    data_ = pint.Quantity(data)
                case None:
                    data_ = pint.Quantity(
                        np.full(
                            shape=data_shape,
                            fill_value=np.nan,
                            dtype=np.float32
                        )
                    )
                case _:
                    data_ = pint.Quantity(np.asarray(data))
            if data_.shape != data_shape:
                raise ValueError(
                    "'data' must be two-dimensional and have the same shape "
                    "as the coordinates"
                )

        # TODO: Move the common part for all field types to `Field.__init__`,
        # together with the `_name` field.
        data_vars = {}
        attrs = properties if properties is not None else {}
        attrs['units'] = str(data_.units)
        # TODO: Check if domain/axis time has intervals when the cell
        # method exists and is different than points.
        # NOTE: Cell methods and intervals go together.
        if cell_method:
            attrs['cell_methods'] = cell_method
        data_vars[name] = xr.DataArray(
            data=data_.magnitude, dims=self._DIMS_, attrs=attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=self._DIMS)

        dset = domain._dset
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(data_vars)

        super().__init__(dset)
        names = self._get_var_names()
        assert len(names) == 1
        self._name = names.pop()

    def as_points(self) -> PointsField:
        """
        Return points representation of the data and coordinates.

        Returns
        -------
        PointsField
            Points Field with the same data and coordinates.
        """
        return PointsField._from_xarray_dataset(_as_points_dataset(self))


class GridField(Field, GridFeature):
    """
    Field defined on a gridded domain.

    Parameters
    ----------
    name : str
        The name of a represented quantity.
    domain : grid domain
        The domain construct over which a field is defined.
    data : array_like or quantity
        The values of a single physical variable.
    ancillary : dict_like, optional
        Ancillary variables and data of a field.
    properties : dict_like, optional
        The static metadata of the field
    encoding : dict_like, optional
        The encoding of a data variable.
    cell_method : str, default ''
        The cell method of a field if any.

    Attributes
    ----------
    domain : grid domain
        The domain construct over which a field is defined.
    ancillary : dict or quantity
        Ancillary variables and data of a field.
    name : str
        The name of a represented quantity.
    data : array_like or quantity
        The values of a single physical variable.
    properties : dict
        The static metadata of the field
    encoding : dict
        The encoding of a data variable.
    cell_method : CellMethod or None
        The cell method of a field if any, otherwise None.
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
    regrid(target, method)
        Return the new field with the data that correspond to `target`.
    interpolate_(target, method, **kwargs)
        Return the new interpolated field onto the target coordinates.
    is_geodetic()
        Return a Boolean if a field has a geodetic CRS.
    as_geodetic()
        Return the transformed gridded field with the geodetic CRS.
    time_range(start, end)
        Return a subset defined within the time bounds.
    nearest_time(time)
        Return a subset with the nearest time values.
    latest()
        Return a subset with just the latest (largest) time value.
    resample(freq, operator, **kwargs)
        Return a field with resampled data and time coordinate.
    as_points()
        Return points representation of the data and coordinates.
    to_netcdf(path, **xarray_kwargs)
        Write the contents to a netCDF file.

    See Also
    --------
    :class:`geokube.core.domain.Grid` :
        Gridded domain defined at a spatial and temporal grid.
    :class:`geokube.core.field.PointsField` :
        Field defined on a point domain.
    :class:`geokube.core.field.ProfilesField` :
        Field defined on a profile domain

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
    >>> grid_field = GridField(
    ...     name='temperature',
    ...     domain=grid_domain,
    ...     data=np.arange(120).reshape(5, 4, 3, 2) * units.deg_C,
    ...     cell_method='time: maximum'
    ... )

    """

    __slots__ = ('_name',)
    _DOMAIN_CLS_ = Grid
    # NOTE: The default order of axes is assumed.

    def __init__(
        self,
        name: str,
        domain: Grid,
        data: npt.ArrayLike | pint.Quantity | None = None,
        ancillary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None,
        cell_method: str = ''
    ) -> None:

        match data:
            case pint.Quantity():
                data_ = (
                    data
                    if isinstance(data.magnitude, _ARRAY_TYPES) else
                    pint.Quantity(np.asarray(data.magnitude), data.units)
                )
            case np.ndarray() | da.Array():
                # NOTE: The pattern arr * unit does not work when arr has
                # stings.
                data_ = pint.Quantity(data)
            case None:
                data_ = None
            case _:
                data_ = pint.Quantity(np.asarray(data))

        #NOTE: THIS CODE CAN BE PUT IN ONE METHOD COMMON FOR ALL FIELDS! (MAYBE!)

        grid_mapping_attrs = domain.coord_system.spatial.crs.to_cf()
        grid_mapping_name = grid_mapping_attrs['grid_mapping_name']

        # TODO: Move the common part for all field types to `Field.__init__`,
        # together with the `_name` field.
        data_vars = {}
        field_attrs = properties if properties is not None else {} # TODO: attrs can contain both properties and CF attrs
        field_attrs |= {
            'units': str(data_.units), 'grid_mapping': grid_mapping_name
        }
        # TODO: Check if domain/axis time has intervals when the cell
        # method exists and is different than points.
        # NOTE: Cell methods and intervals go together.
        if cell_method:
            field_attrs['cell_methods'] = cell_method
        data_vars[name] = xr.DataArray(
            data=None if data_ is None else data_.magnitude,
            dims=domain._dset.dims,
            attrs=field_attrs
        )
        data_vars[name].encoding = encoding if encoding is not None else {}

        if ancillary is not None:
            ancillary_names = []
            for anc_name, anc_data in ancillary.items():
                data_vars[anc_name] = xr.DataArray(data=anc_data, dims=domain._dset.dims)
                data_vars[name].attrs['grid_mapping'] = grid_mapping_name
                ancillary_names.append(anc_name)

            data_vars[name].attrs['ancillary_variables'] = " ".join(ancillary_names)

        dset = domain._dset
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(data_vars)

        super().__init__(dset)
        names = self._get_var_names()
        assert len(names) == 1
        self._name = names.pop()

    def regrid(
        self, 
        target: GridField | Grid,
        method: str = 'bilinear'
    ) -> Self:
        """
        Return the new field with the data that correspond to `target`.

        This method regrids the field according to the specified target
        domain and regridding method.  The resulting field has the
        spatial coordinates from `target`, keeps other coordinates and
        attributes, and contains the regridded data.

        Parameters
        ----------
        target : grid domain or field
            Target grid for regridding.  If a grid domain is passed, it
            is used directly.  If a field is passed, only its domain is
            used.
        method : str, default 'bilinear'
            Regridding method

        Returns
        -------
        GridField
            Resulting field with the regridded data.

        Raises
        ------
        TypeError
            If `target` is not an instance of
            :class:`geokube.core.domain.Grid` or
            :class:`geokube.core.field.GridField`.
        """
        import xesmf as xe

        if not isinstance(target, Grid):
            if isinstance(target, GridField):
                target = target.domain
            else:
                raise TypeError(
                    "'target' must be an instance of Grid or GridField"
                )
        #
        # TODO: check if they have the same CRS
        # if source CRS and target CRS are different
        # first transform source CRS to target CRS
        # 
        # get spatial lat/lon coordinates
        # we should get all horizontal coordinates -> e.g. Projection, RotatedPole ...
        # 
        
        # NOTE: Maybe it is better to use `target._dset.coords` instead of
        # `target.coords` because the former keeps the attributes.
        # lat = target.coords[axis.latitude]
        # lon = target.coords[axis.longitude]
        target_coords = dict(target._dset.coords)
        lat = target_coords[axis.latitude].to_numpy()
        lon = target_coords[axis.longitude].to_numpy()

        ds_out = xr.Dataset({"lat": (["lat"], lat), "lon": (["lon"], lon)})

        # NOTE: before regridding we need to dequantify  
        # ds = self._dset.pint.dequantify()
        #
        # if we have ancillary data how they should be regridded?
        # for the moment we assume the same method for the field
        # TODO: maybe user should specify method for ancillary too!
        #
        # NOTE: The new underlying dataset does not need dequantification
        # because coordinates and data variables are already dequantified,
        # while the units are in the attributes.
        regridder = xe.Regridder(
            self._dset, ds_out, method, unmapped_to_nan=True
        )
        dset_reg = regridder(self._dset, keep_attrs=True)

        ancillary = {
            name: darr
            for name, darr in dset_reg.data_vars.items()
            if name != self.name
        }

        new_cs = CoordinateSystem(
            horizontal=target.coord_system.spatial.crs,
            elevation=self.coord_system.spatial.elevation,
            time=self.coord_system.time,
            user_axes=self.coord_system.user_axes
        )

        coords = {}
        for ax in new_cs.axes:
            if isinstance(ax, axis.Horizontal):
                coords[ax] = target_coords[ax]
            else:
                coords[ax] = self._dset.coords[ax]

        return GridField(
            name=self.name,
            data=dset_reg[self.name],  # .pint.quantify(),
            domain=Grid(coords=coords, coord_system=new_cs),
            ancillary=ancillary,
            properties=self.properties,
            encoding=self.encoding
        )

    def interpolate(
        self, target: Domain | Field, method: str = 'nearest', **xarray_kwargs
    ) -> Field:
        """
        Return the new interpolated field onto the target coordinates.

        The resulting field has the spatial coordinates from `target`,
        keeps other coordinates and attributes, and contains the
        interpolated data.

        Parameters
        ----------
        target : domain or field
            Target that contains the horizontal coordinates for
            interpolation.  If a domain is passed, it is used directly.
            If a field is passed, only the spatial part of its domain is
            used.
        method : str, default 'nearest'
            Interpolation method.
        **xarray_kwargs : dict, optional
            Additional arguments passed to
            :meth:`xarray.Dataset.interp`.

        Returns
        -------
        GridField
            Resulting field with the interpolated data.

        Raises
        ------
        TypeError
            If `target` is not an instance of
            :class:`geokube.core.domain.Domain` or
            :class:`geokube.core.field.Field`.
        """
        # spatial interpolation
        # dset = self._dset.pint.dequantify() - it is needed since units are not kept!
        # dset = self._dset.pint.dequantify()
        dset = self._dset

        match target:
            case Domain():
                pass
            case Field():
                target = target.domain
            case _:
                raise TypeError("'target' must be a domain or field")

        if self.crs._crs != target.crs._crs:
            # we need to transform the target domain to the same crs of the domain to 
            # be interpolated and we perform the interpolation on the horizontal axes
            #
            target_x, target_y = target.spatial_transform_to(self.crs)
            kwargs.setdefault('fill_value', None)
            target_dims = target.crs.dim_axes
            dims = (axis.latitude, axis.longitude)
            target_ = xr.Dataset(
                data_vars={
                    self.crs.dim_X_axis: xr.DataArray(data=target_x, dims=dims),
                    self.crs.dim_Y_axis: xr.DataArray(data=target_y, dims=dims)
                },
                coords={axis: target.coords[axis] for axis in dims}
            )
            dset = dset.drop(labels=(axis.latitude, axis.longitude))
            ds = dset.interp(coords=target_, method=method, kwargs=kwargs)
            # TODO: workaround - remove when adding attributes to field for ancillary data.
            ds = ds.drop(labels=(self.crs.dim_X_axis, self.crs.dim_Y_axis))
        else:
            coords = dict(dset.coords)
            del coords[dset.attrs['grid_mapping']]
            coords = {axis: coord.to_numpy() for axis, coord in coords.items()}
            target_coords = {
                axis: coord.to_numpy()
                for axis, coord in target._dset.coords.items()
                if axis in target.crs.axes
            }
            target_coords = coords | target_coords
            kwargs.setdefault('fill_value', 'extrapolate')
            ds = dset.interp(
                coords=target_coords, method=method, kwargs=kwargs
            )

        # ds contains the data interpolated on the new domain
        ancillary = {}
        for v in ds.data_vars:
            if v != self.name:
                ancillary[v] = ds[v].pint.quantify()

        cs = CoordinateSystem(
            horizontal=target.crs,
            elevation=self.coord_system.spatial.elevation,
            time=self.coord_system.time,
            user_axes=self.coord_system.user_axes
        )

        coords = {}
        for ax in cs.axes:
            if isinstance(ax, axis.Horizontal):
                coords[ax] = target.coords[ax]
            else:
                coords[ax] = self.coords[ax]

        return GridField(
            name=self.name,
            data=ds[self.name].pint.quantify(),
            domain=Grid(coords=coords, coord_system=cs),
            ancillary=ancillary,
            properties=self.properties,
            encoding=self.encoding
        )

    def is_geodetic(self) -> bool:
        """
        Return a Boolean if a field has a geodetic CRS.

        Returns
        -------
        bool
            True if a field has a geodetic CRS and False otherwise.

        """
        return isinstance(self.crs, Geodetic) 

    def as_geodetic(self) -> Field:
        """
        Return the transformed gridded field with the geodetic CRS.

        Returns
        -------
        Field
            New gridded field with the geodetic CRS.

        """
        if self.is_geodetic():
            return self
        
        return self.interpolate(
            target=self.domain.as_geodetic(), # this has a semantic -> domain geodetic grid can be different from the original
            method="nearest"
        )

    def as_points(self) -> PointsField:
        """
        Return points representation of the data and coordinates.

        Returns
        -------
        PointsField
            Points field with the same data and coordinates.
        """
        return PointsField._from_xarray_dataset(_as_points_dataset(self))

    def resample(
        self, freq: str, operator: Callable | str = 'nanmean', **kwargs
    ) -> Self:
        """
        Return a field with resampled data and time coordinate.

        This method performs frequency conversion of a field.  The field
        must have the time axis and coordinate.  Time can be represented
        either as instances or intervals.

        Parameters
        ----------
        freq : str
            Target resample frequency.
        operator : callable or str, default 'nanmean'
            Operation to be applied on the data during resampling.
        **kwargs : dict, optional
            Additional arguments passed to :mod:`pandas` objects.

        Returns
        -------
        GridField
            A field defined on the gridded domain obatained as a result
            of resampling.

        Raises
        ------
        ValueError
            * If the time values are intervals and `keargs` are passed.
            * If the time values are intervals of varying durations.
            * If `freq` does not correspond to the interval durations.
        NotImplementedError
            * If the time value array contains anything other than
              datetime or interval data.

        """
        dset = self._dset
        data = dset[self.name].data
        time_idx = dset.xindexes[axis.time].index.index
        n_time = time_idx.size

        match time_idx:
            case pd.DatetimeIndex():
                time = pd.Series(data=np.arange(n_time), index=time_idx)
            case pd.IntervalIndex():
                # TODO: Check if this code has issue with ERA5 (rotated pole),
                #  when `freq`is e.g. `'5H'`.
                if kwargs:
                    raise ValueError(
                        "'kwargs' are not allowed for interval indices"
                    )
                left_bnd, right_bnd = time_idx.left, time_idx.right
                src_freqs = pd.unique(right_bnd - left_bnd)
                if src_freqs.size == 1:
                    src_freq = abs(src_freqs[0])
                else:
                    raise ValueError(
                        "'time_idx' must have equal differences for resampling"
                    )
                src_diff = n_time * src_freq
                dst_freq = pd.to_timedelta(freq).to_timedelta64()
                ratio = float(dst_freq / src_freq)
                if ratio.is_integer() and ratio >= 1 and dst_freq <= src_diff:
                    time = pd.Series(data=np.arange(n_time), index=left_bnd)
                else:
                    raise ValueError(
                        "'freq' does not correspond to the interval durations"
                    )
            case _:
                raise NotImplementedError(
                    f"'time_idx' has the type {type(time_idx).__name__}, "
                    "which is not supported; it must be an instance of "
                    "'DatetimeIndex' or 'IntervalIndex'"
                )

        left_time_res = time.resample(
            rule=freq, label='left', origin='start', **kwargs
        )
        left_gr = left_time_res.grouper
        # TODO: Try to avoid the second call to `time.resample` and find the
        # right bound another way, e.g. with `time_res.freq.delta.to_numpy()`.
        right_time_res = time.resample(
            rule=freq, label='right', origin='start', **kwargs
        )
        right_gr = right_time_res.grouper
        new_time = pd.IntervalIndex.from_arrays(
            left=left_gr.binlabels, right=right_gr.binlabels, closed='both'
        )
        time_axis_idx = self.dim_axes.index(axis.time)
        new_shape = list(data.shape)
        new_shape[time_axis_idx] = new_time.size
        # NOTE: For NumPy arrays, it is possible to use `numpy.split`, but it
        # seems that Dask does not have such a function.
        bins = left_gr.bins
        slices = (
            [slice(bins[0])]
            + [slice(bins[i], bins[i + 1]) for i in range(bins.size - 1)]
        )
        arr_lib = da if isinstance(data, da.Array) else np
        func = operator if callable(operator) else getattr(arr_lib, operator)
        # FIXME: If `data.dtype` is integral, we want a floating-point result
        # for some operators like `mean` or `median` and integral for others
        # like `min` or `max`.
        new_data = arr_lib.empty(shape=new_shape, dtype=data.dtype)
        whole_axis = (slice(None),)
        for i, s in enumerate(slices):
            idx_before = whole_axis * time_axis_idx
            idx_after = whole_axis * (len(new_shape) - time_axis_idx - 1)
            i_ = idx_before + (i,) + idx_after
            s_ = idx_before + (s,) + idx_after
            # TODO: Test optimization with something like:
            # `func(data[s_], axis=time_axis_idx, out=new_data[i_])`.
            new_data[i_] = func(data[s_], axis=time_axis_idx)

        domain = self.domain
        new_coords = domain.coords.copy()
        new_coords[axis.time] = new_time

        return type(self)(
            name=self.name,
            domain=type(domain)(new_coords, domain.coord_system),
            data=pint.Quantity(new_data, dset[self.name].attrs.get('units')),
            ancillary=self.ancillary,
            properties=self.properties,
            encoding=self.encoding
        )
