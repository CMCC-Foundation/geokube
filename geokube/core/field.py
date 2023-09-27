from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Number
from typing import Any, Self
from warnings import warn

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import xarray as xr

from . import axis, indexes
from .crs import Geodetic
from .domain import Grid, Points, Profile
from .indexers import get_array_indexer, get_indexer
from .points import to_points_dict
from .quantity import get_magnitude


_ARRAY_TYPES = (np.ndarray, da.Array)


class PointsField:
    __slots__ = (
        '__name',
        '__data',
        '__anciliary',
        '__domain',
        '__properties',
        '__encoding'
    )

    _DOMAIN_TYPE = Points

    def __init__(
        self,
        name: str,
        domain: Points,
        data: npt.ArrayLike | pint.Quantity | None = None,
        anciliary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        self.__name = str(name)
        domain_type = self._DOMAIN_TYPE
        if not isinstance(domain, domain_type):
            raise TypeError(
                f"'domain' must be an instance of '{domain_type.__name__}'"
            )
        self.__anciliary = dict(anciliary) if anciliary else {}
        self.__domain = domain
        self.__properties = dict(properties) if properties else {}
        self.__encoding = dict(encoding) if encoding else {}

        n_pts = domain.number_of_points
        match data:
            # case pint.Quantity() if isinstance(data.magnitude, _ARRAY_TYPES):
            #     data_ = data
            # case pint.Quantity():
            #     data_ = pint.Quantity(np.asarray(data.magnitude), data.units)
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
        dset = xr.Dataset(
            data_vars={self.__name: (('_points',), data_)},
            coords=domain._coords
        )
        coord_system = domain.coord_system
        hor_axes = set(coord_system.spatial.crs.AXES)
        for axis_ in coord_system.axes:
            if axis_ not in hor_axes:
                dset = dset.set_xindex(axis_, indexes.OneDimIndex)
        dset = dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )
        self.__data = dset

    @property
    def name(self) -> str:
        return self.__name

    @property
    def domain(self) -> Points:
        return self.__domain

    @property
    def anciliary(self) -> dict:
        return self.__anciliary

    @property
    def properties(self) -> dict:
        return self.__properties

    @property
    def encoding(self) -> dict:
        return self.__encoding

    @property
    def _data(self) -> xr.Dataset:
        return self.__data

    @property
    def data(self) -> pint.Quantity:
        return self.__data[self.__name].data

    # TODO: Consider making this a method of a future base class.
    # TODO: Replace `Any` with an appropriate `TypeVar` instance that
    # represents all eligible types.
    def _new_field(
        self, new_data: xr.Dataset, result_type: type[Any] | None = None
    ) -> Any:
        field_type = type(self) if result_type is None else result_type
        domain_type = field_type._DOMAIN_TYPE
        name = self.__name
        return field_type(
            name=name,
            domain=domain_type(
                coords={
                    axis_: coord.data
                    for axis_, coord in new_data.coords.items()
                },
                coord_system=self.__domain.coord_system
            ),
            data=new_data[name].data
        )

    # Spatial operations ------------------------------------------------------

    def bounding_box(
        self,
        south: Number | None = None,
        north: Number | None = None,
        west: Number | None = None,
        east: Number | None = None,
        bottom: Number | None = None,
        top: Number | None = None
    ) -> Self:
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        new_data = self.__data.sel(h_idx)
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            new_data = self._new_field(new_data)._data
            new_data = new_data.sel(v_idx)
        return self._new_field(new_data)

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        new_data = self.__data.sel(idx, method='nearest', tolerance=np.inf)
        return self._new_field(new_data)

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.vertical: elevation}
        new_data = self.__data.sel(idx, method='nearest', tolerance=np.inf)
        return self._new_field(new_data)

    # Temporal operations -----------------------------------------------------

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        idx = {axis.time: slice(start, end)}
        new_data = self.__data.sel(idx)
        return self._new_field(new_data)

    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        new_data = self.__data.sel(idx, method='nearest', tolerance=None)
        return self._new_field(new_data)


class ProfileField:
    __slots__ = (
        '__name',
        '__data',
        '__anciliary',
        '__domain',
        '__properties',
        '__encoding'
    )

    _DOMAIN_TYPE = Profile

    def __init__(
        self,
        name: str,
        domain: Profile,
        data: npt.ArrayLike | pint.Quantity | None = None,
        anciliary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        self.__name = str(name)
        domain_type = self._DOMAIN_TYPE
        if not isinstance(domain, domain_type):
            raise TypeError(
                f"'domain' must be an instance of '{domain_type.__name__}'"
            )
        self.__anciliary = dict(anciliary) if anciliary else {}
        self.__domain = domain
        self.__properties = dict(properties) if properties else {}
        self.__encoding = dict(encoding) if encoding else {}

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
        dset = xr.Dataset(
            data_vars={self.__name: (('_profiles', '_levels'), data_)},
            coords=domain._coords
        )
        coord_system = domain.coord_system
        spat_axes = set(coord_system.spatial.axes)
        for axis_ in coord_system.axes:
            if axis_ not in spat_axes:
                dset = dset.set_xindex(axis_, indexes.OneDimIndex)
        dset = dset.set_xindex(
            axis.vertical,
            indexes.TwoDimVertProfileIndex,
            data=dset[self.__name],
            name=self.__name
        )
        dset = dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )
        self.__data = dset

    @property
    def name(self) -> str:
        return self.__name

    @property
    def domain(self) -> Profile:
        return self.__domain

    @property
    def anciliary(self) -> dict:
        return self.__anciliary

    @property
    def properties(self) -> dict:
        return self.__properties

    @property
    def encoding(self) -> dict:
        return self.__encoding

    @property
    def _data(self) -> xr.Dataset:
        return self.__data

    @property
    def data(self) -> pint.Quantity:
        return self.__data[self.__name].data

    # TODO: Consider making this a method of a future base class.
    # TODO: Replace `Any` with an appropriate `TypeVar` instance that
    # represents all eligible types.
    def _new_field(
        self, new_data: xr.Dataset, result_type: type[Any] | None = None
    ) -> Any:
        field_type = type(self) if result_type is None else result_type
        domain_type = field_type._DOMAIN_TYPE
        name = self.__name
        return field_type(
            name=name,
            domain=domain_type(
                coords={
                    axis_: coord.data
                    for axis_, coord in new_data.coords.items()
                },
                coord_system=self.__domain.coord_system
            ),
            data=new_data[name].data
        )

    # Spatial operations ------------------------------------------------------

    def bounding_box(
        self,
        south: Number | None = None,
        north: Number | None = None,
        west: Number | None = None,
        east: Number | None = None,
        bottom: Number | None = None,
        top: Number | None = None
    ) -> Self:
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        new_data = self.__data.sel(h_idx)
        if not (bottom is None and top is None):
            # TODO: Try to move this functionality to
            # `indexes.TwoDimHorPointsIndex.sel`.
            warn(
                "'bounding_box' loads in memory and makes a copy of the data "
                "and vertical coordinate when 'bottom' or 'top' is not 'None'"
            )
            v_slice = slice(bottom, top)
            v_idx = {axis.vertical: v_slice}
            new_data = self._new_field(new_data)._data
            new_data = new_data.sel(v_idx)
            vert = new_data[axis.vertical]
            vert_dims = vert.dims
            vert_data = vert.data
            vert_mag, vert_units = vert_data.magnitude, vert_data.units
            data = new_data[self.__name].data
            mask = get_indexer(
                [vert_mag], [get_magnitude(v_slice, vert_units)]
            )[0]
            masked_vert = np.where(mask, vert_mag, np.nan)
            vert_ = pint.Quantity(masked_vert, vert_units)
            new_data[axis.vertical] = xr.Variable(dims=vert_dims, data=vert_)
            masked_data = np.where(mask, data.magnitude, np.nan)
            data_ = pint.Quantity(masked_data, data.units)
            new_data[self.__name] = xr.Variable(dims=vert_dims, data=data_)
        return self._new_field(new_data)

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        new_data = self.__data.sel(idx, method='nearest', tolerance=np.inf)
        return self._new_field(new_data)

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        # TODO: Try to move this functionality to
        # `indexes.TwoDimHorPointsIndex.sel`.
        dset = self.__data
        data_qty = dset[self.__name].data
        data_mag = data_qty.magnitude
        vert = dset[axis.vertical]
        vert_qty = vert.data
        vert_mag, vert_units = vert_qty.magnitude, vert_qty.units
        dims = vert.dims
        profile_idx = dims.index('_profiles')
        n_profiles = vert_mag.shape[profile_idx]
        kwa = {
            'y_data': [get_magnitude(elevation, vert_units)],
            'return_all': False,
            'method': 'nearest',
            'tolerance': np.inf
        }
        order = slice(None, None, -1 if profile_idx else 1)
        shape = (n_profiles, len(elevation))[order]
        new_data_mag = np.empty(shape=shape, dtype=data_mag.dtype)
        new_vert_mag = np.empty(shape=shape, dtype=vert_mag.dtype)
        # TODO: Try to implement this in a more efficient way.
        for p_idx in range(n_profiles):
            all_l_idx = (p_idx, slice(None))[order]
            l_idx = get_indexer([vert_mag[all_l_idx]], **kwa)
            p_l_idx = (p_idx, l_idx)[order]
            new_data_mag[all_l_idx] = data_mag[p_l_idx]
            new_vert_mag[all_l_idx] = vert_mag[p_l_idx]
        domain = self.__domain
        new_coords = self.__domain.coordinates().copy()
        new_coords[axis.vertical] = pint.Quantity(new_vert_mag, vert_units)
        return type(self)(
            name=self.__name,
            domain=type(domain)(new_coords, domain.coord_system),
            data=pint.Quantity(new_data_mag, data_qty.units)
        )

    # Temporal operations -----------------------------------------------------

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        idx = {axis.time: slice(start, end)}
        new_data = self.__data.sel(idx)
        return self._new_field(new_data)

    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        new_data = self.__data.sel(idx, method='nearest', tolerance=None)
        return self._new_field(new_data)


class GridField:
    # NOTE: The default order of axes is assumed.

    __slots__ = (
        '__name',
        '__data',
        '__dim_axes',
        '__anciliary',
        '__domain',
        '__properties',
        '__encoding'
    )

    _DOMAIN_TYPE = Grid

    def __init__(
        self,
        name: str,
        domain: Grid,
        data: npt.ArrayLike | pint.Quantity | None = None,
        dim_axes: Sequence[axis.Axis] | None = None,
        anciliary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        self.__name = str(name)
        domain_type = self._DOMAIN_TYPE
        if not isinstance(domain, domain_type):
            raise TypeError(
                f"'domain' must be an instance of '{domain_type.__name__}'"
            )
        self.__anciliary = dict(anciliary) if anciliary else {}
        self.__domain = domain
        self.__properties = dict(properties) if properties else {}
        self.__encoding = dict(encoding) if encoding else {}

        coord_system = domain.coord_system
        coords = domain._coords
        crs = coord_system.spatial.crs
        dim_axes_: tuple[axis.Axis, ...]
        aux_axes: tuple[axis.Axis, ...]
        if dim_axes is None:
            if isinstance(crs, Geodetic):
                dim_axes_, aux_axes = coord_system.axes, ()
            else:
                default_axes = coord_system.axes
                aux_hor_axes = {axis.latitude, axis.longitude}
                dim_axes_tmp, aux_axes_tmp = [], []
                for axis_ in default_axes:
                    if axis_ in aux_hor_axes:
                        aux_axes_tmp.append(axis_)
                    else:
                        dim_axes_tmp.append(axis_)
                dim_axes_, aux_axes = tuple(dim_axes_tmp), tuple(aux_axes_tmp)
        else:
            dim_axes_ = tuple(dim_axes)
            aux_axes = tuple(axis for axis in coords if axis not in dim_axes)
        self.__dim_axes = dim_axes_

        match data:
            # case pint.Quantity() if isinstance(data.magnitude, _ARRAY_TYPES):
            #     data_ = data
            # case pint.Quantity():
            #     data_ = pint.Quantity(np.asarray(data.magnitude), data.units)
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

        dset = xr.Dataset(
            data_vars={
                self.__name: (tuple(f'_{axis}' for axis in dim_axes_), data_)
            },
            coords=domain._coords
        )
        for axis_ in dim_axes_:
            dset = dset.set_xindex(axis_, indexes.OneDimPandasIndex)
        if {axis.latitude, axis.longitude} <= set(aux_axes):
            dset = dset.set_xindex(aux_axes, indexes.TwoDimHorGridIndex)
        self.__data = dset

    @property
    def name(self) -> str:
        return self.__name

    @property
    def domain(self) -> Grid:
        return self.__domain

    @property
    def anciliary(self) -> dict:
        return self.__anciliary

    @property
    def properties(self) -> dict:
        return self.__properties

    @property
    def encoding(self) -> dict:
        return self.__encoding

    @property
    def _data(self) -> xr.Dataset:
        return self.__data

    @property
    def data(self) -> pint.Quantity:
        return self.__data[self.__name].data

    # TODO: Consider making this a method of a future base class.
    # TODO: Replace `Any` with an appropriate `TypeVar` instance that
    # represents all eligible types.
    def _new_field(
        self, new_data: xr.Dataset, result_type: type[Any] | None = None
    ) -> Any:
        field_type = type(self) if result_type is None else result_type
        domain_type = field_type._DOMAIN_TYPE
        name = self.__name
        return field_type(
            name=name,
            domain=domain_type(
                coords={
                    axis_: coord.data
                    for axis_, coord in new_data.coords.items()
                },
                coord_system=self.__domain.coord_system
            ),
            data=new_data[name].data,
            dim_axes=self.__dim_axes
        )

    # Spatial operations ------------------------------------------------------

    def bounding_box(
        self,
        south: Number | None = None,
        north: Number | None = None,
        west: Number | None = None,
        east: Number | None = None,
        bottom: Number | None = None,
        top: Number | None = None
    ) -> Self:
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        new_data = self.__data.sel(h_idx)
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            new_data = self._new_field(new_data)._data
            new_data = new_data.sel(v_idx)
        return self._new_field(new_data)

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> PointsField:  # Self:
        # NOTE: This code works with geodetic grids and returns the Cartesian
        # product.
        # idx = {axis.latitude: latitude, axis.longitude: longitude}
        # new_data = self.__data.sel(idx, method='nearest', tolerance=np.inf)
        # return self._new_field(new_data)

        # NOTE: This code works with all tested grids and returns the nearest
        # points.
        # Preparing data, labels, units, and dimensions.
        name = self.__name
        coord_system = self.__domain.coord_system
        dset = self.__data
        lat = dset[axis.latitude]
        lat_data = lat.data
        lat_vals = lat_data.magnitude
        lon = dset[axis.longitude]
        lon_data = lon.data
        lon_vals = lon_data.magnitude

        lat_labels = get_magnitude(latitude, lat_data.units)
        lon_labels = get_magnitude(longitude, lon_data.units)

        if isinstance(coord_system.spatial.crs, Geodetic):
            lat_vals, lon_vals = np.meshgrid(lat_vals, lon_vals, indexing='ij')
            dims = ('_latitude', '_longitude')
        else:
            all_dims = {lat.dims, lon.dims}
            if len(all_dims) != 1:
                raise ValueError(
                    "'dset' must contain latitude and longitude with the same"
                    "dimensions for rotated geodetic and projection grids"
                )
            dims = all_dims.pop()

        # Calculating indexers and subsetting.
        idx = get_array_indexer(
            [lat_vals, lon_vals],
            [lat_labels, lon_labels],
            method='nearest',
            tolerance=np.inf,
            return_all=False
        )
        pts_dim = ('_points',)
        pts_idx = [(pts_dim, dim_idx) for dim_idx in idx]
        result_idx = dict(zip(dims, pts_idx))
        dset = dset.isel(indexers=result_idx)

        # Creating the resulting points field.
        new_coords = to_points_dict(name=name, dset=dset)
        del new_coords['points']
        new_data = new_coords.pop(name)

        return PointsField(
            name=name,
            domain=Points(coords=new_coords, coord_system=coord_system),
            data=new_data
        )


    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.vertical: elevation}
        new_data = self.__data.sel(idx, method='nearest', tolerance=np.inf)
        return self._new_field(new_data)

    # Temporal operations -----------------------------------------------------

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        idx = {axis.time: slice(start, end)}
        new_data = self.__data.sel(idx)
        return self._new_field(new_data)

    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        new_data = self.__data.sel(idx, method='nearest', tolerance=None)
        return self._new_field(new_data)
