from collections.abc import Mapping
from datetime import date, datetime
from itertools import chain
from numbers import Number
from typing import Self
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import pint_xarray
import pyarrow as pa
import xarray as xr

from . import axis, indexes
from .coord_system import CoordinateSystem
from .crs import Geodetic
from .indexers import get_array_indexer, get_indexer
from .points import to_points_dict
from .quantity import get_magnitude


# class FeatureMixin():

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


class Feature:
    __slots__ = ('_dset', '_coord_system')

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, pint.Quantity | xr.DataArray]
            | xr.core.coordinates.DatasetCoordinates
        ),
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray] | None = None,
        attrs: Mapping | None = None
    ) -> None:
        if not isinstance(coord_system, CoordinateSystem):
            raise TypeError(
                "'coord_system' must be an instance of 'CoordinateSystem'"
            )
        self._coord_system = coord_system
        self._dset = xr.Dataset(
            data_vars=data_vars, coords=coords, attrs=attrs
        )

    @classmethod
    def _from_xrdset(
        cls, dset: xr.Dataset, coord_system: CoordinateSystem
    ) -> Self:
        return cls(
            data_vars=dset.data_vars,
            coords=dset.coords,
            attrs=dset.attrs,
            coord_system=coord_system
        )

    #TODO: Implement __getitem__ ??

    @property
    def coords(self) -> dict[axis.Axis, pint.Quantity]:
        return {
            axis_: self._dset[axis_].pint.quantify().data
            for axis_ in self.coord_system.axes
        }

    @property
    def coord_system(self):
        return self._coord_system

    # CF Methods 
    @property  # return dimensional axes
    def dim_axes(self) -> tuple[axis.Axis, ...]:
        return self.coord_system.dim_axes

    @property
    def dim_coords(self) -> dict[axis.Axis, pint.Quantity]:
        return {ax: self._dset[ax].data for ax in self.coord_system.dim_axes}

    @property  # return auxiliary axes
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.coord_system.aux_axes

    @property
    def aux_coords(self) -> dict[axis.Axis, pint.Quantity]:
        return {ax: self._dset[ax].data for ax in self.coord_system.aux_axes}

    # spatial operations
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
        dset = self._dset.sel(h_idx)
        obj = type(self)._from_xrdset(dset, self.coord_system)
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            dset = obj._dset.sel(v_idx)
            obj = type(self)._from_xrdset(dset, self.coord_system)
        return obj
        # return type(self)._from_xrdset(dset, self.coord_system)

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        dset = self._dset.sel(idx, method='nearest', tolerance=np.inf)
        return type(self)._from_xrdset(dset, self.coord_system)

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.vertical: elevation}
        dset = self._dset.sel(idx, method='nearest', tolerance=np.inf)
        return self._from_xrdset(dset, self.coord_system)

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        idx = {axis.time: slice(start, end)}
        dset = self._dset.sel(idx)
        return self._from_xrdset(dset, self.coord_system)

    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        dset = self._dset.sel(idx, method='nearest', tolerance=None)
        return self._from_xrdset(dset, self.coord_system)

    def latest(self) -> Self:
        if axis.time not in self._dset.coords:
            raise NotImplementedError()
        latest = self._dset[axis.time].max().astype(str).item().magnitude
        idx = {axis.time: slice(latest, latest)}
        dset = self._dset.sel(idx)
        return self._from_xrdset(dset, self.coord_system)


class PointsFeature(Feature):
    __slots__ = ('_n_points',)
    _DIMS_ = ('_points',)

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, pint.Quantity | xr.DataArray]
            | xr.core.coordinates.DatasetCoordinates
        ),
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray] | None = None,
        attrs: Mapping | None = None
    ) -> None:
        match coords:
            case Mapping():
                res_coords = {
                    axis_: (
                        coord
                        if isinstance(coord, xr.DataArray) else
                        xr.DataArray(data=coord, dims=self._DIMS_)
                    )
                    for axis_, coord in coords.items()
                }
            case xr.core.coordinates.DatasetCoordinates():
                res_coords = coords
            case _:
                raise TypeError(
                    "'coords' can be a mapping or coordinates object"
                )

        match data_vars:
            case Mapping():
                res_data_vars = {
                    str(name): (
                        var
                        if isinstance(var, xr.DataArray) else
                        xr.DataArray(data=var, dims=self._DIMS_)
                    )
                    for name, var in data_vars.items()
                }
                res_vals = chain(res_coords.values(), res_data_vars.values())
            case None:
                res_data_vars = None
                res_vals = res_coords.values()
            case _:
                raise TypeError("'data_vars' can be a mapping or 'None'")

        super().__init__(
            coords=res_coords,
            coord_system=coord_system,
            data_vars=res_data_vars,
            attrs=attrs
        )

        n_pts = {val.size for val in res_vals}
        if len(n_pts) != 1:
            raise ValueError(
                "'coords' and 'data_vars' must have values of equal sizes"
            )
        self._n_points = n_pts.pop()

        hor_axes = set(coord_system.spatial.crs.axes)
        for axis_ in coord_system.axes:
            if axis_ not in hor_axes:
                self._dset = self._dset.set_xindex(axis_, indexes.OneDimIndex)
        self._dset = self._dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )

    @property
    def number_of_points(self) -> int:
        return self._n_points


class ProfilesFeature(Feature):
    _DIMS_ = ('_profiles', '_levels')
    __slots__ = ('_n_profiles', '_n_levels')

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, pint.Quantity | xr.DataArray]
            | xr.core.coordinates.DatasetCoordinates
        ),
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray] | None = None,
        attrs: Mapping | None = None
    ) -> None:
        match coords:
            case Mapping():
                res_coords = {}
                for axis_, coord in coords.items():
                    if isinstance(coord, xr.DataArray):
                        res_coords[axis_] = coord
                    else:
                        dims = (
                            self._DIMS_
                            if axis_ is axis.vertical else
                            ('_profiles',)
                        )
                        res_coords[axis_] = xr.DataArray(data=coord, dims=dims)
            case xr.core.coordinates.DatasetCoordinates():
                res_coords = coords
            case _:
                raise TypeError(
                    "'coords' can be a mapping or coordinates object"
                )
        self._n_profiles, self._n_levels = res_coords[axis.vertical].shape

        super().__init__(
            data_vars=data_vars,
            coords=res_coords,
            coord_system=coord_system,
            attrs=attrs
        )

        spat_axes = set(coord_system.spatial.axes)
        for axis_ in coord_system.axes:
            if axis_ not in spat_axes:
                self._dset = self._dset.set_xindex(axis_, indexes.OneDimIndex)
        self._dset = self._dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )
        self._dset = self._dset.set_xindex(
            axis.vertical, indexes.TwoDimVertProfileIndex
        )

    @property
    def number_of_profiles(self) -> int:
        return self._n_profiles

    @property
    def number_of_levels(self) -> int:
        return self._n_levels

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
        new_data = self._dset.sel(h_idx)
        new_obj = self._from_xrdset(new_data, self._coord_system)

        if not (bottom is None and top is None):
            # TODO: Try to move this functionality to
            # `indexes.TwoDimHorPointsIndex.sel`.
            warn(
                "'bounding_box' loads in memory and makes a copy of the data "
                "and vertical coordinate when 'bottom' or 'top' is not 'None'"
            )
            v_slice = slice(bottom, top)
            v_idx = {axis.vertical: v_slice}
            new_data = new_obj._dset.sel(v_idx)
            vert = new_data[axis.vertical]
            vert_dims = vert.dims
            vert_data = vert.data
            vert_mag, vert_units = vert_data.magnitude, vert_data.units
            mask = get_indexer(
                [vert_mag], [get_magnitude(v_slice, vert_units)]
            )[0]
            masked_vert = np.where(mask, vert_mag, np.nan)
            vert_ = pint.Quantity(masked_vert, vert_units)
            new_data[axis.vertical] = xr.Variable(dims=vert_dims, data=vert_)
            for name, darr in new_data.data_vars.items():
                data = darr.data
                masked_data = np.where(mask, data.magnitude, np.nan)
                data_ = pint.Quantity(masked_data, data.units)
                new_data[name] = xr.Variable(dims=vert_dims, data=data_)
            new_obj = self._from_xrdset(new_data, self._coord_system)
        return new_obj

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        # TODO: Try to move this functionality to
        # `indexes.TwoDimHorPointsIndex.sel`.
        dset = self._dset
        new_coords = {
            axis_: coord.variable for axis_, coord in dset.coords.items()
        }
        vert_axis = (new_coords.keys() & {axis.vertical}).pop()
        vert = dset[vert_axis]
        vert_qty = vert.data
        vert_mag, vert_units = vert_qty.magnitude, vert_qty.units
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
            dims=vert.dims, data=pint.Quantity(new_vert_mag, vert_units)
        )

        new_data_vars = {}
        for name, darr in dset.data_vars.items():
            data_qty = darr.data
            data_mag = data_qty.magnitude
            new_data_mag = np.empty(shape=shape, dtype=data_mag.dtype)
            # TODO: Try to implement this in a more efficient way.
            for profile_idx in range(n_profiles):
                level_idx = level_indices[profile_idx]
                new_data_mag[profile_idx, :] = vert_mag[profile_idx, level_idx]
            new_data_vars[name] = xr.DataArray(
                data=pint.Quantity(new_data_mag, data_qty.units),
                dims=self._DIMS_,
                coords=new_coords,
                attrs=darr.attrs
            )

        new_dset = xr.Dataset(
            data_vars=new_data_vars, coords=new_coords, attrs=dset.attrs
        )
        new_obj = self._from_xrdset(new_dset, self._coord_system)
        return new_obj


class GridFeature(Feature):
    __slots__ = ('_DIMS_',)

    def __init__(
        self,
        coords: (
            Mapping[axis.Axis, pint.Quantity | xr.DataArray]
            | xr.core.coordinates.DatasetCoordinates
        ),
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray] | None = None,
        attrs: Mapping | None = None
    ) -> None:
        crs = coord_system.spatial.crs
        self._DIMS_ = coord_system.dim_axes

        match coords:
            case Mapping():
                res_coords = {}
                for axis_, coord in coords.items():
                    if isinstance(coord, xr.DataArray):
                        res_coords[axis_] = coord
                    else:
                        dims: tuple[axis.Axis, ...]
                        if axis_ in self._DIMS_:
                            # Dimension coordinates.
                            match coord.ndim:
                                case 0:
                                    dims = ()
                                case 1:
                                    dims = (axis_,)
                                case _:
                                    raise ValueError(
                                        "'coords' have a dimension axis "
                                        f"{axis_} that has multi-dimensional "
                                        "values"
                                    )
                        else:
                            # Auxiliary coordinates.
                            dims = crs.dim_axes if coord.ndim else ()
                        coord_ = xr.DataArray(data=coord, dims=dims)
                        #
                        # dequantify is needed because pandas index do not keep quantity 
                        # in the coordinates
                        # -> dequantify put units as attributes in the dataset
                        # we need to add also cf-attributes
                        # 
                        res_coords[axis_] = coord_.pint.dequantify()
            case xr.core.coordinates.DatasetCoordinates():
                res_coords = coords
            case _:
                raise TypeError(
                    "'coords' can be a mapping or coordinates object"
                )

        match data_vars:
            case Mapping():
                res_data_vars = {
                    str(name): (
                        var
                        if isinstance(var, xr.DataArray) else
                        xr.DataArray(data=var, dims=self._DIMS_)
                    )
                    for name, var in data_vars.items()
                }
            case None:
                res_data_vars = None
            case _:
                raise TypeError("'data_vars' can be a mapping or 'None'")

        super().__init__(
            data_vars=res_data_vars,
            coords=res_coords,
            coord_system=coord_system,
            attrs=attrs
        )
        if (
            {axis.latitude, axis.longitude}
            == set(self._coord_system.spatial.crs.aux_axes)
        ):
            self._dset = self._dset.set_xindex(
                self.aux_axes, indexes.TwoDimHorGridIndex
            )

    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> PointsFeature:
        # Preparing data, labels, units, and dimensions.
        # TODO: Reconsider this.
        name = self._dset.attrs['_geokube.field_name']
        coord_system = self.coord_system
        dset = self._dset
        lat = dset[axis.latitude]
        lat_vals = lat.data
        lon = dset[axis.longitude]
        lon_vals = lon.data

        # lat_labels = get_magnitude(latitude, lat_data.units)
        # lon_labels = get_magnitude(longitude, lon_data.units)
        lat_labels = np.asarray(latitude)
        lon_labels = np.asarray(longitude)

        if isinstance(coord_system.spatial.crs, Geodetic):
            lat_vals, lon_vals = np.meshgrid(lat_vals, lon_vals, indexing='ij')
            # dims = ('_latitude', '_longitude')
            dims = (axis.latitude, axis.longitude)
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
        dset = dset.drop_vars(
            names=[self.coord_system.spatial.crs.to_cf()['grid_mapping_name']]
        )

        # Creating the resulting points field.
        new_coords = to_points_dict(name=name, dset=dset)
        del new_coords['points']
        new_data = new_coords.pop(name)
        # NOTE: This is important to quantify data since it seems that
        # `xarray.Dataset.pint.quantify()` does not work well with non-string
        # dimensions.
        new_coords = {
            axis_: pint.Quantity(coord, dset[axis_].attrs['units'])
            for axis_, coord in new_coords.items()
        }

        return PointsFeature(
            coords=new_coords,
            coord_system=coord_system,
            data_vars={name: new_data}
        )
