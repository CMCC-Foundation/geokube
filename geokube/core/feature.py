from collections.abc import Mapping
from datetime import date, datetime
from itertools import chain
from numbers import Number
from typing import Self, Any
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
from .crs import CRS, Geodetic, RotatedGeodetic
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

class FeatureMixin:
    @property
    def coords(self) -> dict[axis.Axis, pint.Quantity]:
        return self._coords

    @property
    def coord_system(self):
        return self._coord_system

    @property
    def crs(self) -> dict[axis.Axis, pint.Quantity]:
        return self.coord_system.spatial.crs

    # CF Methods 
    @property  # return dimensional axes
    def dim_axes(self) -> tuple[axis.Axis, ...]:
        return self.coord_system.dim_axes

    @property
    def dim_coords(self) -> dict[axis.Axis, pint.Quantity]:
        return self._dim_coords

    @property  # return auxiliary axes
    def aux_axes(self) -> tuple[axis.Horizontal, ...]:
        return self.coord_system.aux_axes

    @property
    def aux_coords(self) -> dict[axis.Axis, pint.Quantity]:
        return self._aux_coords

    def sel(
        self, indexers: Mapping[axis.Axis, Any], **xarray_kwargs: Mapping
    ) -> Self:
#        return type(self)(self._dset.sel(indexers, **xarray_kwargs))
        return self._from_xarray_dataset(
            self._dset.sel(indexers, **xarray_kwargs)
        )
    
    def isel(
        self, indexers: Mapping[axis.Axis, Any], **xarray_kwargs: Mapping
    ) -> Self:
#        return type(self)(self._dset.isel(indexers, **xarray_kwargs))
        return self._from_xarray_dataset(
            self._dset.isel(indexers, **xarray_kwargs)
        )

    # spatial operations
    def bounding_box(
        self,
        south: Number | pint.Quantity | None = None,
        north: Number | pint.Quantity | None = None,
        west: Number | pint.Quantity | None = None,
        east: Number | pint.Quantity | None = None,
        bottom: Number | pint.Quantity | None = None,
        top: Number | pint.Quantity | None = None
    ) -> Self:
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
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        return self.sel(idx, method='nearest', tolerance=np.inf)

    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.vertical: elevation}
        return self.sel(idx, method='nearest', tolerance=np.inf)

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        idx = {axis.time: slice(start, end)}
        return self.sel(idx)

    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        return self.sel(idx, method='nearest', tolerance=None)

    def latest(self) -> Self:
        if axis.time not in self._dset.coords:
            raise NotImplementedError()
        latest = self._dset[axis.time].max().astype(str).item()
        idx = {axis.time: slice(latest, latest)}
        return self.sel(idx)


class Feature(FeatureMixin):
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

        self._coords = {
            axis_: ds[axis_].pint.quantify().data
            for axis_ in coord_system.axes
        }

        self._dim_coords = {ax: ds[ax].data for ax in coord_system.dim_axes}

        self._aux_coords = {ax: ds[ax].data for ax in coord_system.aux_axes}

    @classmethod
    def _from_xarray_dataset(
        cls,
        ds: xr.Dataset,
        cf_mappings: Mapping[str, str] | None = None # this could be used to pass CF compliant hints
    ) -> Self:
        return cls(ds, cf_mappings)

    #TODO: Implement __getitem__ ??


class PointsFeature(Feature):
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
        return self._dset['_points'].size


class ProfilesFeature(Feature):
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
        return self._dset['_profiles'].size

    @property
    def number_of_levels(self) -> int:
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
        return PointsFeature._from_xarray_dataset(_as_points_dataset(self))


class GridFeature(Feature):
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

    name = dset.attrs.get('_geokube.field_name')

    if name is None:
        return data

    # Data.
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
    if (name := feature._dset.attrs.get('_geokube.field_name')) is not None:
        new_data = {name: new_coords.pop(name)}
    else:
        new_data = None

    # Creating the resulting dataset and corresponding result.
    new_dset = xr.Dataset(
        data_vars=new_data, coords=new_coords, attrs=feature._dset.attrs
    )
    return new_dset


class GridFeature_(Feature):
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

        # TODO: Refactor this.
        match coords:
            case Mapping():
                res_coords = {}
                for axis_, coord in coords.items():
                    if isinstance(coord, xr.DataArray):
                        coord_data = coord.data
                        if isinstance(coord_data, pint.Quantity):
                            coord_vals = coord_data.magnitude
                            coord_units = str(coord_data.units)
                        else:
                            coord_vals = coord_data
                            coord_units = 'dimensionless'
                        if (
                            coord.dtype is np.dtype(object)
                            and isinstance(coord_vals[0], pd.Interval)
                        ):
                            coord = xr.DataArray(
                                data=pd.IntervalIndex(
                                    coord_vals, closed='both'
                                ),
                                dims=coord.dims,
                                attrs={'units': coord_units}
                            )
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
                        coord_vals = coord.magnitude
                        if (
                            coord_vals.dtype is np.dtype(object)
                            and isinstance(coord_vals[0], pd.Interval)
                        ):
                            coord_data = pd.IntervalIndex(
                                coord_vals, closed='both'
                            )
                        else:
                            coord_data = coord_vals
                        coord_ = xr.DataArray(
                            data=coord_data,
                            dims=dims,
                            attrs={'units': str(coord.units)}
                        )
                        #
                        # dequantify is needed because pandas index do not keep quantity 
                        # in the coordinates
                        # -> dequantify put units as attributes in the dataset
                        # we need to add also cf-attributes
                        # 
                        # res_coords[axis_] = coord_.pint.dequantify()
                        res_coords[axis_] = coord_

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
        # ---> ???
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