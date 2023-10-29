from collections.abc import Mapping
from datetime import date, datetime
from itertools import chain
from numbers import Number
from typing import Self

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
    def dim_axes(self) -> tuple[axis.Axis]:
        return self.coord_system.dim_axes

    @property
    def dim_coords(self) -> dict[axis.Axis, pint.Quantity]:
        return {ax: self._dset[ax].data for ax in self.coord_system.dim_axes}

    @property  # return auxiliary axes
    def aux_axes(self) -> tuple[axis.Horizontal]:
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
        
        super().__init__(data_vars=data_vars,
                         coords=coords,
                         coord_system=coord_system,
                         attrs=attrs)

        spat_axes = set(coord_system.spatial.axes)
        for axis_ in coord_system.axes:
            if axis_ not in spat_axes:
                self._dset = self._dset.set_xindex(axis_, indexes.OneDimIndex)
        
        self._dset = self._dset.set_xindex(
            axis.vertical,
            indexes.TwoDimVertProfileIndex,
            data=self._dset[self.name], # why is this needed?
            name=self.name # why is this needed?
        )

        self._dset = self._dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )

    @property
    def number_of_profiles(self) -> int:
        return self._n_profiles

    @property
    def number_of_levels(self) -> int:
        return self._n_levels


class GridFeature(Feature):
    # TODO: Clarify whether/why is this required.
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
        super().__init__(
            data_vars=data_vars,
            coords=coords,
            coord_system=coord_system,
            attrs=attrs
        )
        if {axis.latitude, axis.longitude} == set(self.aux_axes):
            self._dset = self._dset.set_xindex(
                self.aux_axes, indexes.TwoDimHorGridIndex
            )
