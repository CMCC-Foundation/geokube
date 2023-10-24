from typing import Mapping, Sequence, Self
import numpy.typing as npt
import numpy as np
import pyarrow as pa
from datetime import date, datetime

from . import axis, indexes
from .crs import Geodetic
from .indexers import get_array_indexer, get_indexer
import xarray as xr
import pint
import pint_xarray
from .coord_system import CoordinateSystem
import pandas as pd

from numbers import Number

from enum import Enum

class FeatureType(Enum):
    Points = 1
    Profile = 2
    Grid = 3

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
class Feature():
    __slots__ = ('_dset', '_coord_system')

    def __init__(
        self,
        coords: Mapping[axis.Axis, pint.Quantity] | xr.Coordinates,
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray ] | None = None,
        attrs: Mapping | None = None,
    ) -> None:

        self._dset = xr.Dataset(
            data_vars=data_vars, 
            coords=coords, 
            attrs=attrs
        )

        self.__coord_system = coord_system

    @classmethod
    def _from_xrdset(cls, 
                     dset:xr.Dataset, 
                     coord_system: CoordinateSystem) -> Self:
        return type(cls)(
                data_vars = dset.data_vars,
                coords = dset.coords,
                attrs = dset.attrs,
                coord_system = coord_system
        )

    #TODO: Implement __getitem__ ??

    @property
    def coords(self) -> dict[axis.Axis, pint.Quantity]:
        coords = {}
        for ax in self.coord_system.axes:
            coords[ax] = self._dset[ax].pint.quantify().data
        return coords

    @property
    def coord_system(self):
        return self.__coord_system

# CF Methods 
    @property  # return dimensional axes
    def dim_axes(
        self
    ) -> Sequence[axis.Axis]:
        return self.coord_system.dim_axes

    @property
    def dim_coords(self) -> Mapping[axis.Axis, pint.Quantity]:
        return {
            ax: self._dset[ax].data
            for ax in self.coord_system.dim_axes
        }

    @property  # return dimensional axes
    def aux_axes(
        self
    ) -> Sequence[axis.Axis]:
        return self.coord_system.aux_axes

    @property
    def aux_coords(self) -> Mapping[axis.Axis, pint.Quantity]:
        return {
            ax: self._dset[ax].data
            for ax in self.coord_system.aux_axes
        }

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
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            dset = dset.sel(v_idx)
        return type(self)._from_xrdset(dset, self.coord_system)
    
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
        dset = self._dset.sel({axis.time: slice(start, end)})
        return self._from_xrdset(dset, self.coord_system)
        
    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        dset = self._dset.sel(idx, method='nearest', tolerance=None)
        return self._from_xrdset(dset, self.coord_system)

    def latest(self) -> Self:
        if axis.time not in self.coordinates():
            raise NotImplementedError()
        latest = self._dset[axis.time].max().astype(str).item().magnitude
        idx = {axis.time: slice(latest, latest)}
        dset = self._dset.sel(idx)
        return self._from_xrdset(dset, self.coord_system)

class PointsFeature(Feature):
    _DIMS_ = ('_points')
    __slots__ = ('_n_points')

    def __init__(
        self,
        coords: Mapping[axis.Axis, pint.Quantity] | xr.Coordinates,
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray ] | None = None,
        attrs: Mapping | None = None
    ) -> None:
        
        res_coords = {
            axis_: xr.DataArray(
                vals,
                dims=self._DIMS_
            ) if isinstance(vals, pint.Quantity) else vals 
            for axis_, vals in coords.items()
        } if isinstance (coords, Mapping) else coords

        res_data_vars = {
            name: xr.DataArray(
                    vals,
                    dims=self._DIMS_
                ) if isinstance(vals, pint.Quantity) else vals
                    for name, vals in data_vars.items()
            } if data_vars is not None else None

        super().__init__(data_vars=res_data_vars,
                         coords=res_coords,
                         coord_system=coord_system,
                         attrs=attrs)

        spat_axes = set(coord_system.spatial.crs.axes)
        for axis_ in coord_system.axes:
            if axis_ not in spat_axes:
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
        coords: Mapping[axis.Axis, pint.Quantity] | xr.Coordinates,
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray ] | None = None,
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
    __slots__ = ('_DIMS_',)

    def __init__(
        self,
        coords: Mapping[axis.Axis, pint.Quantity] | xr.Coordinates,
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray ] | None = None,
        attrs: Mapping | None = None
    ) -> None:
        
        super().__init__(data_vars=data_vars,
                         coords=coords,
                         coord_system=coord_system,
                         attrs=attrs)
 
        # for axis_ in self.dim_axes:
        #     ds = self._dset.reset_index(axis_).pint.quantify()
        #     self._dset = ds.set_xindex(axis_, indexes.OneDimPandasIndex)
        
        if {axis.latitude, axis.longitude} <= set(self.aux_axes):
           self._dset = self._dset.set_xindex(self.aux_axes, indexes.TwoDimHorGridIndex)