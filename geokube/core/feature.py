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
class Feature():
    __slots__ = ('_dset', '_coord_system')

    def __init__(
        self,
        coords: xr.Coordinates,
        coord_system: CoordinateSystem,
        data_vars: Mapping[str, pint.Quantity | xr.DataArray ] | None = None,
        attrs: Mapping | None = None,
    ) -> None:

        self._dset = xr.Dataset(
            data_vars=data_vars, 
            coords=coords, 
            attrs=attrs
        )

        self._coord_system = coord_system

    def _from_xrdset(self, dset:xr.Dataset) -> Self:
        return type(self)(
                data_vars = dset.data_vars,
                coords = dset.coords,
                attrs = dset.attrs,
                coord_system = self.coord_system
        )

    #TODO: does __getitem__ works with Axis? does it returns an xr.DataArray?

    @property  # override xr.Dataset coords method to return Mapping axis -> data
    def coords(
        self, coord_axis: axis.Axis | None = None
    ) -> dict[axis.Axis, pint.Quantity] | pint.Quantity:
        if coord_axis is None:
            coords = {}
            for ax in self.coord_system.axes:
                 coords[ax] = self._dset[ax].data
            return coords
        return self.coords[coord_axis]

    @property
    def coord_system(self):
        return self._coord_system

# CF Methods 
    @property  # return dimensional axes
    def dim_axes(
        self
    ) -> Sequence[axis.Axis]:
        return self.coord_system.dim_axes

    @property
    def dim_coords(self) -> Mapping[axis.Axis, pint.Quantity]:
        return {
            ax: coord.data
            for ax, coord in self._dset[self.dim_axes]
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
        new_ds = self._dset.sel(h_idx)
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            new_ds = new_ds._dset.sel(v_idx)
        return self._from_xrdset(new_ds)
    
    def nearest_horizontal(
        self,
        latitude: npt.ArrayLike | pint.Quantity,
        longitude: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.latitude: latitude, axis.longitude: longitude}
        dset = self._dset.sel(idx, method='nearest', tolerance=np.inf)
        return self._from_xrdset(dset)
        
    def nearest_vertical(
        self, elevation: npt.ArrayLike | pint.Quantity
    ) -> Self:
        idx = {axis.vertical: elevation}
        dset = self._dset.sel(idx, method='nearest', tolerance=np.inf)
        return self._from_xrdset(dset)

    def time_range(
        self,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None
    ) -> Self:
        dset = self._dset.sel({axis.time: slice(start, end)})
        return self._from_xrdset(dset)
        
    def nearest_time(
        self, time: date | datetime | str | npt.ArrayLike
    ) -> Self:
        idx = {axis.time: pd.to_datetime(time).to_numpy().reshape(-1)}
        dset = self._dset.sel(idx, method='nearest', tolerance=None)
        return self._from_xrdset(dset)

    def latest(self) -> Self:
        if axis.time not in self.coordinates():
            raise NotImplementedError()
        latest = self._dset[axis.time].max().astype(str).item().magnitude
        idx = {axis.time: slice(latest, latest)}
        dset = self._dset.sel(idx)
        return self._from_xrdset(dset)

class PointsFeature(Feature):
    _DIMS = ('_points')
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
                dims=self._DIMS
            ) if isinstance(vals, pint.Quantity) else vals 
            for axis_, vals in coords.items()
        } if isinstance (coords, Mapping) else coords

        res_data_vars = {
            name: xr.DataArray(
                    vals,
                    dims=self._DIMS
                ) if isinstance(vals, pint.Quantity) else vals
                    for name, vals in data_vars.items()
            } if data_vars is not None else None
        
        super().__init__(data_vars=res_data_vars,
                         coords=res_coords,
                         coord_system=coord_system,
                         attrs=attrs)

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