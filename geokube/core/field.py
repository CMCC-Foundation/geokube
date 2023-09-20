from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Number
from typing import Any, Self

import dask.array as da
import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from . import axis, domain as domain_, indexes


class PointsField:
    __slots__ = (
        '__name',
        '__data',
        '__anciliary',
        '__domain',
        '__properties',
        '__encoding'
    )

    _DOMAIN_TYPE = domain_.Points

    def __init__(
        self,
        name: str,
        domain: domain_.Points,
        data: npt.ArrayLike | pint.Quantity | None = None,
        anciliary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        self.__name = str(name)
        if not isinstance(domain, domain_.Points):
            raise TypeError("'domain' must be an instance of 'Points'")
        self.__anciliary = dict(anciliary) if anciliary else {}
        self.__domain = domain
        self.__properties = dict(properties) if properties else {}
        self.__encoding = dict(encoding) if encoding else {}

        n_pts = domain.number_of_points
        arr_types = (np.ndarray, da.Array)
        match data:
            case pint.Quantity() if isinstance(data.magnitude, arr_types):
                data_ = data
            case pint.Quantity():
                data_ = pint.Quantity(np.asarray(data.magnitude), data.units)
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
    def domain(self) -> domain_.Points:
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
    ) -> Self | PointsField:
        h_idx = {
            axis.latitude: slice(south, north),
            axis.longitude: slice(west, east)
        }
        new_data = self.__data.sel(h_idx, method='nearest', tolerance=np.inf)
        if not (bottom is None and top is None):
            v_idx = {axis.vertical: slice(bottom, top)}
            new_data = self._new_field(new_data)._data
            new_data = new_data.sel(v_idx, method='nearest', tolerance=np.inf)
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

    def nearest_time(self, time: date | datetime | str) -> Self:
        idx = {axis.time: time}
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

    _DOMAIN_TYPE = domain_.Profile

    def __init__(
        self,
        name: str,
        domain: domain_.Profile,
        data: npt.ArrayLike | pint.Quantity | None = None,
        anciliary: Mapping | None = None,
        properties: Mapping | None = None,
        encoding: Mapping | None = None
    ) -> None:
        self.__name = str(name)
        if not isinstance(domain, domain_.Profile):
            raise TypeError("'domain' must be an instance of 'Profile'")
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
            all_sizes, all_units = [], set()
            for data_item in data:
                all_sizes.append(len(data_item))
                if isinstance(data_item, pint.Quantity):
                    all_units.add(data_item.units)
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
            for i, (stop_idx, vals) in enumerate(zip(all_sizes, data)):
                if stop_idx == n_lev:
                    data_vals[i, :] = vals
                else:
                    data_vals[i, :stop_idx] = vals
                    data_vals[i, stop_idx:] = np.nan
            data_ = pint.Quantity(data_vals, unit)
        else:
            arr_types = (np.ndarray, da.Array)
            match data:
                case pint.Quantity() if isinstance(data.magnitude, arr_types):
                    data_ = data
                case pint.Quantity():
                    data_ = pint.Quantity(
                        np.asarray(data.magnitude), data.units
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
        # TODO: Consider adding the index for vertical.
        coord_system = domain.coord_system
        spat_axes = set(coord_system.spatial.axes)
        for axis_ in coord_system.axes:
            if axis_ not in spat_axes:
                dset = dset.set_xindex(axis_, indexes.OneDimIndex)
        dset = dset.set_xindex(
            [axis.latitude, axis.longitude], indexes.TwoDimHorPointsIndex
        )
        self.__data = dset

    @property
    def name(self) -> str:
        return self.__name

    @property
    def domain(self) -> domain_.Profile:
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
        new_data = self.__data.sel(h_idx, method='nearest', tolerance=np.inf)
        if not (bottom is None and top is None):
            raise NotImplementedError()
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
        raise NotImplementedError()
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

    def nearest_time(self, time: date | datetime | str) -> Self:
        idx = {axis.time: time}
        new_data = self.__data.sel(idx, method='nearest', tolerance=None)
        return self._new_field(new_data)
