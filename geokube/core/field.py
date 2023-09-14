from collections.abc import Mapping, Sequence

import dask.array as da
import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from . import domain as domain_, indexes


class PointsField:
    __slots__ = (
        '__name',
        '__data',
        '__anciliary',
        '__domain',
        '__properties',
        '__encoding'
    )

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
        for axis_ in domain.coord_system.axes:
            dset = dset.set_xindex(axis_, indexes.OneDimIndex)
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


class ProfileField:
    __slots__ = (
        '__name',
        '__data',
        '__anciliary',
        '__domain',
        '__properties',
        '__encoding'
    )

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
        # for axis_ in domain.coord_system.axes:
        #     dset = dset.set_xindex(axis_)
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
