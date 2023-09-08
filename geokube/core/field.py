from collections.abc import Mapping

import dask.array as da
import numpy as np
import numpy.typing as npt
import pint
import xarray as xr

from . import domain as domain_


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
        data: npt.ArrayLike | pint.Quantity = np.nan,
        data_units: str | pint.Quantity = '',
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

        # TODO: Consider converion of `data.units` if it does not match
        # `data_units`.
        match data:
            case pint.Quantity():
                data_ = data
            case np.ndarray() | da.Array():
                # NOTE: The pattern arr * unit does not work when arr has
                # stings.
                data_ = pint.Quantity(data, str(data_units))
            case _:
                data_ = pint.Quantity(np.asarray(data), str(data_units))
        if data_.ndim != 1:
            raise ValueError(
                "'coords' must have only one-dimensional values"
            )
        self.__data = xr.Dataset(
            data_vars={self.__name: (('_points',), data_)},
            coords=domain._coords
        )

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
