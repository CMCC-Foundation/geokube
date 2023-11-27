from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping, Sequence
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from .axis import Axis
from .coord_system import _COORDINATE_SYSTEMS
from .field import Field

from .cube import Cube

class Collection:
    __slots__ = ('__dataframe', '__cube_idx')

    __CUBE_COL = "cube"

    def __init__(
        self,
        data: Sequence[Sequence[str | Cube]] | pd.DataFrame,
        filters: Sequence[str] | None = None,
    ) -> None:
        if isinstance(data, Sequence):
            self.__filters = tuple(filters) if filters is not None else ()
            self.__dataframe = pd.DataFrame(
                data=data, columns=self.__filters + (self.__CUBE_COL,)
            )
        elif isinstance(data, pd.DataFrame):
            self.__dataframe = data
            if filters:
                raise ValueError(
                    "'filters' must be 'None' or empty sequence when 'data' is "
                    "instance of 'pd.DataFrame'"
                )
            reserved_names = {self.__FIELD_COL}
            self.__filters = tuple(
                filters
                for filter in data.columns.to_list()
                if filter not in reserved_names
            )
        else:
            raise TypeError(
                "'data' must be either a sequence that contain parameters and "
                "fields or an instance of 'pandas.DataFrame'"
            )
        self.__cube_idx = len(self.__filters) + 1

    def filter(
        self, **filters_kwargs
    ) -> Collection | Cube:
        
        filters = {**filters_kwargs}

        if not (idx := filters.keys()) <= (filters := set(self.__filters)):
            # TODO: Make better message.
            raise ValueError(
                f"'filter' cannot use the argument(s): {sorted(idx - filters)}"
            )

        mask = np.full(shape=len(self.__dataframe), fill_value=True, dtype=np.bool_)
        for name, value in filters.items():
            mask &= np.in1d(self.__data[name], value)
        data = self.__dataframe.loc[mask, : self.__CUBE_COL]
        data.index = np.arange(len(data))

        return Collection(
            data=data,
            filters=self.__filters,
        )
    
    def cubes(self) -> Sequence[Cube]:
        return list(self.__dataframe[self.__cube_idx])
    
    def merge(self) -> xr.Dataset:
        pass
