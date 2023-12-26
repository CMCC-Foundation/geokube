from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .cube import Cube


class Collection:
    __slots__ = ('__data_frame', '__cube_idx', '__filters')

    __CUBE_COL = "cube"

    def __init__(
        self,
        data: Sequence[Sequence[str | Cube]] | pd.DataFrame,
        filters: Sequence[str] = (),
    ) -> None:
        if isinstance(data, Sequence):
            self.__filters = tuple(filters)
            self.__data_frame = pd.DataFrame(
                data=data, columns=self.__filters + (self.__CUBE_COL,)
            )
        elif isinstance(data, pd.DataFrame):
            self.__data_frame = data
            if filters:
                raise ValueError(
                    "'filters' must be an empty sequence when 'data' is an "
                    "instance of 'pandas.DataFrame'"
                )
            reserved_names = {self.__CUBE_COL}
            self.__filters = tuple(
                filter
                for filter in data.columns.to_list()
                if filter not in reserved_names
            )
        else:
            raise TypeError(
                "'data' must be either a sequence that contains filters and "
                "cubes or an instance of 'pandas.DataFrame'"
            )
        self.__cube_idx = len(self.__filters)

    # TODO: Consider removing `_data`. It is added for testing purposes.
    @property
    def _data(self) -> pd.DataFrame:
        return self.__data_frame

    def filter(self, **filter_kwargs) -> Collection:
        if not (
            (idx := filter_kwargs.keys()) <= (filters := set(self.__filters))
        ):
            # TODO: Make better message.
            raise ValueError(
                f"'filter' cannot use the argument(s): {sorted(idx - filters)}"
            )
        mask = np.full(
            shape=len(self.__data_frame), fill_value=True, dtype=np.bool_
        )
        for filter_name, filter_values in filter_kwargs.items():
            mask &= np.in1d(self.__data_frame[filter_name], filter_values)
        data = self.__data_frame.loc[mask]
        data.index = np.arange(len(data))
        data = self.__data_frame.loc[mask]
        data.reset_index(inplace=True, drop=True)
        return Collection(data=data)

    def cubes(self) -> list[Cube]:
        return self.__data_frame.iloc[:, self.__cube_idx].to_list()

    def merge(self) -> xr.Dataset:
        pass
