from __future__ import annotations
import warnings

import os
import uuid
import tempfile
from collections.abc import Callable, Mapping, Sequence
from numbers import Number
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dask.delayed import Delayed
from zipfile import ZipFile

from ..utils.decorators import geokube_logging
from ..utils import util_methods
from ..utils.hcube_logger import HCubeLogger
from .errs import EmptyDataCubeError
from .axis import Axis
from .datacube import DataCube


class Dataset:
    __slots__ = ("__data", "__metadata", "__attrs", "__cube_idx")

    _LOG = HCubeLogger(name="Dataset")

    FIELD_COL = "fields"
    DATACUBE_COL = "datacube"
    FILES_COL = "files"

    def __init__(
        self,
        hcubes: Mapping[tuple[str, ...], DataCube] | pd.DataFrame,
        attrs: Sequence[str] = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        # TODO: Make `attrs` capable of taking `np.ndarray`.
        if attrs is None:
            attrs = []
        if isinstance(hcubes, Mapping):
            self.__data = pd.Series(hcubes).reset_index()
            self.__data.columns = attrs + [self.DATACUBE_COL]
            self.__attrs = list(attrs)
        elif isinstance(hcubes, pd.DataFrame):
            self.__data = hcubes
            self.__attrs = [
                attr
                for attr in hcubes.columns
                if attr != self.DATACUBE_COL
                and attr != self.FILES_COL
                and attr != self.FIELD_COL
            ]
        else:
            raise TypeError("'hcubes' must be mapping or pandas DataFrame")

        self.__data[self.FIELD_COL] = [
            None
            if isinstance(hcube, Delayed) or hcube is None
            else list(hcube._fields.keys())
            for hcube in self.__data[self.DATACUBE_COL].to_numpy().flat
        ]

        self.__cube_idx = len(self.__attrs) + 1
        self.__metadata = dict(metadata) if metadata is not None else {}

    def __getitem__(self, key: Union[str, Tuple[str]]) -> Dataset:
        # TODO: Check if `.copy()` is necessary here.
        data = self.__data.iloc[:, : self.__cube_idx].copy()
        key = {key} if isinstance(key, str) else set(key)
        data[self.DATACUBE_COL] = [
            None
            if isinstance(hcube, Delayed)
            else Dataset._get_eligible_fields_for_datacube(hcube, key)
            for hcube in self.__data[self.DATACUBE_COL].to_numpy().flat
        ]
        dset = Dataset(
            attrs=self.__attrs, hcubes=data, metadata=self.__metadata
        )
        return dset._drop_empty()

    def __len__(self):
        return len(self.__data)

    @staticmethod
    def _get_eligible_fields_for_datacube(hcube, key: set):
        return hcube[
            (key & hcube._fields.keys()) | (key & hcube._ncvar_to_name.keys())
        ]

    def geobbox(
        self,
        north=None,
        south=None,
        west=None,
        east=None,
        top=None,
        bottom=None,
    ):
        # this returns a new Dataset where each Datacube is subsetted according to the coordinates
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.geobbox(
                north=north,
                south=south,
                west=west,
                east=east,
                top=top,
                bottom=bottom,
            )
        )
        return Dataset(
            attrs=self.__attrs, hcubes=_copy, metadata=self.__metadata
        )

    def locations(
        self,
        latitude,
        longitude,
        vertical: Optional[List[Number]] = None,
    ):
        # this returns a new Dataset where each Datacube is subsetted according to the coordinates
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.locations(
                latitude=latitude, longitude=longitude, vertical=vertical
            )
        )
        return Dataset(
            attrs=self.__attrs, hcubes=_copy, metadata=self.__metadata
        )

    def sel(
        self,
        indexers: Mapping[Union[Axis, str], Any] = None,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        roll_if_needed: bool = True,
        **indexers_kwargs: Any,
    ) -> Dataset:  # this can be only independent variables
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.sel(
                indexers=indexers,
                method=method,
                tolerance=tolerance,
                drop=drop,
                roll_if_needed=roll_if_needed,
                **indexers_kwargs,
            )
        )
        return Dataset(
            attrs=self.__attrs, hcubes=_copy, metadata=self.__metadata
        )

    def update_metadata(self, metadata: dict):
        self.__metadata.update(metadata)

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    @property
    def metadata(self) -> dict[str, str]:
        return self.__metadata

    @property
    def cubes(self) -> List[DataCube]:
        return self.__data[self.DATACUBE_COL].tolist()

    def to_dict(self) -> dict[Tuple[str, ...], DataCube]:
        # NOTE: List of files is not hashable and it can be extremely large
        res = (
            self.__data.drop(labels=Dataset.FILES_COL, inplace=False, axis=1)
            .applymap(util_methods.to_dict_if_possible)
            .to_dict("records")
        )
        return res

    def _drop_empty(self) -> Dataset:
        mask = self.__data[self.FIELD_COL].astype(dtype=np.bool_)
        data = self.__data.loc[mask, : self.DATACUBE_COL]
        data.index = np.arange(len(data))
        return Dataset(
            attrs=self.__attrs, hcubes=data, metadata=self.__metadata
        )

    @geokube_logging
    def filter(
        self, indexers: Optional[Mapping[str, str]] = None, **indexers_kwargs
    ) -> Dataset:
        if indexers is None:
            params = indexers_kwargs
        else:
            if intersect := sorted(indexers.keys() & indexers_kwargs.keys()):
                raise ValueError(
                    "'indexers' and 'indexers_kwargs' have common parameters:"
                    " {intersect}"
                )
            params = {**indexers, **indexers_kwargs}

        if not (idx := params.keys()) <= (attrs := set(self.__attrs)):
            # TODO: Make better message.
            raise ValueError(
                f"'filter' cannot use the argument(s): {sorted(idx - attrs)}"
            )

        mask = np.full(shape=len(self.__data), fill_value=True, dtype=np.bool_)
        for param_name, param_value in params.items():
            mask &= np.in1d(self.__data[param_name], param_value)
        data = self.__data.loc[mask, : self.DATACUBE_COL]
        data.index = np.arange(len(data))

        return Dataset(
            attrs=self.__attrs, hcubes=data, metadata=self.__metadata
        )

    def apply(
        self,
        func: Union[Callable[[DataCube], DataCube], property, str],
        drop_empty: Optional[bool] = False,
        **kwargs,
    ) -> Dataset:
        data = self.__data.iloc[:, : self.__cube_idx].copy()
        data[self.DATACUBE_COL] = [
            _apply(hcube, func, **kwargs)
            for hcube in self.__data[self.DATACUBE_COL].to_numpy().flat
        ]
        data.index = np.arange(len(data))
        dset = Dataset(
            attrs=self.__attrs, hcubes=data, metadata=self.__metadata
        )
        return dset._drop_empty() if drop_empty else dset

    def persist(self, path=None, zip_if_many=False) -> str:
        if path is None:
            path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        if not os.path.exists(path):
            os.makedirs(path)

        paths = self.data.apply(self._persist_datacube, path=path, axis=1)
        # NOTE: omit None paths -- results of empty DataCube persistance
        paths = paths[~paths.isna()]
        if len(paths) == 0:
            warnings.warn(
                "No files were created while geokube.Dataset persisting!"
            )
            return None
        elif len(paths) == 1:
            return paths.iloc[0]
        if zip_if_many:
            path = os.path.join(path, f"{str(uuid.uuid4())}.zip")
            with ZipFile(path, "w") as archive:
                for file in paths:
                    archive.write(file, arcname=os.path.basename(file))
            for file in paths:
                os.remove(file)
        return path

    @property
    def nbytes(self) -> int:
        return sum(cube.nbytes for cube in self.cubes)

    def _persist_datacube(self, dataframe_item, path):
        path_to_store = os.path.join(
            path, os.path.basename(dataframe_item[self.FILES_COL][0])
        )
        dcube = dataframe_item[self.DATACUBE_COL]
        if isinstance(dcube, Delayed):
            dcube.compute()
        try:
            return dcube.persist(path_to_store)
        except EmptyDataCubeError:
            self._LOG.warn(f"Skipping empty Dataset item!")
            return None


def _apply(
    hcube: DataCube,
    func: Union[Callable[[DataCube], DataCube], property, str],
    **kwargs,
) -> DataCube:
    if isinstance(func, str):
        func = getattr(DataCube, func)
    if callable(func):
        return func(hcube, **kwargs)
    if isinstance(func, property):
        if kwargs:
            raise ValueError(
                "'func' is a property and cannot accept additional arguments"
            )
        return func.fget(hcube)
    raise TypeError(
        "'func' must be callable or str which represents the name of a "
        "callable member of DataCube class"
    )
