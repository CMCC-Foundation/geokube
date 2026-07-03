from __future__ import annotations
import warnings

import os
import json
import uuid
import tempfile
import shutil
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
from .enums import MethodType, RegridMethod
from .errs import EmptyDataError
from .axis import Axis
from .datacube import DataCube
from .domain import Domain
from .field import Field


class Dataset:
    __slots__ = (
        "__data",
        "__metadata",
        "__attrs",
        "__cube_idx",
        "__load_files_on_persistance",
        "__representative",
    )

    _LOG = HCubeLogger(name="Dataset")

    FIELD_COL = "fields"
    DATACUBE_COL = "datacube"
    FILES_COL = "files"

    def __init__(
        self,
        hcubes: Mapping[tuple[str, ...], DataCube] | pd.DataFrame,
        attrs: Sequence[str] = None,
        metadata: Mapping[str, str] | None = None,
        load_files_on_persistance: None | bool = True,
    ) -> None:
        # NOTE: to support Dataset operations
        # for not-netcdf files
        self.__load_files_on_persistance = load_files_on_persistance
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
        # Cache for the lazily-opened representative cube (the schema source when the
        # cubes are Delayed handles). See `_representative`.
        self.__representative = None

    def __getitem__(self, key: Union[str, Tuple[str]]) -> Dataset:
        # TODO: Check if `.copy()` is necessary here.
        data = self.__data.iloc[:, : self.__cube_idx].copy()
        key = {key} if isinstance(key, str) else set(key)
        # Delayed cubes don't expose their fields without opening one. Use the
        # (homogeneous) representative's schema to pick eligible names, then subset
        # through dask's proxy so the cube stays Delayed. Empty rows are dropped inline:
        # Delayed rows carry FIELD_COL=None, so the FIELD_COL-based `_drop_empty` cannot
        # be used on the lazy path.
        valid = None
        new_cubes, keep = [], []
        for hcube in self.__data[self.DATACUBE_COL].to_numpy().flat:
            if isinstance(hcube, Delayed):
                if valid is None:
                    rep = self._representative()
                    valid = (
                        set(rep.fields) | set(rep._ncvar_to_name or {})
                        if rep is not None
                        else set()
                    )
                elig = key & valid
                new_cubes.append(hcube[elig] if elig else None)
                keep.append(bool(elig))
            elif hcube is None:
                new_cubes.append(None)
                keep.append(False)
            else:
                sub = Dataset._get_eligible_fields_for_datacube(hcube, key)
                new_cubes.append(sub)
                keep.append(bool(len(sub)))
        data[self.DATACUBE_COL] = new_cubes
        # numpy bool array -> unambiguous positional row-mask (a plain [] list would be
        # read as an empty column selection when the dataset has no rows).
        data = data[np.array(keep, dtype=bool)]
        data.index = np.arange(len(data))
        return Dataset(
            attrs=self.__attrs,
            hcubes=data,
            metadata=self.__metadata,
            load_files_on_persistance=self.__load_files_on_persistance,
        )

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

    def resample(
        self,
        operator: Union[Callable, MethodType, str],
        frequency: str,
        **resample_kwargs,
    ) -> Dataset:
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.resample(
                operator=operator, frequency=frequency, **resample_kwargs
            )
        )
        return Dataset(
            attrs=self.__attrs, hcubes=_copy, metadata=self.__metadata
        )

    def average(self, dim: str | None = None) -> Dataset:
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.average(dim=dim)
        )
        return Dataset(
            attrs=self.__attrs, hcubes=_copy, metadata=self.__metadata
        )

    def regrid(
        self,
        target: Union[Domain, Field],
        method: Union[str, RegridMethod] = "bilinear",
        weights_path: Optional[str] = None,
        reuse_weights: bool = True,
    ) -> Dataset:
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.regrid(
                target=target,
                method=method,
                weights_path=weights_path,
                reuse_weights=reuse_weights,
            )
        )
        return Dataset(
            attrs=self.__attrs, hcubes=_copy, metadata=self.__metadata
        )

    def to_regular(self) -> Dataset:
        _copy = self.__data.copy()
        _copy[self.DATACUBE_COL] = _copy[self.DATACUBE_COL].apply(
            lambda x: x.to_regular()
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
        if self.__load_files_on_persistance:
            return self.__data[self.DATACUBE_COL].tolist()
        else:
            return None

    @property
    def variables(self) -> List[str]:
        """Variable (field) names exposed by the catalog.

        Read from a single representative cube (see :meth:`_representative`), so the list
        is available even when the cubes are lazy ``Delayed`` handles
        (``delay_read_cubes=True``).
        """
        rep = self._representative()
        return list(rep.fields.keys()) if rep is not None else []

    def _representative(self) -> Optional[DataCube]:
        """The schema-defining cube for a (homogeneous) catalog, opened once and cached.

        With ``delay_read_cubes=True`` the cubes are lazy ``Delayed`` handles whose
        variables are unknown until one is opened. Since a catalog's groups are
        homogeneous (same variables/grid), a single representative cube defines the schema
        for the whole dataset (variable listing, :meth:`to_dict`, variable selection). It
        is computed lazily and memoized, so pure filter/persist flows that never need the
        schema open no cube at all.
        """
        if self.__representative is not None:
            return self.__representative
        for hcube in self.__data[self.DATACUBE_COL].to_numpy().flat:
            if hcube is None:
                continue
            self.__representative = (
                hcube.compute() if isinstance(hcube, Delayed) else hcube
            )
            return self.__representative
        return None

    def to_dict(self, unique_values=False) -> dict:
        # Delayed cubes have no schema until one is opened. For a homogeneous catalog,
        # open a single representative cube and reuse its dict (variables + domain) for
        # every Delayed row instead of emitting `datacube: None`. The dict is shared
        # across those rows (the description is read-only). Per-cube attribute values
        # (e.g. the forecast start-date) still come from `attributes`, not from the
        # representative's domain.
        rep_dict = None
        if any(
            isinstance(hcube, Delayed)
            for hcube in self.__data[self.DATACUBE_COL].to_numpy().flat
        ):
            rep = self._representative()
            rep_dict = rep.to_dict(unique_values) if rep is not None else None
        res = self.__data.drop(
            labels=Dataset.FILES_COL, inplace=False, axis=1
        ).apply(
            Dataset._row_to_dict,
            attrs=self.__attrs,
            unique_values=unique_values,
            representative_dict=rep_dict,
            axis=1,
        )
        return list(res)

    @staticmethod
    def _row_to_dict(row, attrs, unique_values, representative_dict=None):
        cube = row[Dataset.DATACUBE_COL]
        if isinstance(cube, Delayed):
            datacube = representative_dict
        elif cube is None:
            datacube = None
        else:
            datacube = cube.to_dict(unique_values)
        return {
            "datacube": datacube,
            "attributes": {attr_name: row[attr_name] for attr_name in attrs},
        }

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
            # np.isin (not the removed-in-NumPy-2.2 np.in1d): same boolean mask over the
            # 1-D attribute column, scalar or list `param_value`.
            mask &= np.isin(self.__data[param_name], param_value)
        data = self.__data.loc[mask, : self.DATACUBE_COL]
        data.index = np.arange(len(data))

        return Dataset(
            attrs=self.__attrs,
            hcubes=data,
            metadata=self.__metadata,
            load_files_on_persistance=self.__load_files_on_persistance,
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
        if any(isinstance(cube, Delayed) for cube in self.cubes):
            return sum(
                sum(os.path.getsize(f) for f in files)
                for files in self.__data[Dataset.FILES_COL]
            )
        return sum(cube.nbytes for cube in self.cubes)

    def _persist_datacube(self, dataframe_item, path):
        if self.__load_files_on_persistance:
            dcube = dataframe_item[self.DATACUBE_COL]
            attr_str = self._form_attr_str(dataframe_item)
            if isinstance(dcube, Delayed):
                dcube = dcube.compute()
            try:
                return dcube.persist(os.path.join(path, f"{attr_str}.nc"))
            except EmptyDataError:
                self._LOG.warn(f"Skipping empty Dataset item!")
                return None
        else:
            attr_str = self._form_attr_str(dataframe_item)
            if len(dataframe_item[Dataset.FILES_COL]) > 1:
                raise ValueError(
                    "Too many files! Copying source files is supported for"
                    " `1` but provided"
                    f" `{len(dataframe_item[Dataset.FILES_COL])}`"
                )
            for file in dataframe_item[Dataset.FILES_COL]:
                dst_path = os.path.join(
                    path,
                    self._convert_attributes_to_file_name(attr_str, file),
                )
                shutil.copyfile(file, dst_path)
                return dst_path

    def _form_attr_str(self, dataframe_item):
        return "-".join(
            [
                f"{attr_name}={dataframe_item[attr_name]}"
                for attr_name in self.__attrs
            ]
        )

    def _convert_attributes_to_file_name(self, attr_str, file):
        return f"{attr_str}-{os.path.basename(file)}"


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
