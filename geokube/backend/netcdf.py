from __future__ import annotations

__all__ = (
    "open_datacube",
    "open_dataset",
)

import glob
import logging
import os
import pickle
from string import Formatter
from typing import Any, Hashable, Mapping, Optional

import dask
import pandas as pd
import xarray as xr
from intake.source.utils import reverse_format
import rioxarray

import geokube.backend.base
import geokube.core.datacube
import geokube.core.dataset
from geokube.utils.hcube_logger import HCubeLogger

LOG = HCubeLogger(name="netcdf.py")

FILES_COL = geokube.core.dataset.Dataset.FILES_COL
DATACUBE_COL = geokube.core.dataset.Dataset.DATACUBE_COL


def _get_engine(path: list | str):
    if isinstance(path, list):
        if len(path) > 0:
            path = path[0]
        else:
            raise ValueError("empty path list provided!")
    if isinstance(path, str):
        _, ext = os.path.splitext(path)
    else:
        raise TypeError(f"unsupported path type: `{type(path)}`")
    if ext == ".tif":
        return "rasterio"
    elif ext == ".nc":
        return "netcdf4"
    elif ext == ".jp2":
        return "rasterio"
    else:
        raise ValueError(
            f"there is not engine associated with the extension `{ext}`"
        )


def _read_cache(cache_path: str):
    cached = None
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
    return cached


def _write_cache(ds, cache_path: str):
    with open(cache_path, "wb") as f:
        pickle.dump(ds, f)


def open_datacube(
    path: str,
    id_pattern: Optional[str] = None,
    mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    metadata_caching: bool = False,
    metadata_cache_path: str = None,
    **kwargs,  # optional kw args for xr.open_mfdataset
) -> geokube.core.datacube.DataCube:
    # TODO: incremental metadata caching
    # we could load the cache file and compare the files from the paths with the files
    # in the dataframe cached and read only the files that are not in the cache
    if metadata_caching:
        ds = _read_cache(metadata_cache_path)
        if ds is not None:
            return ds
    engine = kwargs.pop('engine', None) or _get_engine(path)
    if engine == "netcdf4":
        kwargs.setdefault("decode_coords", "all")
    ds = geokube.core.datacube.DataCube.from_xarray(
        xr.open_mfdataset(path, engine=engine, **kwargs),
        id_pattern=id_pattern,
        mapping=mapping,
    )
    if metadata_caching:
        _write_cache(ds, metadata_cache_path)

    return ds


def _get_ds_attrs_names(pattern):
    fmt = Formatter()
    # get the dataset attrs from the pattern
    ds_attr_names = [i[1] for i in fmt.parse(pattern) if i[1]]
    return ds_attr_names


def _get_df_from_files_list(files, pattern, ds_attr_names):
    l = []
    for f in files:
        d = reverse_format(pattern, f)
        d[FILES_COL] = f
        l.append(d)
    df = pd.DataFrame(l)
    if len(l) == 0:
        raise ValueError(f"No files found for the provided path!")
    # unique index for each dataset attribute combos - we create a list of files
    df = df.groupby(ds_attr_names)[FILES_COL].apply(list).reset_index()
    df = df.set_index(ds_attr_names)
    return df


def open_dataset(
    path: str,
    pattern: str,
    id_pattern: Optional[str] = None,
    mapping: Optional[Mapping[str, Mapping[str, str]]] = None,
    metadata_caching: bool = False,
    metadata_cache_path: str = None,
    ds_attr_mapping: Mapping[
        Hashable, Any
    ] = None,  # dataset attributes mapping - TBA
    # { 'attr_name1': { 'id': ..., 'description': ... }, ...}
    ncvars_mapping: Mapping[
        Hashable, Any
    ] = None,  # netcdf variables mapping - TBA
    # { 'ncvar_name1': {'id': ..., 'description': }, ...}
    delay_read_cubes: bool = False,  # when True the method will not create datacubes when opening a dataset; this is useful
    # when the number of rows is really high and the number of files per row is low
    # (e.g CMIP, CORDEX, observations). The datacube will be read when trying to accessing it
    load_files_on_persistance: bool = True,
    **kwargs,  # optional kw args for xr.open_mfdataset
) -> geokube.core.dataset.Dataset:
    # incremental metadata caching:
    # load the cache file and compare the files from the paths with the files
    # in the dataframe cached and read only the files that are not in the cache
    ds_attr_names = _get_ds_attrs_names(pattern)
    if metadata_caching:
        if metadata_cache_path is None:
            raise ValueError(
                "If `metadata_caching` set to True, `metadata_cache_path`"
                " argument needs to be provided!"
            )
        cached_ds = _read_cache(metadata_cache_path)
        if cached_ds is not None:
            # cached_files = list(cached_ds.reset_index()[FILES_COL])
            # cached_files = [
            #     item for sublist in cached_files for item in sublist
            # ]
            # TODO: below glob takes too much time
            # while cache loading, e.g. for gutta-visir
            # files = glob.glob(path)  # all files
            # not_cached_files = list(set(files) - set(cached_files))
            # if len(not_cached_files) == 0:  # there are no new files
            return geokube.core.dataset.Dataset(
                hcubes=cached_ds.reset_index(),
                load_files_on_persistance=load_files_on_persistance,
            )

            # there are new files we need to update the cache
            # we consider the case we only add files
            # we should consider also when files are deleted (e.g. rolling archives)
            not_cached_ds = _get_df_from_files_list(
                not_cached_files, pattern, ds_attr_names
            )
            for i in not_cached_ds.index:
                # check if index i is in the cached
                # if index exists update the datacube  (merge __FILES column and open_datacube)
                # if index does not exist add a new row
                if i in cached_ds.index:
                    new_files = [
                        *cached_ds[FILES_COL][i],
                        *not_cached_ds[FILES_COL][i],
                    ]
                    if not load_files_on_persistance:
                        cube = None
                    elif delay_read_cubes:
                        cube = dask.delayed(open_datacube)(
                            path=new_files,
                            id_pattern=id_pattern,
                            mapping=mapping,
                            **kwargs,
                        )
                    else:
                        cube = open_datacube(
                            path=new_files,
                            id_pattern=id_pattern,
                            mapping=mapping,
                            **kwargs,
                        )
                    cached_ds.loc[i] = {
                        FILES_COL: new_files,
                        DATACUBE_COL: cube,
                    }
                elif i in not_cached_ds.index:
                    not_cached_files = not_cached_ds[FILES_COL][i]
                    if delay_read_cubes:
                        cube = dask.delayed(open_datacube)(
                            path=not_cached_files,
                            id_pattern=id_pattern,
                            mapping=mapping,
                            **kwargs,
                        )
                    else:
                        cube = open_datacube(
                            path=not_cached_files,
                            id_pattern=id_pattern,
                            mapping=mapping,
                            **kwargs,
                        )
                    # To find better way
                    cached_ds.loc[i, :] = [not_cached_files, None]
                    cached_ds.loc[i, :] = [not_cached_files, cube]
                else:  # the row from cached_ds should be deleted!
                    pass

            _write_cache(cached_ds, metadata_cache_path)
            return geokube.core.dataset.Dataset(
                hcubes=cached_ds.reset_index(),
                load_files_on_persistance=load_files_on_persistance,
            )

    # if cache is not True or cache file is not available proceed with reading files in paths
    files = glob.glob(path)  # all files
    df = _get_df_from_files_list(files, pattern, ds_attr_names)
    cubes = []
    if not load_files_on_persistance:
        pass
    elif delay_read_cubes:
        for i in df.index:
            cubes.append(
                dask.delayed(open_datacube)(
                    path=df[FILES_COL][i],
                    id_pattern=id_pattern,
                    mapping=mapping,
                    **kwargs,
                )
            )
    else:
        for i in df.index:
            cubes.append(
                open_datacube(
                    path=df[FILES_COL][i],
                    id_pattern=id_pattern,
                    mapping=mapping,
                    **kwargs,
                )
            )  # we do not need to enable caching here!
    if load_files_on_persistance:
        df[DATACUBE_COL] = cubes
    else:
        df[DATACUBE_COL] = None
    # write cache if cache file does not exist and caching is true
    if metadata_caching:
        _write_cache(df, metadata_cache_path)

    return geokube.core.dataset.Dataset(
        hcubes=df.reset_index(),
        load_files_on_persistance=load_files_on_persistance,
    )
