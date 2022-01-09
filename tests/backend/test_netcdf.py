import os
import numpy as np
from dask.delayed import Delayed
import pytest
import timeit

import geokube.utils.exceptions as ex
from geokube.backend.netcdf import open_dataset


def test_open_dataset_1():
    with pytest.raises(ex.HCubeValueError):
        _ = open_dataset(
            paths="not_existing//resources//era5-single-levels*",
            pattern="not_existing//{res}//era5-single-levels-{prod}_{var}",
        )
    ds = open_dataset(
        paths="tests//resources//era5-single-levels*",
        pattern="tests//{res}//era5-single-levels-{prod}_{var}.nc",
    )
    assert len(ds.cubes) == 2
    assert np.all(ds._Dataset__data["prod"] == "reanalysis")
    assert np.all(ds._Dataset__data["res"] == "resources")
    assert set(ds._Dataset__data["var"].values) == {"2_mdt", "total_precipitation"}
    assert len(ds.cubes) == 2
    res = ds.filter({"res": "resources"})
    assert np.all(res._Dataset__data["res"] == "resources")

    res = ds.filter({"prod": "reanalysis"})
    assert len(res.cubes) == 2
    assert np.all(res._Dataset__data["prod"] == "reanalysis")

    res = ds.filter({"var": "total_precipitation"})
    assert len(res.cubes) == 1
    assert np.all(res._Dataset__data["var"] == "total_precipitation")

    res = ds.filter({"var": "2_mdt"})
    assert len(res.cubes) == 1
    assert np.all(res._Dataset__data["var"] == "2_mdt")

    res = ds.filter({"var": ["2_mdt", "total_precipitation"], "prod": "reanalysis"})
    assert len(res.cubes) == 2
    assert set(ds._Dataset__data["var"].values) == {"2_mdt", "total_precipitation"}
    assert np.all(res._Dataset__data["prod"] == "reanalysis")


def test_open_dataset_with_cache():
    ds = open_dataset(
        paths="tests//resources//era5-single-levels*",
        pattern="tests//{res}//era5-single-levels-{prod}_{var}.nc",
        parallel=True,
        metadata_cache_path="tests//resources//cache-path",
        delay_read_cubes=True,
    )
    assert list(map(lambda dc: isinstance(dc, Delayed), ds.data["datacube"]))

    with pytest.raises(ex.HCubeValueError):
        _ = open_dataset(
            paths="tests//resources//era5-single-levels*",
            pattern="tests//{res}//era5-single-levels-{prod}_{var}.nc",
            parallel=True,
            metadata_caching=True,
            delay_read_cubes=True,
        )

    ds = open_dataset(
        paths="tests//resources//era5-single-levels*",
        pattern="tests//{res}//era5-single-levels-{prod}_{var}.nc",
        parallel=True,
        metadata_cache_path="tests//resources//cache-path",
        metadata_caching=True,
        delay_read_cubes=True,
    )
    assert os.path.exists("tests//resources//cache-path")
    try:
        os.remove("tests//resources//cache-path")
    except:
        pass

    assert len(ds["tp"].data) == 0  # as all are delayes
    ds = open_dataset(
        paths="tests//resources//era5-single-levels*",
        pattern="tests//{res}//era5-single-levels-{prod}_{var}.nc",
        parallel=True,
    )
    assert len(ds["tp"].data) == 1
    assert len(ds[["tp", "d2m"]].data) == 2

    ds = open_dataset(
        paths="tests//resources//era5-single-levels*",
        pattern="tests//{res}//era5-single-levels-{prod}_{}.nc",
        parallel=True,
    )
    assert len(ds["tp"].data) == 1
    assert len(ds[["tp", "d2m"]].data) == 1  # all vars in single attributes combnation
