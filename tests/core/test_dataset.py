import os
import pytest
from zipfile import ZipFile

from geokube.backend.netcdf import open_dataset

from tests.fixtures import *
from tests import RES_PATH, RES_DIR, clear_test_res


def test_keeping_files_after_selecting_field(dataset):
    assert "files" in dataset.data
    tp = dataset["tp"]
    assert "files" in tp.data


def test_select_fields_by_name(dataset_idpattern):
    d2m = dataset_idpattern["std_K"]
    assert len(d2m) == 1
    tp = dataset_idpattern["std_m"]
    assert len(tp) == 1


def test_select_fields_by_ncvar(dataset_idpattern):
    d2m = dataset_idpattern["d2m"]
    assert len(d2m) == 1
    tp = dataset_idpattern["tp"]
    assert len(tp) == 1


def test_if_to_dict_produces_json_serializable(dataset, dataset_single_att):
    import json

    _ = json.dumps(dataset.to_dict())
    _ = json.dumps(dataset_single_att.to_dict())


def test_nbytes_estimation(dataset_single_att):
    import os

    clear_test_res()
    d2m = dataset_single_att.sel(
        time={"day": [5, 8], "hour": [1, 2, 3, 12, 13, 14, 22, 23]}
    ).geobbox(north=44, south=39, east=12, west=7)
    precomputed_nbytes = d2m.nbytes
    assert precomputed_nbytes != 0
    os.mkdir(RES_DIR)
    d2m.persist(RES_DIR)
    postcomputed_nbytes = sum(
        [
            os.path.getsize(os.path.join(RES_DIR, f))
            for f in os.listdir(RES_DIR)
        ]
    )
    clear_test_res()
    assert (
        (precomputed_nbytes - postcomputed_nbytes) / postcomputed_nbytes
    ) < 0.25  # TODO: maybe estimation should be more precise


def test_persist_and_return_paths_no_zipping(dataset):
    clear_test_res()
    _ = dataset.persist(RES_DIR, zip_if_many=False)
    files = os.listdir(RES_DIR)
    assert len(files) == 4
    assert "era5-single-levels-reanalysis_2_mdt.nc" in files
    assert "era5-single-levels-reanalysis_total_precipitation.nc" in files
    assert "other-era5-single-levels-reanalysis_2_mdt.nc" in files
    assert (
        "other-era5-single-levels-reanalysis_total_precipitation.nc" in files
    )


def test_persist_and_return_paths_with_zipping(dataset):
    clear_test_res()
    res = dataset.persist(RES_DIR, zip_if_many=True)
    files = os.listdir(RES_DIR)
    assert len(files) == 1
    assert os.path.join(RES_DIR, files[0]) == res
    assert res.endswith(".zip")
    with ZipFile(res, "r") as archive:
        names = archive.namelist()
    for n in names:
        assert RES_DIR not in n
    assert len(names) == 4
    clear_test_res()
