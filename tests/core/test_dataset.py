import os
import pytest
from zipfile import ZipFile

from geokube.backend import open_dataset

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
    res_dir = dataset.persist(RES_DIR, zip_if_many=False)
    files = os.listdir(res_dir)
    assert len(files) == 4


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


def test_persisting_with_empty_one_datacube(dataset_single_att):
    clear_test_res()
    cube0 = dataset_single_att.data.iloc[0].datacube
    cube0 = cube0.sel(time=slice("1-01-01", "1-02-01"))
    dataset_single_att.data.iat[0, 2] = cube0
    path = dataset_single_att.persist(RES_DIR)
    assert ".nc" in path
    assert ".zip" not in path
    clear_test_res()


def test_persisting_with_empty_all_datacube(dataset_single_att):
    clear_test_res()
    cube0 = dataset_single_att.data.iloc[0].datacube
    cube0 = cube0.sel(time=slice("1-01-01", "1-02-01"))
    cube1 = dataset_single_att.data.iloc[1].datacube
    cube1 = cube1.sel(time=slice("1-01-01", "1-02-01"))
    dataset_single_att.data.iat[0, 2] = cube0
    dataset_single_att.data.iat[1, 2] = cube1
    with pytest.warns(
        UserWarning,
        match=r"No files were created while geokube.Dataset persisting!",
    ):
        path = dataset_single_att.persist(RES_DIR)
    assert path is None
    clear_test_res()


def test_attr_str_for_persistance(dataset):
    clear_test_res()
    attr_str = dataset._form_attr_str(dataset.data.iloc[0])
    assert attr_str == "dataset=era5-vars=2_mdt"
    for file in dataset.data.iloc[0]["files"]:
        assert (
            dataset._convert_attributes_to_file_name(
                attr_str, dataset.data.iloc[0]["files"][0]
            )
            == f"dataset=era5-vars=2_mdt-{os.path.basename(file)}"
        )
    clear_test_res()


def test_to_dict_contains_proper_keys(dataset):
    details = dataset.to_dict()
    assert isinstance(details, list)
    assert len(details) == 4
    for d in details:
        assert isinstance(d, dict)
        assert "datacube" in d
        assert "attributes" in d


def test_to_dict_contain_attributes(dataset):
    details = dataset.to_dict()
    for d in details:
        assert "attributes" in d
        assert len(d["attributes"]) == 2
        assert d["attributes"]["vars"] in {"total_precipitation", "2_mdt"}
        assert d["attributes"]["dataset"] in {"other-era5", "era5"}


def test_to_dict_contains_proper_datacube_fields_rot(dataset_rotated):
    details = dataset_rotated.to_dict()
    d = details[0]
    fields = d["datacube"]["fields"]
    assert isinstance(fields, dict)
    assert len(fields) == 1
    assert "air_temperature" in fields.keys()

    d = details[1]
    fields = d["datacube"]["fields"]
    assert isinstance(fields, dict)
    assert len(fields) == 1
    assert "lwe_thickness_of_moisture_content_of_soil_layer" in fields.keys()


def test_to_dict_contains_proper_datacube_fields(dataset):
    details = dataset.to_dict()
    d = details[0]
    fields = d["datacube"]["fields"]
    assert isinstance(fields, dict)
    assert len(fields) == 1
    assert "d2m" in fields.keys()

    d = details[1]
    fields = d["datacube"]["fields"]
    assert isinstance(fields, dict)
    assert len(fields) == 1
    assert "tp" in fields.keys()

    d = details[2]
    fields = d["datacube"]["fields"]
    assert isinstance(fields, dict)
    assert len(fields) == 1
    assert "d2m" in fields.keys()

    d = details[3]
    fields = d["datacube"]["fields"]
    assert isinstance(fields, dict)
    assert len(fields) == 1
    assert "tp" in fields.keys()
