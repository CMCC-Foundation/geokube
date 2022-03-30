import pytest

from geokube.backend.netcdf import open_dataset

from tests.fixtures import *


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


def test_to_dict(dataset, dataset_single_att):
    dict_dset = dataset.to_dict()
    assert len(dict_dset.keys()) == 4
    keys = dict_dset.keys()
    assert ("era5", "2_mdt") in keys
    assert ("era5", "total_precipitation") in keys
    assert ("other-era5", "2_mdt") in keys
    assert ("other-era5", "total_precipitation") in keys
    dict_dset = dataset_single_att.to_dict()
    keys = dict_dset.keys()
    assert len(dict_dset.keys()) == 2
    assert ("era5",) in keys
    assert ("other-era5",) in keys
