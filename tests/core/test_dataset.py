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

def test_if_to_dict_produces_json_serializable(dataset, dataset_single_att):
    import json
    _ = json.dumps(dataset.to_dict())
    _ = json.dumps(dataset_single_att.to_dict())