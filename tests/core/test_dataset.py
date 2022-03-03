import pytest

from geokube.backend.netcdf import open_dataset

from tests.fixtures import *


def test_keeping_files_after_selecting_field(dataset):
    assert "files" in dataset.data
    tp = dataset["tp"]
    assert "files" in tp.data
