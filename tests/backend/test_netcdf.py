import os
import timeit

import numpy as np
import pytest
from dask.delayed import Delayed

from geokube.core.field import Field
from geokube.backend import open_dataset, open_datacube

from tests import *


def test_open_datacube_and_single_time_sel():
    kube = open_datacube(
        os.path.join("tests", "resources", "rlat-rlon-tmin2m.nc")
    )
    res = kube["air_temperature"].sel(time="2007-05-02")

    assert isinstance(res, Field)
    assert np.all(
        res["time"].values.astype("datetime64[D]")
        == np.datetime64("2007-05-02")
    )


def test_open_dataset_with_load_files_on_persistance_set_to_false():
    clear_test_res()
    dset = open_dataset(
        os.path.join(
            "tests", "resources", "era5-single-levels-reanalysis_*.nc"
        ),
        os.path.join(
            "tests", "resources", "era5-single-levels-reanalysis_{var}.nc"
        ),
        load_files_on_persistance=False,
    )
    assert dset.cubes is None
    dset.persist(RES_DIR)
    assert len(os.listdir(RES_DIR)) == 2
    clear_test_res()


def test_open_geotif():
    open_datacube(
        path=os.path.join("tests", "resources", "geotiff_sample.tif")
    )
