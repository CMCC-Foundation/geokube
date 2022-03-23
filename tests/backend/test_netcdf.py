import os
import timeit

import numpy as np
import pytest
from dask.delayed import Delayed


from geokube.core.field import Field
from geokube.backend.netcdf import open_dataset, open_datacube


def test_open_datacube_and_single_time_sel():
    kube = open_datacube(os.path.join("tests", "resources", "rlat-rlon-tmin2m.nc"))
    res = kube["air_temperature"].sel(time="2007-05-02")

    assert isinstance(res, Field)
    assert np.all(
        res["time"].values.astype("datetime64[D]") == np.datetime64("2007-05-02")
    )
