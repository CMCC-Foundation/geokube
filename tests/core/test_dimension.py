import cf_units as cf
import pytest

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, Axis
from geokube.core.dimension import Dimension
from tests.fixtures import *


def test_construct_1():
    axis = Axis("latitude", "lat1")
    dim = Dimension(name="dim1_lat", axis=axis)
    assert axis.name == "lat1"
    assert dim.atype is Axis.LATITUDE
    assert dim.default_units == cf.Unit("degrees_north")

    axis = Axis("longitude")
    dim = Dimension(name="dim1_lat", axis=axis)
    assert axis.name == "longitude"
    assert dim.atype is Axis.LONGITUDE
    assert dim.default_units == cf.Unit("degrees_east")


def test_from_xarray_dataarray(era5_netcdf, era5_rotated_netcdf_tmin2m, nemo_ocean_16):
    with pytest.raises(ex.HCubeTypeError):
        Dimension.from_xarray_dataarray(era5_netcdf)
    dim = Dimension.from_xarray_dataarray(era5_netcdf["time"])
    assert dim.atype is Axis.TIME
    assert dim.axis.default_units == cf.Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )
    assert dim.name == "time"
    assert dim.axis.name == "time"

    dim = Dimension.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["rlat"])
    assert dim.atype is Axis.Y
    assert dim.axis.default_units == cf.Unit("m")
    assert dim.name == "grid_latitude"
    assert dim.axis.name == "rlat"

    dim = Dimension.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["lat"])
    assert dim.atype is Axis.LATITUDE
    assert dim.axis.default_units == cf.Unit("degrees_north")
    assert dim.name == "latitude"
    assert dim.axis.name == "lat"

    dim = Dimension.from_xarray_dataarray(nemo_ocean_16["depthv"])
    assert dim.atype is Axis.VERTICAL
    assert dim.axis.default_units == cf.Unit("m")
    assert dim.name == "Z"
    assert dim.axis.name == "depthv"

    dim = Dimension.from_xarray_dataarray(nemo_ocean_16["nav_lon"])
    assert dim.atype is Axis.LONGITUDE
    assert dim.axis.default_units == cf.Unit("degrees_east")
    assert dim.name == "longitude"
    assert dim.axis.name == "nav_lon"
