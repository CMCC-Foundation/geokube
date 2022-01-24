import pytest
from cf_units import Unit

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, Axis
from tests.fixtures import *


def test_axis_type():
    assert len(Axis.get_available_names()) == 12
    assert "time" in Axis.get_available_names()
    assert "generic" in Axis.get_available_names()
    assert Axis.LATITUDE.default_units == Unit("degrees_north")
    assert Axis.TIME.name == "time"
    assert Axis("aaa") is Axis.GENERIC
    assert Axis.parse_type("latitude") is Axis.LATITUDE
    assert Axis.parse_type("lat") is Axis.LATITUDE
    assert Axis.parse_type("rlat") is Axis.Y
    assert Axis.parse_type("x").default_units == Unit("m")
    assert Axis.parse_type("depth") is Axis.VERTICAL
    assert Axis.parse_type("time").default_units == Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )
    assert Axis.generic() is Axis.GENERIC
    assert Axis.generic().default_units == Unit("unknown")


def test_axis_1():
    with pytest.raises(ex.HCubeTypeError):
        Axis.from_xarray_dataarray("some_object")


def test_axis_2(era5_netcdf, era5_rotated_netcdf_tmin2m, nemo_ocean_16):
    at = Axis.from_xarray_dataarray(era5_netcdf["longitude"])
    assert at.name == "longitude"
    assert at.atype is Axis.LONGITUDE

    at = Axis.from_xarray_dataarray(era5_netcdf["time"])
    assert at.name == "time"
    assert at.atype is Axis.TIME
    assert at.atype.default_units == Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )

    at = Axis.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["rlat"])
    assert at.name == "rlat"
    assert at.atype is Axis.Y

    at = Axis.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["rlon"])
    assert at.name == "rlon"
    assert at.atype is Axis.X

    at = Axis.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["height_2m"])
    assert at.name == "height_2m"
    assert at.atype is Axis.VERTICAL

    at = Axis.from_xarray_dataarray(nemo_ocean_16["nav_lat"])
    assert at.name == "nav_lat"
    assert at.atype is Axis.LATITUDE

    at = Axis.from_xarray_dataarray(nemo_ocean_16["nav_lon"])
    assert at.name == "nav_lon"
    assert at.atype is Axis.LONGITUDE

    at = Axis.from_xarray_dataarray(nemo_ocean_16["depthv"])
    assert at.name == "depthv"
    assert at.atype is Axis.VERTICAL
