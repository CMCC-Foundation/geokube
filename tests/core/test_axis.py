import pytest
from cf_units import Unit

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, AxisType
from tests.fixtures import *


def test_axis_type():
    assert len(AxisType.get_available_names()) == 12
    assert "time" in AxisType.get_available_names()
    assert "generic" in AxisType.get_available_names()
    assert AxisType.LATITUDE.default_units == Unit("degrees_north")
    assert AxisType.TIME.name == "time"
    assert AxisType("aaa") is AxisType.GENERIC
    assert AxisType.parse_type("latitude") is AxisType.LATITUDE
    assert AxisType.parse_type("lat") is AxisType.LATITUDE
    assert AxisType.parse_type("rlat") is AxisType.Y
    assert AxisType.parse_type("x").default_units == Unit("m")
    assert AxisType.parse_type("depth") is AxisType.VERTICAL
    assert AxisType.parse_type("time").default_units == Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )
    assert AxisType.generic() is AxisType.GENERIC
    assert AxisType.generic().default_units == Unit("unknown")


def test_axis_1():
    with pytest.raises(ex.HCubeTypeError):
        Axis.from_xarray_dataarray("some_object")


def test_axis_2(era5_netcdf, era5_rotated_netcdf_tmin2m, nemo_ocean_16):
    at = Axis.from_xarray_dataarray(era5_netcdf["longitude"])
    assert at.name == "longitude"
    assert at.atype is AxisType.LONGITUDE

    at = Axis.from_xarray_dataarray(era5_netcdf["time"])
    assert at.name == "time"
    assert at.atype is AxisType.TIME
    assert at.atype.default_units == Unit(
        "hours since 1970-01-01", calendar="gregorian"
    )

    at = Axis.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["rlat"])
    assert at.name == "rlat"
    assert at.atype is AxisType.Y

    at = Axis.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["rlon"])
    assert at.name == "rlon"
    assert at.atype is AxisType.X

    at = Axis.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["height_2m"])
    assert at.name == "height_2m"
    assert at.atype is AxisType.VERTICAL

    at = Axis.from_xarray_dataarray(nemo_ocean_16["nav_lat"])
    assert at.name == "nav_lat"
    assert at.atype is AxisType.LATITUDE

    at = Axis.from_xarray_dataarray(nemo_ocean_16["nav_lon"])
    assert at.name == "nav_lon"
    assert at.atype is AxisType.LONGITUDE

    at = Axis.from_xarray_dataarray(nemo_ocean_16["depthv"])
    assert at.name == "depthv"
    assert at.atype is AxisType.VERTICAL
