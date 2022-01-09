import numpy as np
import pytest

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, AxisType
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.dimension import Dimension
from geokube.core.variable import Variable
from tests.fixtures import *


def test_construct_1():
    with pytest.raises(ex.HCubeTypeError):
        var = Variable(
            data="aaaa",
            name="lat",
            units="degrees_north",
            dims=[Dimension("y", AxisType.Y), Dimension("x", AxisType.X)],
        )
    data = np.random.random((10, 5))
    var = Variable(
        data=data,
        name="lat",
        units="degrees_north",
        dims=[Dimension("y", Axis(AxisType.Y)), Dimension("x", Axis(AxisType.X))],
    )
    coord = Coordinate(variable=var, axis=Axis("latitude"), bounds=None)
    assert coord.bounds is None
    assert coord.name == "lat"
    assert coord.dims_names == ("y", "x")
    assert coord.ctype is CoordinateType.DEPENDENT

    data = np.random.random(
        10,
    )
    var = Variable(
        data=data,
        name="lat",
        units="degrees_north",
        dims=Dimension("lat", Axis(AxisType.LATITUDE)),
    )
    coord = Coordinate(
        variable=var, axis=Axis("latitude"), bounds=np.random.random((10, 2))
    )
    assert coord.has_bounds
    assert coord.bounds.dims[0].atype is AxisType.LATITUDE
    assert coord.bounds.dims[1].atype is AxisType.GENERIC
    assert coord.bounds.dims[1].name == "bounds"
    assert coord.bounds.name == "lat_bounds"
    assert coord.ctype is CoordinateType.INDEPENDENT

    var = Variable(data=np.array(10), name="lat", units="degrees_north")
    coord = Coordinate(variable=var, axis=Axis("latitude"))
    assert not coord.has_bounds
    assert coord.ctype is CoordinateType.SCALAR


def test_from_xarray_dataarray(era5_rotated_netcdf_tmin2m):
    res = Coordinate.from_xarray_dataarray(era5_rotated_netcdf_tmin2m["time"])
    assert res.axis.atype is AxisType.TIME
    assert res.units.cftime_unit == era5_rotated_netcdf_tmin2m["time"].encoding["units"]
    assert res.units.calendar == era5_rotated_netcdf_tmin2m["time"].encoding["calendar"]
    assert res.bounds is None
    assert res.name == "time"


def test_from_xarray_dataset(era5_rotated_netcdf_tmin2m):
    res = Coordinate.from_xarray_dataset(era5_rotated_netcdf_tmin2m, coord_name="time")
    assert res.axis.atype is AxisType.TIME
    assert res.units.cftime_unit == era5_rotated_netcdf_tmin2m["time"].encoding["units"]
    assert res.units.calendar == era5_rotated_netcdf_tmin2m["time"].encoding["calendar"]
    assert res.bounds is not None
    assert res.bounds.dims_names == ("time", "bnds")
    assert res.bounds.dims[0].atype is AxisType.TIME
    assert res.bounds.dims[1].atype is AxisType.GENERIC
    assert res.bounds.dims[0].name == "time"
    assert res.bounds.dims[1].name == "bnds"
    assert res.name == "time"
    assert res.ctype is CoordinateType.INDEPENDENT


def test_from_xarray_dataset_2(era5_rotated_netcdf_tmin2m):
    with pytest.raises(ex.HCubeKeyError):
        _ = Coordinate.from_xarray_dataset(
            era5_rotated_netcdf_tmin2m, coord_name="some_new_lat"
        )
    res = Coordinate.from_xarray_dataset(era5_rotated_netcdf_tmin2m, coord_name="lat")
    assert res.axis.atype is AxisType.LATITUDE
    assert str(res.units) == era5_rotated_netcdf_tmin2m["lat"].attrs["units"]
    assert res.bounds is None
    assert res.name == "lat"
    assert res.dims[0].atype is AxisType.Y
    assert res.dims[0].name == "grid_latitude"
    assert res.dims[0].axis.name == "rlat"
    assert res.dims[1].atype is AxisType.X
    assert res.dims[1].name == "grid_longitude"
    assert res.dims[1].axis.name == "rlon"
    assert res.ctype is CoordinateType.DEPENDENT


def test_to_xarray_dataarray(era5_rotated_netcdf_tmin2m):
    res = Coordinate.from_xarray_dataset(era5_rotated_netcdf_tmin2m, coord_name="time")
    darr = res.to_xarray_dataarray()
    # TODO: more test


def test_to_xarray_dataset(era5_rotated_netcdf_tmin2m):
    res = Coordinate.from_xarray_dataset(era5_rotated_netcdf_tmin2m, coord_name="time")
    darr = res.to_xarray_dataset()
    assert darr.time.encoding["bounds"] == "time_bnds"
    # TODO: more test
