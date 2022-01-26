import numpy as np
import pandas as pd
import pytest

import geokube.core.coord_system as crs
import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, Axis
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.dimension import Dimension
from geokube.core.domain import Domain, DomainType
from geokube.core.variable import Variable
from tests.fixtures import *


def test_construct_1():
    x_axis = Axis(Axis.X)
    x_axis_dim = Dimension("x", x_axis)
    y_axis = Axis(Axis.Y)
    y_axis_dim = Dimension("y", y_axis)
    lat_axis = Axis(name="latT", atype=Axis.LATITUDE)
    lon_axis = Axis(name="longitude", atype=Axis.LONGITUDE)

    x = Coordinate(
        variable=Variable(name="x", data=np.linspace(-10, 10, 100), dims=x_axis_dim),
        axis=x_axis,
    )
    y = Coordinate(
        variable=Variable(name="y", data=np.linspace(40, 60, 40), dims=y_axis_dim),
        axis=y_axis,
    )

    lat = Coordinate(
        variable=Variable(
            name="lat", data=np.random.random((100, 40)), dims=(x_axis_dim, y_axis_dim)
        ),
        axis=lat_axis,
    )  # lat_data is a 2-dimensional numpy array
    lon = Coordinate(
        variable=Variable(
            name="lon", data=np.random.random((100, 40)), dims=(x_axis_dim, y_axis_dim)
        ),
        axis=lon_axis,
    )  # lon

    time_axis = Axis(Axis.TIME)
    time_axis_dim = Dimension(name="time", axis=time_axis)
    time = Coordinate(
        variable=Variable(
            name="time",
            data=np.array(pd.date_range("2001-01-01", "2001-05-10", freq="H")),
            dims=time_axis_dim,
        ),
        axis=time_axis,
    )

    dom = Domain(
        coordinates=[x, y, lat, lon, time],
        crs=crs.RotatedGeogCS(
            grid_north_pole_latitude=10, grid_north_pole_longitude=-25
        ),
    )

    assert Axis.LATITUDE in dom._Axis_to_name
    assert dom[Axis.LATITUDE].ctype is CoordinateType.DEPENDENT
    assert dom[Axis.LATITUDE].name == "lat"
    assert Axis.LONGITUDE in dom._Axis_to_name
    assert dom[Axis.LONGITUDE].ctype is CoordinateType.DEPENDENT
    assert dom[Axis.LONGITUDE].name == "lon"
    assert Axis.TIME in dom._Axis_to_name
    assert dom[Axis.TIME].ctype is CoordinateType.INDEPENDENT
    assert Axis.X in dom._Axis_to_name
    assert dom[Axis.X].ctype is CoordinateType.INDEPENDENT
    assert Axis.Y in dom._Axis_to_name
    assert dom[Axis.Y].ctype is CoordinateType.INDEPENDENT

    assert dom.domtype is None
    assert dom.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=10, grid_north_pole_longitude=-25
    )
    assert id(dom[Axis.LATITUDE].dims[0].atype) == id(dom[Axis.LONGITUDE].dims[0].atype)
    assert dom[Axis.LATITUDE].axis.name == "latT"
    assert dom[Axis.LATITUDE].axis.atype is Axis.LATITUDE
    assert dom[Axis.LONGITUDE].axis.name == "longitude"
    assert dom[Axis.LONGITUDE].axis.atype is Axis.LONGITUDE
    assert id(dom[Axis.LATITUDE].dims[0]) == id(dom[Axis.LONGITUDE].dims[0])
    assert id(dom[Axis.LATITUDE].dims[1]) == id(dom[Axis.LONGITUDE].dims[1])

    # TODO: more tests


def test_from_xarray_dataarray(era5_globe_netcdf):
    dom = Domain.from_xarray_dataarray(era5_globe_netcdf["tp"])
    assert "time" in dom
    assert "latitude" in dom
    assert "longitude" in dom
    assert dom["time"].axis.atype is Axis.TIME
    assert dom["latitude"].axis.atype is Axis.LATITUDE
    assert dom["longitude"].axis.atype is Axis.LONGITUDE
    assert dom["time"].ctype is CoordinateType.INDEPENDENT
    assert dom["latitude"].ctype is CoordinateType.INDEPENDENT
    assert dom["longitude"].ctype is CoordinateType.INDEPENDENT
    assert isinstance(dom.crs, crs.RegularLatLon)
    assert dom.crs == crs.RegularLatLon()
    assert np.all(dom["time"].data == era5_globe_netcdf["time"].values)
    assert np.all(dom["latitude"].data == era5_globe_netcdf["latitude"].values)
    assert np.all(dom["longitude"].data == era5_globe_netcdf["longitude"].values)


def test_from_xarray_dataarray_2(era5_rotated_netcdf_wso):
    dom = Domain.from_xarray_dataarray(era5_rotated_netcdf_wso["W_SO"])
    assert "time" in dom
    assert "lat" in dom
    assert "lon" in dom
    assert "x" in dom
    assert "y" in dom
    assert "soil1" in dom

    assert dom["time"].ctype is CoordinateType.INDEPENDENT
    assert dom["time"].axis.atype is Axis.TIME
    assert dom["time"].variable.dims[0].name == "time"
    assert dom["time"].variable.dims[0].axis.atype is Axis.TIME

    assert dom["soil1"].ctype is CoordinateType.INDEPENDENT
    assert dom["soil1"].axis.atype is Axis.VERTICAL
    assert dom["soil1"].variable.dims[0].name == "depth"  # from standard_name
    # assert dom["soil1"].variable._cf_encoding == era5_rotated_netcdf_wso["soil1"].attrs
    assert dom["soil1"].variable.dims[0].axis.atype is Axis.VERTICAL

    assert dom["lat"].ctype is CoordinateType.DEPENDENT
    assert dom["lat"].axis.atype is Axis.LATITUDE
    assert dom["lat"].variable.dims[0].name == "grid_latitude"
    assert dom["lat"].variable.dims[0].atype is Axis.Y
    assert dom["lat"].variable.dims[1].name == "grid_longitude"
    assert dom["lat"].variable.dims[1].atype is Axis.X

    assert dom["lon"].ctype is CoordinateType.DEPENDENT
    assert dom["lon"].axis.atype is Axis.LONGITUDE
    assert dom["lon"].variable.dims[0].name == "grid_latitude"
    assert dom["lon"].variable.dims[0].atype is Axis.Y
    assert dom["lon"].variable.dims[1].name == "grid_longitude"
    assert dom["lon"].variable.dims[1].atype is Axis.X

    assert dom["rlat"].ctype is CoordinateType.INDEPENDENT
    assert dom["rlat"].axis.atype is Axis.Y
    assert dom["rlat"].variable.dims[0].name == "grid_latitude"
    assert dom["rlat"].variable.dims[0].atype is Axis.Y

    assert dom["rlon"].ctype is CoordinateType.INDEPENDENT
    assert dom["rlon"].axis.atype is Axis.X
    assert dom["rlon"].variable.dims[0].name == "grid_longitude"
    assert dom["rlon"].variable.dims[0].atype is Axis.X

    assert isinstance(dom.crs, crs.RotatedGeogCS)


def test_from_xarray_dataarray_3(nemo_ocean_16):
    dom = Domain.from_xarray_dataarray(nemo_ocean_16["vt"])
    assert "nav_lat" in dom
    assert "nav_lon" in dom
    assert "time_counter" in dom
    assert "time_centered" in dom
    assert "x" in dom
    assert "y" in dom
    assert "depthv" in dom
    assert dom["time_counter"].bounds is None

    dom = Domain.from_xarray_dataset(nemo_ocean_16, field_name="vt")
    assert "nav_lat" in dom
    assert "nav_lon" in dom
    assert "time_counter" in dom
    assert "time_centered" in dom
    assert "x" in dom
    assert "y" in dom
    assert "depthv" in dom
    assert dom["time_counter"].bounds is not None
    assert dom["time_centered"].bounds is not None
    assert dom["depthv"].bounds is not None
    assert dom["nav_lat"].bounds is not None
    assert dom["nav_lon"].bounds is not None

    assert dom["nav_lat"].ctype is CoordinateType.DEPENDENT
    assert dom["nav_lat"].axis.atype is Axis.LATITUDE
    assert dom["nav_lat"].variable.dims[0].name == "y"
    assert dom["nav_lat"].variable.dims[1].name == "x"
    assert dom["nav_lat"].variable.dims[0].atype is Axis.Y
    assert dom["nav_lat"].variable.dims[1].atype is Axis.X

    assert dom["nav_lon"].ctype is CoordinateType.DEPENDENT
    assert dom["nav_lon"].axis.atype is Axis.LONGITUDE
    assert dom["nav_lon"].variable.dims[0].name == "y"
    assert dom["nav_lon"].variable.dims[1].name == "x"
    assert dom["nav_lon"].variable.dims[0].atype is Axis.Y
    assert dom["nav_lon"].variable.dims[1].atype is Axis.X

    assert dom["x"].ctype is CoordinateType.INDEPENDENT
    assert dom["x"].axis.atype is Axis.X
    assert dom["x"].variable.dims[0].name == "x"
    assert dom["x"].variable.dims[0].atype is Axis.X

    assert dom["y"].ctype is CoordinateType.INDEPENDENT
    assert dom["y"].axis.atype is Axis.Y
    assert dom["y"].variable.dims[0].name == "y"
    assert dom["y"].variable.dims[0].atype is Axis.Y

    # assert isinstance(dom.crs, ???)


def test_process_time_combo(era5_netcdf):
    dom = Domain.from_xarray_dataset(era5_netcdf, field_name="tp")
    inds = dom._process_time_combo({"year": 2020, "month": 6, "hour": 12})
    res = era5_netcdf.isel(inds)

    assert np.all(res.time.dt.hour == 12)
    assert np.all(res.time.dt.month == 6)
    assert np.all(res.time.dt.year == 2020)

    inds = dom._process_time_combo({"day": [1, 28, 7, 2], "hour": [4, 22]})
    res = era5_netcdf.isel(inds)
    assert np.all((res.time.dt.hour == 4) | (res.time.dt.hour == 22))
    assert np.all(
        (res.time.dt.day == 1)
        | (res.time.dt.day == 28)
        | (res.time.dt.day == 7)
        | (res.time.dt.day == 2)
    )


def test_compute_bounds(era5_netcdf, nemo_ocean_16):
    dom = Domain.from_xarray_dataset(era5_netcdf, field_name="tp")
    assert not dom[Axis.LATITUDE].has_bounds
    assert not dom[Axis.LONGITUDE].has_bounds
    dom.compute_bounds(coord="latitude")
    assert dom[Axis.LATITUDE].has_bounds
    assert dom[Axis.LATITUDE].bounds.data.ndim == 2
    assert np.min(dom[Axis.LATITUDE].bounds.data) < era5_netcdf["latitude"].values.min()
    assert np.max(dom[Axis.LATITUDE].bounds.data) > era5_netcdf["latitude"].values.max()
    assert dom[Axis.LATITUDE].bounds.dims[0].atype is Axis.LATITUDE
    assert dom[Axis.LATITUDE].bounds.dims[1].atype is Axis.GENERIC
    assert not dom[Axis.LONGITUDE].has_bounds

    with pytest.raises(ex.HCubeValueError):
        dom = Domain.from_xarray_dataarray(nemo_ocean_16["vt"])
        dom.compute_bounds(coord="nav_lat")
