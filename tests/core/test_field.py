import numpy as np
import pytest
import xarray as xr
import pandas as pd
import cf_units as cf

import geokube.core.coord_system as crs
import geokube.utils.exceptions as ex
from geokube.backend.netcdf import open_datacube, open_dataset
from geokube.core.axis import Axis, AxisType
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.datacube import DataCube
from geokube.core.domain import Domain
from geokube.core.enums import LongitudeConvention, MethodType
from geokube.core.field import Field
from geokube.core.unit import Unit
from geokube.core.variable import Variable
from geokube.utils import util_methods
from tests import RES_PATH, clear_test_res
from tests.fixtures import *


def test_from_xarray_with_point_domain(era5_point_domain):
    field = Field.from_xarray(era5_point_domain, ncvar="W_SO")
    assert "points" in field.domain["longitude"].dim_names
    assert "points" in field.domain["latitude"].dim_names
    assert "points" in field.domain[Axis("x")].dim_names
    assert "points" in field.domain[AxisType.Y].dim_names

    dset = field.to_xarray(encoding=False)
    assert "latitude" in dset.coords
    assert "lat" not in dset.coords
    assert "longitude" in dset.coords
    assert "lon" not in dset.coords
    assert "points" in dset.dims
    assert "points" in dset["grid_latitude"].dims
    assert "points" in dset["grid_longitude"].dims
    assert "points" in dset["latitude"].dims
    assert "points" in dset["longitude"].dims


def test_from_xarray_rotated_pole(era5_rotated_netcdf):
    field = Field.from_xarray(era5_rotated_netcdf, ncvar="TMIN_2M")

    assert field.name == "air_temperature"
    assert field.ncvar == "TMIN_2M"
    assert "height_2m" in field.domain
    assert "lon" in field.domain
    assert "lat" in field.domain
    assert "rlat" in field.domain
    assert "rlon" in field.domain
    assert field.domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )


def test_to_xarray_rotated_pole_without_encoding(era5_rotated_netcdf):
    field = Field.from_xarray(era5_rotated_netcdf, ncvar="TMIN_2M")
    xr_res = field.to_xarray(encoding=False)
    assert "air_temperature" in xr_res.data_vars
    assert "height" in xr_res.coords
    assert "time" in xr_res.coords
    assert "longitude" in xr_res.coords
    assert "latitude" in xr_res.coords
    assert "grid_latitude" in xr_res.coords
    assert "grid_longitude" in xr_res.coords
    assert "crs" in xr_res.coords
    assert "time" in xr_res["air_temperature"].dims
    assert "grid_latitude" in xr_res["air_temperature"].dims
    assert "grid_longitude" in xr_res["air_temperature"].dims
    assert "grid_mapping" in xr_res["air_temperature"].encoding
    assert xr_res["air_temperature"].encoding["grid_mapping"] == "crs"
    assert set(xr_res["air_temperature"].encoding["coordinates"].split(" ")) == {
        "height",
        "latitude",
        "longitude",
    }
    assert "cell_methods" in xr_res["air_temperature"].attrs
    assert (
        xr_res["air_temperature"].attrs["cell_methods"]
        == era5_rotated_netcdf["TMIN_2M"].attrs["cell_methods"]
    )


def test_to_xarray_rotated_pole_with_encoding(era5_rotated_netcdf):
    field = Field.from_xarray(era5_rotated_netcdf, ncvar="TMIN_2M")
    xr_res = field.to_xarray(encoding=True)
    assert "TMIN_2M" in xr_res.data_vars
    assert "height_2m" in xr_res.coords
    assert "lon" in xr_res.coords
    assert "lat" in xr_res.coords
    assert "rlat" in xr_res.coords
    assert "rlon" in xr_res.coords
    assert "crs" in xr_res.coords
    assert "time" in xr_res["TMIN_2M"].dims
    assert "rlat" in xr_res["TMIN_2M"].dims
    assert "rlon" in xr_res["TMIN_2M"].dims
    assert "grid_mapping" in xr_res["TMIN_2M"].encoding
    assert xr_res["TMIN_2M"].encoding["grid_mapping"] == "crs"
    assert set(xr_res["TMIN_2M"].encoding["coordinates"].split(" ")) == {
        "height_2m",
        "lat",
        "lon",
    }
    assert "cell_methods" in xr_res["TMIN_2M"].attrs
    assert (
        xr_res["TMIN_2M"].attrs["cell_methods"]
        == era5_rotated_netcdf["TMIN_2M"].attrs["cell_methods"]
    )


def test_from_xarray_rotated_pole_with_mapping_and_id_pattern(era5_rotated_netcdf):
    field = Field.from_xarray(
        era5_rotated_netcdf,
        ncvar="TMIN_2M",
        id_pattern="prefix:{standard_name}",
        mapping={"rlat": {"name": "myrlat"}},
    )
    assert field.name == "prefix:air_temperature"
    assert field.ncvar == "TMIN_2M"
    assert "prefix:height" in field.domain._coords
    assert "prefix:longitude" in field.domain._coords
    assert "prefix:latitude" in field.domain._coords
    assert "myrlat" in field.domain._coords
    assert "prefix:grid_latitude" not in field.domain._coords
    assert "prefix:grid_longitude" in field.domain._coords
    assert field.domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )

    xr_res = field.to_xarray(encoding=False)
    assert "prefix:air_temperature" in xr_res.data_vars
    assert "myrlat" in xr_res.coords
    assert "prefix:grid_latitude" not in xr_res.coords
    assert "prefix:grid_longitude" in xr_res.coords
    assert "prefix:time" in xr_res.coords

    xr_res = field.to_xarray(encoding=True)
    assert "prefix:air_temperature" not in xr_res.data_vars
    assert "TMIN_2M" in xr_res.data_vars
    assert "myrlat" not in xr_res.coords
    assert "rlat" in xr_res.coords
    assert "prefix:grid_longitude" not in xr_res.coords
    assert "rlon" in xr_res.coords
    assert "prefix:time" not in xr_res.coords
    assert "time" in xr_res.coords


def test_from_xarray_curvilinear_grid(nemo_ocean_16):
    field = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    assert field.name == "vt"
    assert field.ncvar == "vt"
    assert field.units == Unit("degree_C m/s")
    assert str(field.cell_methods) == nemo_ocean_16["vt"].attrs["cell_methods"]
    assert "time" in field.domain._coords
    assert "depthv" in field.domain._coords
    assert "latitude" in field.domain._coords
    assert "longitude" in field.domain._coords
    assert "x" not in field.domain._coords
    assert "y" not in field.domain._coords
    assert isinstance(field.domain.crs, crs.CurvilinearGrid)

    assert field.domain["longitude"].dims[0].type == AxisType.Y
    assert field.domain["longitude"].dims[1].type == AxisType.X

    xr_res = field.to_xarray()
    assert xr_res["vt"].encoding["grid_mapping"] == "crs"
    assert "crs" in xr_res.coords


def test_from_xarray_regular_latlon(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    assert field._id_pattern is None
    assert field._mapping is None
    assert field.name == "tp"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")


def test_from_xarray_regular_latlon_with_id_pattern(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp", id_pattern="{__ddsapi_name}")
    assert field._id_pattern == "{__ddsapi_name}"
    assert field._mapping is None
    assert field.name == "total_precipitation"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")


def test_from_xarray_regular_latlon_with_complex_id_pattern(era5_netcdf):
    field = Field.from_xarray(
        era5_netcdf, ncvar="tp", id_pattern="{units}__{long_name}"
    )
    assert field.name == "m__Total precipitation"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")


def test_from_xarray_regular_latlon_with_mapping(era5_netcdf):
    field = Field.from_xarray(
        era5_netcdf, ncvar="tp", mapping={"tp": {"name": "tot_prep"}}
    )
    assert field._id_pattern is None
    assert field._mapping == {"tp": {"name": "tot_prep"}}
    assert field.name == "tot_prep"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")


def test_geobbox_regular_latlon(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    with pytest.raises(ValueError):
        __ = tp.geobbox(north=10, south=-10, west=50, east=80, top=5, bottom=10)

    res = tp.geobbox(north=10, south=-10, west=50, east=80)
    assert np.all(res["latitude"].values <= 10)
    assert np.all(res["latitude"].values >= -10)
    assert np.all(res.latitude.values <= 10)
    assert np.all(res.latitude.values >= -10)

    assert np.all(res["longitude"].values <= 80)
    assert np.all(res["longitude"].values >= 50)
    assert np.all(res.longitude.values <= 80)
    assert np.all(res.longitude.values >= 50)

    dset = res.to_xarray(True)
    assert np.all(dset.latitude <= 10)
    assert np.all(dset.latitude >= -10)
    assert dset.latitude.attrs["units"] == "degrees_north"

    assert np.all(dset.longitude <= 80)
    assert np.all(dset.longitude >= 50)
    assert dset.longitude.attrs["units"] == "degrees_east"


def test_geobbox_regular_latlon_2(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")

    res = tp.geobbox(north=10, south=-10, west=-20, east=20)
    assert np.all(res["latitude"].values <= 10)
    assert np.all(res["latitude"].values >= -10)
    assert np.all(res.latitude.values <= 10)
    assert np.all(res.latitude.values >= -10)

    assert np.all(res["longitude"].values <= 20)
    assert np.all(res["longitude"].values >= -20)
    assert np.all(res.longitude.values <= 20)
    assert np.all(res.longitude.values >= -20)

    dset = res.to_xarray(True)
    assert np.all(dset.latitude <= 10)
    assert np.all(dset.latitude >= -10)
    assert dset.latitude.attrs["units"] == "degrees_north"

    assert np.all(dset.longitude <= 20)
    assert np.all(dset.longitude >= -20)
    assert dset.longitude.attrs["units"] == "degrees_east"


def test_geobbox_regular_latlon_3(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")

    res = tp.geobbox(north=10, west=2)
    assert np.all(res["latitude"].values <= 10)
    assert np.all(res.latitude.values <= 10)

    assert np.all(res["longitude"].values >= 2)
    assert np.all(res.longitude.values >= 2)

    dset = res.to_xarray(True)
    assert np.all(dset.latitude <= 10)
    assert dset.latitude.attrs["units"] == "degrees_north"

    assert np.all(dset.longitude >= 2)
    assert dset.longitude.attrs["units"] == "degrees_east"


def test_geobbox_regular_latlon_4(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")

    with pytest.raises(ValueError, match="'top' and 'bottom' must be None"):
        tp.geobbox(top=2)

    res = tp.geobbox(north=10)
    assert np.all(res["latitude"].values <= 10)
    assert np.all(res.latitude.values <= 10)

    dset = res.to_xarray(True)
    assert np.all(dset.latitude <= 10)
    assert dset.latitude.attrs["units"] == "degrees_north"


def test_geobbox_rotated_pole(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")
    assert wso.latitude.name == "latitude"
    assert wso.latitude.ncvar == "lat"
    assert wso.longitude.name == "longitude"
    assert wso.longitude.ncvar == "lon"

    res = wso.geobbox(north=40, south=38, west=16, east=19)
    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "lon"
    assert res.domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= 38) / W > 0.95
    assert np.sum(res.latitude.values <= 40) / W > 0.95
    assert np.sum(res.longitude.values >= 16) / W > 0.95
    assert np.sum(res.longitude.values <= 19) / W > 0.95

    dset = res.to_xarray(encoding=True)
    assert "W_SO" in dset.data_vars
    assert "rlat" in dset.coords
    assert "rlon" in dset.coords
    assert "lat" in dset
    assert dset.lat.attrs["units"] == "degrees_north"
    assert "lon" in dset
    assert dset.lon.attrs["units"] == "degrees_east"
    assert "crs" in dset.coords

    dset = res.to_xarray(False)
    assert "lwe_thickness_of_moisture_content_of_soil_layer" in dset.data_vars
    assert "latitude" in dset
    assert dset.latitude.attrs["units"] == "degrees_north"
    assert "longitude" in dset
    assert dset.longitude.attrs["units"] == "degrees_east"
    assert "crs" in dset.coords


def test_geobbox_curvilinear_grid(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    assert vt.latitude.name == "latitude"
    assert vt.longitude.name == "longitude"
    assert vt.latitude.ncvar == "nav_lat"
    assert vt.longitude.ncvar == "nav_lon"

    res = vt.geobbox(north=-19, south=-22, west=-115, east=-110)
    assert vt.latitude.name == "latitude"
    assert vt.longitude.name == "longitude"
    assert vt.latitude.ncvar == "nav_lat"
    assert vt.longitude.ncvar == "nav_lon"
    assert res.domain.crs == crs.CurvilinearGrid()

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= -22) / W > 0.95
    assert np.sum(res.latitude.values <= -19) / W > 0.95
    assert np.sum(res.longitude.values >= -115) / W > 0.95
    assert np.sum(res.longitude.values <= -110) / W > 0.95


def test_timecombo_single_hour(era5_netcdf):
    tp = Field.from_xarray(era5_netcdf, ncvar="tp")
    res = tp.sel(time={"year": 2020, "day": [1, 6, 10], "hour": 5})
    dset = res.to_xarray(True)
    assert np.all(
        (dset.time.dt.day == 1) | (dset.time.dt.day == 6) | (dset.time.dt.day == 10)
    )
    assert np.all(dset.time.dt.hour == 5)
    assert np.all(dset.time.dt.month == 6)
    assert np.all(dset.time.dt.year == 2020)


def test_timecombo_single_day(era5_netcdf):
    tp = Field.from_xarray(era5_netcdf, ncvar="tp")
    res = tp.sel(time={"year": 2020, "day": 10, "hour": [22, 5, 4]})
    dset = res.to_xarray(True)
    assert np.all(
        (dset.time.dt.hour == 4) | (dset.time.dt.hour == 5) | (dset.time.dt.hour == 22)
    )
    assert np.all(dset.time.dt.day == 10)
    assert np.all(dset.time.dt.month == 6)
    assert np.all(dset.time.dt.year == 2020)


@pytest.mark.skip("Should lat and lon depend on points if crs is RegularLatLon?")
def test_locations_regular_latlon_single_lat_multiple_lon(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")

    res = d2m.locations(latitude=[41, 41], longitude=[9, 12])
    assert res["latitude"].type is CoordinateType.INDEPENDENT
    assert res["longitude"].type is CoordinateType.INDEPENDENT
    assert np.all(res.latitude.values == 41)
    assert np.all((res.longitude.values == 9) | (res.longitude.values == 12))

    dset = res.to_xarray()
    assert np.all(dset.latitude == 41)
    assert np.all((dset.longitude == 9) | (dset.longitude == 12))
    assert dset["d2m"].attrs["units"] == "K"
    coords = dset["d2m"].attrs.get(
        "coordinates", dset["d2m"].encoding.get("coordinates")
    )
    assert "latitude" in coords


@pytest.mark.skip(
    f"Lat depends on `points` but is single-element and should be SCALAR not DEPENDENT"
)
def test_locations_regular_latlon_single_lat_single_lon(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")
    res = d2m.locations(latitude=41, longitude=9)
    assert np.all(res.latitude.values == 41)
    assert np.all(res.longitude.values == 9)
    assert res["latitude"].type is CoordinateType.SCALAR
    assert res["longitude"].type is CoordinateType.SCALAR


def test_locations_regular_latlon_multiple_lat_multiple_lon(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")

    res = d2m.locations(latitude=[41, 42], longitude=[9, 12])
    assert np.all((res.latitude.values == 41) | (res.latitude.values == 42))
    assert np.all((res.longitude.values == 9) | (res.longitude.values == 12))

    dset = res.to_xarray()
    assert np.all((dset.latitude == 41) | (dset.latitude == 42))
    assert np.all((dset.longitude == 9) | (dset.longitude == 12))
    assert dset["d2m"].attrs["units"] == "K"
    coords_str = dset["d2m"].attrs.get(
        "coordinates", dset["d2m"].encoding.get("coordinates")
    )
    assert set(coords_str.split(" ")) == {"latitude", "longitude"}


@pytest.mark.skip("`as_cartopy_crs` is not implemented for NEMO CurvilinearGrid")
def test_locations_curvilinear_grid_multiple_lat_multiple_lon(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.locations(latitude=[-20, -21], longitude=[-111, -114])

    dset = res.to_xarray(True)
    assert "points" in dset.dims
    assert dset.points.shape == (2,)
    assert np.all((dset.nav_lat.values + 20 < 0.2) | (dset.nav_lat.values + 21 < 0.2))
    assert np.all((dset.nav_lon.values + 111 < 0.2) | (dset.nav_lon.values + 114 < 0.2))
    assert dset.vt.attrs["units"] == "degree_C m/s"
    assert "coordinates" not in dset.vt.attrs


def test_sel_fail_on_missing_x_y(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    with pytest.raises(ex.HCubeKeyError, match=r"Axis of type*"):
        _ = vt.sel(depth=[1.2, 29], x=slice(60, 100), y=slice(130, 170))


def test_nemo_sel_vertical_fail_on_missing_value_if_method_undefined(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    with pytest.raises(KeyError):
        _ = vt.sel(depth=[1.2, 29])


def test_nemo_sel_vertical_with_std_name(nemo_ocean_16):
    nemo_ocean_16.depthv.attrs["standard_name"] = "vertical"
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    res = vt.sel(depth=[1.2, 29], method="nearest")
    assert len(res.vertical) == 2
    assert np.all(
        (res["vertical"].values == nemo_ocean_16.depthv.values[1])
        | (res["vertical"].values == nemo_ocean_16.depthv.values[-2])
    )


def test_nemo_sel_time_empty_result(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    res = vt.sel(time={"hour": 13}, method="nearest")
    assert res["time"].shape == (0,)


def test_rotated_pole_sel_rlat_rlon_with_axis(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")
    res = wso.sel(y=slice(-1.7, -0.9), x=slice(4, 4.9))
    assert len(res[Axis("x")]) > 0
    assert len(res[Axis("y")]) > 0
    assert np.all(res[Axis("y")].values >= -1.7)
    assert np.all(res[Axis("y")].values <= -0.9)
    assert np.all(res[Axis("x")].values >= 4.0)
    assert np.all(res[Axis("x")].values <= 4.9)


def test_rotated_pole_sel_rlat_rlon_with_std_name(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")
    res = wso.sel(grid_latitude=slice(-1.7, -0.9), grid_longitude=slice(4, 4.9))
    assert len(res[Axis("rlat")]) > 0
    assert len(res[Axis("rlon")]) > 0
    assert np.all(res[Axis("rlat")].values >= -1.7)
    assert np.all(res[Axis("rlat")].values <= -0.9)
    assert np.all(res[Axis("rlon")].values >= 4.0)
    assert np.all(res[Axis("rlon")].values <= 4.9)

    assert np.all(res[Axis("y")].values >= -1.7)
    assert np.all(res[Axis("y")].values <= -0.9)
    assert np.all(res[Axis("x")].values >= 4.0)
    assert np.all(res[Axis("x")].values <= 4.9)


def test_rotated_pole_sel_lat_with_std_name_fails(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")
    with pytest.raises(KeyError):
        _ = wso.sel(latitude=slice(39, 41))


def test_rotated_pole_sel_time_with_diff_ncvar(era5_rotated_netcdf):
    era5_rotated_netcdf = era5_rotated_netcdf.rename({"time": "tm"})
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")
    res = wso.sel(time=slice("2007-05-02T00:00:00", "2007-05-02T11:00:00"))
    assert len(res.time) > 0
    assert np.all(res[Axis("time")].values <= np.datetime64("2007-05-02T11:00:00"))
    assert np.all(res[Axis("time")].values >= np.datetime64("2007-05-02T00:00:00"))
    assert np.all(res.time.values <= np.datetime64("2007-05-02T11:00:00"))
    assert np.all(res.time.values >= np.datetime64("2007-05-02T00:00:00"))
    assert np.all(res["time"].values <= np.datetime64("2007-05-02T11:00:00"))
    assert np.all(res["time"].values >= np.datetime64("2007-05-02T00:00:00"))

    dset = res.to_xarray(encoding=True)
    assert "time" not in dset.coords
    assert "tm" in dset.coords


def test_nemo_sel_proper_ncvar_name_in_res(nemo_ocean_16):
    nemo_ocean_16["vt"].attrs["standard_name"] = "vt_std_name"
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    assert vt.name == "vt_std_name"
    assert vt.ncvar == "vt"
    res = vt.sel(depth=1.2, method="nearest")
    assert res.name == "vt_std_name"
    assert res.ncvar == "vt"

    dset = res.to_xarray(encoding=True)
    assert "vt" in dset.data_vars
    assert "vt_std_name" not in dset.data_vars
    assert "nav_lat" in dset.coords
    assert "latitude" not in dset.coords
    assert "bounds_lat" in dset.coords
    assert "bounds_lon" in dset.coords

    dset = res.to_xarray(encoding=False)
    assert "vt" not in dset.data_vars
    assert "vt_std_name" in dset.data_vars
    assert "nav_lat" not in dset.coords
    assert "latitude" in dset.coords
    assert "bounds_lat" in dset.coords
    assert "bounds_lon" in dset.coords


def test_field_create_with_dict_coords():
    dims = ("time", Axis("latitude"), AxisType.LONGITUDE)
    coords = {
        "time": pd.date_range("06-06-2019", "19-12-2019", periods=50),
        AxisType.LATITUDE: np.linspace(15, 100, 40),
        AxisType.LONGITUDE: np.linspace(5, 10, 30),
    }
    f = Field(
        name="ww",
        data=np.random.random((50, 40, 30)),
        dims=dims,
        coords=coords,
        units="m",
        encoding={"name": "w_ncvar"},
    )
    assert f.name == "ww"
    assert f.ncvar == "w_ncvar"
    assert f.dim_names == ("time", "latitude", "longitude")
    assert f.domain.crs == RegularLatLon()
    assert np.all(
        f[Axis("time")].values
        == np.array(pd.date_range("06-06-2019", "19-12-2019", periods=50))
    )
    assert np.all(
        f[AxisType.TIME].values
        == np.array(pd.date_range("06-06-2019", "19-12-2019", periods=50))
    )
    assert np.all(f[AxisType.LATITUDE].values == np.linspace(15, 100, 40))
    assert np.all(f[Axis("lat")].values == np.linspace(15, 100, 40))
    assert np.all(f[Axis("lon")].values == np.linspace(5, 10, 30))
    assert np.all(f[Axis("longitude")].values == np.linspace(5, 10, 30))
    assert np.all(f[AxisType.LONGITUDE].values == np.linspace(5, 10, 30))
    assert f.units._unit == cf.Unit("m")


def test_var_name_when_field_from_field_id_is_missing(era5_rotated_netcdf):
    wso = Field.from_xarray(
        era5_rotated_netcdf,
        ncvar="W_SO",
        id_pattern="{standard_name}_{not_existing_fied}",
    )
    assert wso.name == "W_SO"
    assert wso.latitude.name == "lat"
    assert wso.longitude.name == "lon"
    assert wso.time.name == "time"

    wso = Field.from_xarray(
        era5_rotated_netcdf, ncvar="W_SO", id_pattern="{standard_name}"
    )
    assert wso.name == "lwe_thickness_of_moisture_content_of_soil_layer"
    assert wso.latitude.name == "latitude"
    assert wso.longitude.name == "longitude"
    assert wso.time.name == "time"


def test_to_xarray_time_with_bounds(era5_rotated_netcdf, nemo_ocean_16):
    field = Field.from_xarray(era5_rotated_netcdf, "W_SO")
    da = field.to_xarray(encoding=False)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bnds"
    assert "time_bnds" in da.coords

    da = field.to_xarray(encoding=True)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bnds"
    assert "time_bnds" in da.coords

    field = Field.from_xarray(nemo_ocean_16, "vt")
    da = field.to_xarray(encoding=False)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_centered_bounds"
    assert "time_centered_bounds" in da.coords

    da = field.to_xarray(encoding=True)
    assert "bounds" in da["time_centered"].encoding
    assert da["time_centered"].encoding["bounds"] == "time_centered_bounds"
    assert "time_centered_bounds" in da.coords


def test_to_xarray_time_with_bounds_mapping(era5_rotated_netcdf):
    field = Field.from_xarray(
        era5_rotated_netcdf, "W_SO", mapping={"time_bnds": {"name": "time_bounds_name"}}
    )
    da = field.to_xarray(encoding=False)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bounds_name"
    assert "time_bounds_name" in da.coords


def test_to_xarray_time_with_bounds_nemo_with_mapping(nemo_ocean_16):
    field = Field.from_xarray(
        nemo_ocean_16, "vt", mapping={"time_centered_bounds": {"name": "time_bnds"}}
    )
    da = field.to_xarray(encoding=False)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bnds"
    assert "time_bnds" in da.coords

    da = field.to_xarray(encoding=True)
    assert "bounds" in da["time_centered"].encoding
    assert da["time_centered"].encoding["bounds"] == "time_bnds"
    assert "time_bnds" in da.coords
