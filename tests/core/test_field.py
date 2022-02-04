import numpy as np
import xarray as xr
from geokube.backend.netcdf import open_datacube, open_dataset
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.datacube import DataCube
from geokube.core.axis import Axis, AxisType
from geokube.core.domain import Domain
from geokube.core.variable import Variable
import pytest

import geokube.utils.exceptions as ex
from geokube.core.unit import Unit
from geokube.core.axis import Axis, Axis
from geokube.core.enums import LongitudeConvention, MethodType
from geokube.core.field import Field
from geokube.utils import util_methods
from tests.fixtures import *
from tests import RES_PATH, clear_test_res
import geokube.core.coord_system as crs


def test_1(era5_rotated_netcdf):
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


def test_2(era5_rotated_netcdf):
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


def test_3(nemo_ocean_16):
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


def test_4(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    assert field._id_pattern is None
    assert field._mapping is None
    assert field.name == "tp"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")

    field = Field.from_xarray(era5_netcdf, ncvar="tp", id_pattern="{__ddsapi_name}")
    assert field._id_pattern == "{__ddsapi_name}"
    assert field._mapping is None
    assert field.name == "total_precipitation"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")

    field = Field.from_xarray(
        era5_netcdf, ncvar="tp", id_pattern="{units}__{long_name}"
    )
    assert field.name == "m__Total precipitation"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")

    field = Field.from_xarray(
        era5_netcdf, ncvar="tp", mapping={"tp": {"name": "tot_prep"}}
    )
    assert field._id_pattern is None
    assert field._mapping == {"tp": {"name": "tot_prep"}}
    assert field.name == "tot_prep"
    assert field.ncvar == "tp"
    assert field.units == Unit("m")


def test_geobbox_1(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    with pytest.raises(
        ex.HCubeNotImplementedError, match=r"Selecting by geobbox containing*"
    ):
        _ = tp.geobbox(north=10, south=-10, west=50, east=80, top=5, bottom=10)

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


def test_geobbox_2(era5_globe_netcdf):
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


def test_geobbox_3(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")

    res = wso.geobbox(north=40, south=38, west=16, east=19)
    assert res.domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= 38) / W > 0.95
    assert np.sum(res.latitude.values <= 40) / W > 0.95
    assert np.sum(res.longitude.values >= 16) / W > 0.95
    assert np.sum(res.longitude.values <= 19) / W > 0.95

    dset = res.to_xarray(True)
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


def test_geobbox_4(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    res = vt.geobbox(north=-19, south=-22, west=-115, east=-110)
    assert res.domain.crs == crs.CurvilinearGrid()

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= -22) / W > 0.95
    assert np.sum(res.latitude.values <= -19) / W > 0.95
    assert np.sum(res.longitude.values >= -115) / W > 0.95
    assert np.sum(res.longitude.values <= -110) / W > 0.95


def test_timecombo_1(era5_netcdf):
    tp = Field.from_xarray(era5_netcdf, ncvar="tp")
    res = tp.sel(time={"year": 2020, "day": [1, 6, 10], "hour": 5})
    dset = res.to_xarray(True)
    assert np.all(
        (dset.time.dt.day == 1) | (dset.time.dt.day == 6) | (dset.time.dt.day == 10)
    )
    assert np.all(dset.time.dt.hour == 5)
    assert np.all(dset.time.dt.month == 6)
    assert np.all(dset.time.dt.year == 2020)

    res = tp.sel(time={"year": 2020, "day": 10, "hour": [22, 5, 4]})
    dset = res.to_xarray(True)
    assert np.all(
        (dset.time.dt.hour == 4) | (dset.time.dt.hour == 5) | (dset.time.dt.hour == 22)
    )
    assert np.all(dset.time.dt.day == 10)
    assert np.all(dset.time.dt.month == 6)
    assert np.all(dset.time.dt.year == 2020)


def test_locations_1(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")

    res = d2m.locations(latitude=41, longitude=[9, 12])
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


def test_locations_2(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")

    res = d2m.locations(latitude=[41, 42], longitude=[9, 12])
    assert np.all((res.latitude.values == 41) | (res.latitude.values == 42))
    assert np.all((res.longitude.values == 9) | (res.longitude.values == 12))

    dset = res.to_xarray()
    assert np.all((dset.latitude == 41) | (dset.latitude == 42))
    assert np.all((dset.longitude == 9) | (dset.longitude == 12))
    assert dset["d2m"].attrs["units"] == "K"
    coords = dset["d2m"].attrs.get(
        "coordinates", dset["d2m"].encoding.get("coordinates")
    )
    assert coords is None


def test_locations_3(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.locations(latitude=[-20, -21], longitude=[-111, -114])

    dset = res.to_xarray(True)
    assert "points" in dset.dims
    assert dset.points.shape == (2,)
    assert np.all((dset.nav_lat.values + 20 < 0.2) | (dset.nav_lat.values + 21 < 0.2))
    assert np.all((dset.nav_lon.values + 111 < 0.2) | (dset.nav_lon.values + 114 < 0.2))
    assert dset.vt.attrs["units"] == "degree_C m/s"
    assert "coordinates" not in dset.vt.attrs
