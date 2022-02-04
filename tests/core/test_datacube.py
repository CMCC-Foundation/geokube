import cartopy.crs as ccrs
import numpy as np
from geokube.core.unit import Unit
import xarray as xr
from geokube.backend.netcdf import open_datacube
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.domain import Domain
from geokube.core.variable import Variable
import geokube.core.coord_system as crs
import pytest

from geokube.core.axis import Axis, Axis
from geokube.core.datacube import DataCube
from tests.fixtures import *
from tests import RES_PATH, clear_test_res


def test_1(era5_netcdf):
    dc = DataCube.from_xarray(era5_netcdf)
    assert "tp" in dc
    assert "d2m" in dc
    assert dc.properties == era5_netcdf.attrs
    assert dc.encoding == era5_netcdf.encoding

    dc = DataCube.from_xarray(era5_netcdf, id_pattern="{__ddsapi_name}")
    assert "total_precipitation" in dc
    assert "2_metre_dewpoint_temperature" in dc
    assert dc["total_precipitation"].domain.crs == crs.RegularLatLon()
    assert dc["2_metre_dewpoint_temperature"].domain.crs == crs.RegularLatLon()

    assert dc["2_metre_dewpoint_temperature"].units == Unit("K")
    assert dc["total_precipitation"].units == Unit("m")

    xr_res = dc.to_xarray(encoding=False)
    assert "2_metre_dewpoint_temperature" in xr_res.data_vars
    assert "total_precipitation" in xr_res.data_vars
    assert "crs" in xr_res.coords

    xr_res = dc.to_xarray(encoding=True)
    assert "d2m" in xr_res.data_vars
    assert "tp" in xr_res.data_vars
    assert "crs" in xr_res.coords


def test_geobbox_1(era5_globe_netcdf):
    dc = DataCube.from_xarray(era5_globe_netcdf)
    res = dc.geobbox(north=10, south=-10, west=-20, east=20)
    assert np.all(res["tp"]["latitude"].values <= 10)
    assert np.all(res["tp"]["latitude"].values >= -10)
    assert np.all(res["tp"].latitude.values <= 10)
    assert np.all(res["tp"].latitude.values >= -10)

    assert np.all(res["tp"]["longitude"].values <= 20)
    assert np.all(res["tp"]["longitude"].values >= -20)
    assert np.all(res["tp"].longitude.values <= 20)
    assert np.all(res["tp"].longitude.values >= -20)

    dset = res.to_xarray(True)
    assert "tp" in dset
    assert np.all(dset.latitude <= 10)
    assert np.all(dset.latitude >= -10)
    assert dset.latitude.attrs["units"] == "degrees_north"

    assert np.all(dset.longitude <= 20)
    assert np.all(dset.longitude >= -20)
    assert dset.longitude.attrs["units"] == "degrees_east"


def test_geobbox_2(era5_rotated_netcdf):
    wso = DataCube.from_xarray(era5_rotated_netcdf)

    res = wso.geobbox(north=40, south=38, west=16, east=19)
    assert res[
        "lwe_thickness_of_moisture_content_of_soil_layer"
    ].domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    assert res["air_temperature"].domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    W = np.prod(res["lwe_thickness_of_moisture_content_of_soil_layer"].latitude.shape)
    assert (
        np.sum(
            res["lwe_thickness_of_moisture_content_of_soil_layer"].latitude.values >= 38
        )
        / W
        > 0.95
    )
    assert (
        np.sum(
            res["lwe_thickness_of_moisture_content_of_soil_layer"].latitude.values <= 40
        )
        / W
        > 0.95
    )
    assert (
        np.sum(
            res["lwe_thickness_of_moisture_content_of_soil_layer"].longitude.values
            >= 16
        )
        / W
        > 0.95
    )
    assert (
        np.sum(
            res["lwe_thickness_of_moisture_content_of_soil_layer"].longitude.values
            <= 19
        )
        / W
        > 0.95
    )

    dset = res.to_xarray(True)
    assert "TMIN_2M" in dset.data_vars
    assert "W_SO" in dset.data_vars
    assert "rlat" in dset.coords
    assert "rlon" in dset.coords
    assert "lat" in dset
    assert dset.lat.attrs["units"] == "degrees_north"
    assert "lon" in dset
    assert dset.lon.attrs["units"] == "degrees_east"
    assert "crs" in dset.coords

    dset = res.to_xarray(False)
    assert "air_temperature" in dset.data_vars
    assert "lwe_thickness_of_moisture_content_of_soil_layer" in dset.data_vars
    assert "latitude" in dset
    assert dset.latitude.attrs["units"] == "degrees_north"
    assert "longitude" in dset
    assert dset.longitude.attrs["units"] == "degrees_east"
    assert "crs" in dset.coords
