import os
import cartopy.crs as ccrs
import numpy as np
import pytest
import xarray as xr

import geokube.core.coord_system as crs
from geokube.backend import open_datacube
from geokube.core.axis import Axis, AxisType
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.datacube import DataCube
from geokube.core.field import Field
from geokube.core.domain import Domain
from geokube.core.unit import Unit
from geokube.core.variable import Variable

from tests import RES_PATH, RES_DIR, clear_test_res
from tests.fixtures import *


def test_from_xarray_regular_lat_lon(era5_netcdf):
    dc = DataCube.from_xarray(era5_netcdf)
    assert "tp" in dc
    assert "d2m" in dc
    assert dc.properties == era5_netcdf.attrs
    assert dc.encoding == era5_netcdf.encoding


def test_from_xarray_regular_lat_lon_with_id_pattern(era5_netcdf):
    dc = DataCube.from_xarray(era5_netcdf, id_pattern="{__ddsapi_name}")
    assert "total_precipitation" in dc
    assert "2_metre_dewpoint_temperature" in dc
    assert dc["total_precipitation"].domain.crs == crs.RegularLatLon()
    assert dc["2_metre_dewpoint_temperature"].domain.crs == crs.RegularLatLon()

    assert dc["2_metre_dewpoint_temperature"].units == Unit("K")
    assert dc["total_precipitation"].units == Unit("m")


def test_to_xarray_regular_lat_lon_with_id_pattern_without_encoding(
    era5_netcdf,
):
    dc = DataCube.from_xarray(era5_netcdf, id_pattern="{__ddsapi_name}")
    xr_res = dc.to_xarray(encoding=False)
    assert "2_metre_dewpoint_temperature" in xr_res.data_vars
    assert "total_precipitation" in xr_res.data_vars
    assert "crs" in xr_res.coords


def test_to_xarray_regular_lat_lon_with_id_pattern_with_encoding(era5_netcdf):
    dc = DataCube.from_xarray(era5_netcdf, id_pattern="{__ddsapi_name}")
    xr_res = dc.to_xarray(encoding=True)
    assert "d2m" in xr_res.data_vars
    assert "tp" in xr_res.data_vars
    assert "crs" in xr_res.coords


def test_geobbox_regular_latlon(era5_globe_netcdf):
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


def test_geobbox_rotated_pole(era5_rotated_netcdf):
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
    W = np.prod(
        res["lwe_thickness_of_moisture_content_of_soil_layer"].latitude.shape
    )
    assert (
        np.sum(
            res[
                "lwe_thickness_of_moisture_content_of_soil_layer"
            ].latitude.values
            >= 38
        )
        / W
        > 0.95
    )
    assert (
        np.sum(
            res[
                "lwe_thickness_of_moisture_content_of_soil_layer"
            ].latitude.values
            <= 40
        )
        / W
        > 0.95
    )
    assert (
        np.sum(
            res[
                "lwe_thickness_of_moisture_content_of_soil_layer"
            ].longitude.values
            >= 16
        )
        / W
        > 0.95
    )
    assert (
        np.sum(
            res[
                "lwe_thickness_of_moisture_content_of_soil_layer"
            ].longitude.values
            <= 19
        )
        / W
        > 0.95
    )

    dset = res.to_xarray(encoding=True)
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


def test_getitem_with_coords_and_fiels(era5_rotated_netcdf):
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    assert "W_SO" in dc
    assert "TMIN_2M" in dc
    assert Axis("latitude") in dc
    assert "lat" in dc
    assert AxisType.TIME in dc
    assert "air_temperature" in dc
    assert "lwe_thickness_of_moisture_content_of_soil_layer" in dc
    wso = dc["W_SO"]
    assert wso.name == "lwe_thickness_of_moisture_content_of_soil_layer"
    assert wso.ncvar == "W_SO"
    tmin = dc["TMIN_2M"]
    assert tmin.name == "air_temperature"
    assert tmin.ncvar == "TMIN_2M"
    lat = dc.latitude
    assert lat.name == "latitude"
    assert lat.ncvar == "lat"
    assert lat.axis_type is AxisType.LATITUDE
    time = dc.time
    assert time.ncvar == "time"
    assert time.axis_type is AxisType.TIME
    assert dc["latitude"] is not None
    assert dc.latitude is not None
    assert dc[AxisType.LATITUDE] is not None
    assert dc[Axis("lat")] is not None


def test_locations_rotated_pole(era5_rotated_netcdf):
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    res = dc.locations(latitude=[39, 41], longitude=[17, 17])
    assert "points" in res.to_xarray().dims
    assert len(res.domain[AxisType.LATITUDE]) == 2
    assert len(res.domain[AxisType.LONGITUDE]) == 2
    assert res.domain.crs == RegularLatLon()


def test_get_multiple_fields_by_standard_name_list(rotated_pole_datacube):
    res = rotated_pole_datacube[
        ["lwe_thickness_of_moisture_content_of_soil_layer", "air_temperature"]
    ]
    assert len(res) == 2
    assert "lwe_thickness_of_moisture_content_of_soil_layer" in res.fields
    assert "air_temperature" in res.fields


def test_get_single_field_by_standard_name(rotated_pole_datacube):
    res = rotated_pole_datacube["W_SO"]
    assert isinstance(res, Field)
    assert res.name == "lwe_thickness_of_moisture_content_of_soil_layer"
    assert res.ncvar == "W_SO"

    res = rotated_pole_datacube["TMIN_2M"]
    assert isinstance(res, Field)
    assert res.name == "air_temperature"
    assert res.ncvar == "TMIN_2M"


def test_persist_with_no_extension(rotated_pole_datacube):
    clear_test_res()
    with pytest.warns(UserWarning, match=r"Provided persistance path*"):
        path = rotated_pole_datacube.persist(RES_DIR)
        assert path == RES_DIR + ".nc"
    clear_test_res()


def test_persist_no_path_passed(rotated_pole_datacube):
    path = rotated_pole_datacube.persist()
    assert os.path.exists(path)


def test_to_dict_store_propen_keys(era5_netcdf):
    details = DataCube.from_xarray(era5_netcdf).to_dict()
    assert "domain" in details
    assert isinstance(details["domain"], dict)
    assert "fields" in details
    assert isinstance(details["fields"], dict)
