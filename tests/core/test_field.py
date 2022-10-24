import os
import numpy as np
import pytest
import xarray as xr
import pandas as pd
import cf_units as cf

import geokube.core.coord_system as crs
from geokube.backend import open_datacube, open_dataset
from geokube.core.axis import Axis, AxisType
from geokube.core.coord_system import GeogCS, RegularLatLon
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.datacube import DataCube
from geokube.core.domain import (
    Domain,
    DomainType,
    GeodeticGrid,
    GeodeticPoints,
)
from geokube.core.enums import LongitudeConvention, MethodType
from geokube.core.field import Field
from geokube.core.unit import Unit
from geokube.core.variable import Variable
from geokube.utils import util_methods
from tests import RES_PATH, clear_test_res, compare_dicts
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
    assert set(
        xr_res["air_temperature"].encoding["coordinates"].split(" ")
    ) == {
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


def test_from_xarray_rotated_pole_with_mapping_and_id_pattern(
    era5_rotated_netcdf,
):
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
    field = Field.from_xarray(
        era5_netcdf, ncvar="tp", id_pattern="{__ddsapi_name}"
    )
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
        __ = tp.geobbox(
            north=10, south=-10, west=50, east=80, top=5, bottom=10
        )

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


def test_locations_regular_latlon_1(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")

    lat, lon = 40, 15
    res = tp.locations(latitude=lat, longitude=lon)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "latitude"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "longitude"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon)


def test_locations_regular_latlon_2(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")

    lat, lon = [35, 40], [15, 20]
    res = tp.locations(latitude=lat, longitude=lon)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "latitude"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "longitude"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon)


def test_locations_regular_latlon_wrong_vertical(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")

    with pytest.raises(
        KeyError,
        match="Axis of type `AxisType.VERTICAL` does not exist in the domain!",
    ):
        tp.locations(latitude=[35, 40], longitude=[15, 20], vertical=[1, 2])


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


def test_geobbox_rotated_pole_partial_arguments_1(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")

    res = wso.geobbox(north=40, west=16)
    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "lon"
    assert res.domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values <= 40) / W > 0.95
    assert np.sum(res.longitude.values >= 16) / W > 0.95

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


def test_geobbox_rotated_pole_partial_arguments_2(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")

    res = wso.geobbox(north=40)
    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "lon"
    assert res.domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values <= 40) / W > 0.95

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


def test_locations_rotated_pole_1(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")

    lat, lon = 40, 15
    res = wso.locations(latitude=lat, longitude=lon)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "lon"
    assert res.domain.type.name == "POINTS"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat, atol=0.1)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon, atol=0.1)


def test_locations_rotated_pole_2(era5_rotated_netcdf):
    wso = Field.from_xarray(era5_rotated_netcdf, ncvar="W_SO")

    lat, lon = [38, 40], [15, 18]
    res = wso.locations(latitude=lat, longitude=lon)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "lon"
    assert res.domain.type.name == "POINTS"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat, atol=0.1)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon, atol=0.1)


def test_geobbox_curvilinear_grid_all(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    assert vt.latitude.name == "latitude"
    assert vt.longitude.name == "longitude"
    assert vt.latitude.ncvar == "nav_lat"
    assert vt.longitude.ncvar == "nav_lon"
    assert vt.vertical.name == "depthv"
    assert vt.vertical.ncvar == "depthv"

    res = vt.geobbox(
        north=-19, south=-22, west=-115, east=-110, bottom=-5, top=-1
    )
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= -22) / W > 0.95
    assert np.sum(res.latitude.values <= -19) / W > 0.95
    assert np.sum(res.longitude.values >= -115) / W > 0.95
    assert np.sum(res.longitude.values <= -110) / W > 0.95
    assert res.vertical.values.size == 5
    assert np.all((res.vertical.values >= 1) & (res.vertical.values <= 5))


def test_geobbox_curvilinear_grid_all_wrong_vertical(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.geobbox(
        north=-19, south=-22, west=-115, east=-110, bottom=1, top=5
    )
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= -22) / W > 0.95
    assert np.sum(res.latitude.values <= -19) / W > 0.95
    assert np.sum(res.longitude.values >= -115) / W > 0.95
    assert np.sum(res.longitude.values <= -110) / W > 0.95
    assert res.vertical.values.size == 0


def test_geobbox_curvilinear_grid_horizontal_all(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.geobbox(north=-19, south=-22, west=-115, east=-110)
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values >= -22) / W > 0.95
    assert np.sum(res.latitude.values <= -19) / W > 0.95
    assert np.sum(res.longitude.values >= -115) / W > 0.95
    assert np.sum(res.longitude.values <= -110) / W > 0.95


def test_geobbox_curvilinear_grid_horizontal_partial_1(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.geobbox(north=-19, west=-115)
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values <= -19) / W > 0.95
    assert np.sum(res.longitude.values >= -115) / W > 0.95


def test_geobbox_curvilinear_grid_horizontal_partial_2(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.geobbox(north=-19)
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs

    W = np.prod(res.latitude.shape)
    assert np.sum(res.latitude.values <= -19) / W > 0.95


def test_geobbox_curvilinear_grid_vertical_both(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.geobbox(bottom=-5, top=-1)
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs

    assert res.vertical.values.size > 0
    assert np.all((res.vertical.values >= 1) & (res.vertical.values <= 5))


def test_geobbox_curvilinear_grid_vertical_partial(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.geobbox(top=-2.5)
    assert res.latitude.name == "latitude"
    assert res.longitude.name == "longitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"
    assert res.domain.crs == vt.domain.crs
    assert res.vertical.values.size > 0
    assert np.all(res.vertical.values >= 2.5)


def test_locations_curvilinear_grid_horizontal_1(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    lat, lon = -20, -115
    res = vt.locations(latitude=lat, longitude=lon)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat, atol=0.1)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon, atol=0.1)

    assert res.vertical.type.name == "INDEPENDENT"
    assert res.vertical.shape == vt.vertical.shape
    assert np.allclose(res.vertical.values, vt.vertical.values)


def test_locations_curvilinear_grid_horizontal_2(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    lat, lon = [-20, -22], [-115, -120]
    res = vt.locations(latitude=lat, longitude=lon)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat, atol=0.1)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon, atol=0.1)

    assert res.vertical.type.name == "INDEPENDENT"
    assert res.vertical.shape == vt.vertical.shape
    assert np.allclose(res.vertical.values, vt.vertical.values)


def test_locations_curvilinear_grid_all_1(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    lat, lon, vert = -20, -115, -5
    res = vt.locations(latitude=lat, longitude=lon, vertical=vert)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat, atol=0.1)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon, atol=0.1)

    assert res.vertical.type.name == "DEPENDENT"
    assert len(res.vertical.dims) == 1
    assert res.vertical.dims[0].name == "points"
    assert np.allclose(
        -res.vertical.values, vert, atol=np.diff(vt.vertical.values).max() / 2
    )


def test_locations_curvilinear_grid_all_2(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    lat, lon, vert = [-20, -22], [-115, -120], [-5, -10]
    res = vt.locations(latitude=lat, longitude=lon, vertical=vert)

    assert res.latitude.name == "latitude"
    assert res.latitude.ncvar == "nav_lat"
    assert res.longitude.name == "longitude"
    assert res.longitude.ncvar == "nav_lon"
    assert res.vertical.name == "vertical"
    assert res.vertical.ncvar == "depthv"

    assert res.latitude.type.name == "DEPENDENT"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert np.allclose(res.latitude.values, lat, atol=0.1)

    assert res.longitude.type.name == "DEPENDENT"
    assert len(res.longitude.dims) == 1
    assert res.longitude.dims[0].name == "points"
    assert np.allclose(res.longitude.values, lon, atol=0.1)

    assert res.vertical.type.name == "DEPENDENT"
    assert len(res.vertical.dims) == 1
    assert res.vertical.dims[0].name == "points"
    assert np.allclose(
        -res.vertical.values, vert, atol=np.diff(vt.vertical.values).max() / 2
    )


def test_timecombo_single_hour(era5_netcdf):
    tp = Field.from_xarray(era5_netcdf, ncvar="tp")
    res = tp.sel(time={"year": 2020, "day": [1, 6, 10], "hour": 5})
    dset = res.to_xarray(True)
    assert np.all(
        (dset.time.dt.day == 1)
        | (dset.time.dt.day == 6)
        | (dset.time.dt.day == 10)
    )
    assert np.all(dset.time.dt.hour == 5)
    assert np.all(dset.time.dt.month == 6)
    assert np.all(dset.time.dt.year == 2020)


def test_timecombo_single_day(era5_netcdf):
    tp = Field.from_xarray(era5_netcdf, ncvar="tp")
    res = tp.sel(time={"year": 2020, "day": 10, "hour": [22, 5, 4]})
    dset = res.to_xarray(True)
    assert np.all(
        (dset.time.dt.hour == 4)
        | (dset.time.dt.hour == 5)
        | (dset.time.dt.hour == 22)
    )
    assert np.all(dset.time.dt.day == 10)
    assert np.all(dset.time.dt.month == 6)
    assert np.all(dset.time.dt.year == 2020)


# TODO: verify!
@pytest.mark.skip(
    "Should lat and lon depend on points if crs is RegularLatLon?"
)
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


# TODO: verify!
@pytest.mark.skip(
    f"Lat depends on `points` but is single-element and should be SCALAR not"
    f" DEPENDENT"
)
def test_locations_regular_latlon_single_lat_single_lon(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")
    res = d2m.locations(latitude=41, longitude=9)
    assert np.all(res.latitude.values == 41)
    assert np.all(res.longitude.values == 9)
    assert res["latitude"].type is CoordinateType.SCALAR
    assert res["longitude"].type is CoordinateType.SCALAR


def test_locations_regular_latlon_single_point(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")

    res = d2m.locations(latitude=41, longitude=9)
    assert res.latitude.values.item() == 41
    assert res.longitude.values.item() == 9
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert res.latitude.type.name == "DEPENDENT"
    assert res.longitude.type.name == "DEPENDENT"
    assert res.domain.type.name == "POINTS"

    dset = res.to_xarray()
    assert dset.latitude.item() == 41
    assert dset.longitude.item() == 9
    assert dset["d2m"].attrs["units"] == "K"
    coords_str = dset["d2m"].attrs.get(
        "coordinates", dset["d2m"].encoding.get("coordinates")
    )
    assert set(coords_str.split(" ")) == {"latitude", "longitude"}
    assert "points" in res.to_xarray().dims
    assert "latitude" not in res.to_xarray().dims
    assert "longitude" not in res.to_xarray().dims


def test_locations_regular_latlon_multiple_lat_multiple_lon(era5_netcdf):
    d2m = Field.from_xarray(era5_netcdf, ncvar="d2m")

    res = d2m.locations(latitude=[41, 42], longitude=[9, 12])
    assert np.all((res.latitude.values == 41) | (res.latitude.values == 42))
    assert np.all((res.longitude.values == 9) | (res.longitude.values == 12))
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert len(res.latitude.dims) == 1
    assert res.latitude.dims[0].name == "points"
    assert res.latitude.type.name == "DEPENDENT"
    assert res.longitude.type.name == "DEPENDENT"
    assert res.domain.type.name == "POINTS"

    dset = res.to_xarray()
    assert np.all((dset.latitude == 41) | (dset.latitude == 42))
    assert np.all((dset.longitude == 9) | (dset.longitude == 12))
    assert dset["d2m"].attrs["units"] == "K"
    coords_str = dset["d2m"].attrs.get(
        "coordinates", dset["d2m"].encoding.get("coordinates")
    )
    assert set(coords_str.split(" ")) == {"latitude", "longitude"}
    assert "points" in res.to_xarray().dims
    assert "latitude" not in res.to_xarray().dims
    assert "longitude" not in res.to_xarray().dims


@pytest.mark.skip(
    "`as_cartopy_crs` is not implemented for NEMO CurvilinearGrid"
)
def test_locations_curvilinear_grid_multiple_lat_multiple_lon(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")

    res = vt.locations(latitude=[-20, -21], longitude=[-111, -114])

    dset = res.to_xarray(True)
    assert "points" in dset.dims
    assert dset.points.shape == (2,)
    assert np.all(
        (dset.nav_lat.values + 20 < 0.2) | (dset.nav_lat.values + 21 < 0.2)
    )
    assert np.all(
        (dset.nav_lon.values + 111 < 0.2) | (dset.nav_lon.values + 114 < 0.2)
    )
    assert dset.vt.attrs["units"] == "degree_C m/s"
    assert "coordinates" not in dset.vt.attrs


def test_sel_fail_on_missing_x_y(nemo_ocean_16):
    vt = Field.from_xarray(nemo_ocean_16, ncvar="vt")
    with pytest.raises(KeyError, match=r"Axis of type*"):
        _ = vt.sel(depth=[1.2, 29], x=slice(60, 100), y=slice(130, 170))


def test_nemo_sel_vertical_fail_on_missing_value_if_method_undefined(
    nemo_ocean_16,
):
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


@pytest.mark.skipif(
    xr.__version__ > "2022.1.0",
    reason="Subsetting does not support empty indexers",
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
    res = wso.sel(
        grid_latitude=slice(-1.7, -0.9), grid_longitude=slice(4, 4.9)
    )
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
    assert np.all(
        res[Axis("time")].values <= np.datetime64("2007-05-02T11:00:00")
    )
    assert np.all(
        res[Axis("time")].values >= np.datetime64("2007-05-02T00:00:00")
    )
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
    assert da["time"].encoding["bounds"] == "time_counter_bounds"
    assert "time_counter_bounds" in da.coords

    da = field.to_xarray(encoding=True)
    assert "bounds" in da["time_counter"].encoding
    assert da["time_counter"].encoding["bounds"] == "time_counter_bounds"
    assert "time_counter_bounds" in da.coords


def test_to_xarray_time_with_bounds_mapping(era5_rotated_netcdf):
    field = Field.from_xarray(
        era5_rotated_netcdf,
        "W_SO",
        mapping={"time_bnds": {"name": "time_bounds_name"}},
    )
    da = field.to_xarray(encoding=False)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bounds_name"
    assert "time_bounds_name" in da.coords


def test_to_xarray_time_with_bounds_nemo_with_mapping(nemo_ocean_16):
    field = Field.from_xarray(
        nemo_ocean_16,
        "vt",
        mapping={"time_counter_bounds": {"name": "time_bnds"}},
    )
    da = field.to_xarray(encoding=False)
    assert "bounds" in da["time"].encoding
    assert da["time"].encoding["bounds"] == "time_bnds"
    assert "time_bnds" in da.coords

    da = field.to_xarray(encoding=True)
    assert "bounds" in da["time_counter"].encoding
    assert da["time_counter"].encoding["bounds"] == "time_bnds"
    assert "time_bnds" in da.coords


def test_geobbox_regular_latlon_negative_convention(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    res = tp.geobbox(north=10, south=-10, west=25, east=25)
    assert np.all(res.longitude <= 25)
    assert np.all(res.longitude >= 25)
    assert np.all(res.latitude <= 10)
    assert np.all(res.latitude >= -10)


def test_sel_longitude_with_leftnone_slice(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    selected_longitude = tp.sel(longitude=slice(None, 20)).longitude.values
    assert np.all(selected_longitude <= 20)
    assert np.all(selected_longitude >= np.min(tp.longitude.values))


def test_sel_longitude_with_rightnone_slice(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    selected_longitude = tp.sel(longitude=slice(45, None)).longitude.values
    assert np.all(selected_longitude <= np.max(tp.longitude.values))
    assert np.all(selected_longitude >= 45)


def test_sel_longitude_with_leftnone_slice_change_convention(
    era5_globe_netcdf,
):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    selected_longitude = tp.sel(longitude=slice(None, -10)).longitude.values
    assert np.all(selected_longitude <= -10)
    assert np.all(selected_longitude >= -180)


def test_sel_longitude_with_rightnone_slice_change_convention(
    era5_globe_netcdf,
):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    selected_longitude = tp.sel(longitude=slice(-20, None)).longitude.values
    assert np.all(selected_longitude <= 180)
    assert np.all(selected_longitude >= -20)


def test_sel_latitude_with_leftnone_slice(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    selected_latitude = tp.sel(latitude=slice(None, 20)).latitude.values
    assert np.all(selected_latitude >= 20)
    assert np.all(selected_latitude <= np.max(tp.latitude.values))


def test_sel_latitude_with_rightnone_slice(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    selected_latitude = tp.sel(latitude=slice(45, None)).latitude.values
    assert np.all(selected_latitude >= np.min(tp.latitude.values))
    assert np.all(selected_latitude <= 45)


def test_interpolate_regular_to_regular_gridded(era5_netcdf):
    field = Field.from_xarray(era5_netcdf["d2m"].to_dataset(), ncvar="d2m")
    loc = {
        "latitude": np.linspace(35, 48, num=20),
        "longitude": np.linspace(2, 20, num=20),
    }
    domain = GeodeticGrid(**loc)
    domain.type = DomainType.GRIDDED
    res = field.interpolate(domain=domain, method="linear")
    assert res.domain.crs == domain.crs
    assert res.domain.type == domain.type
    dims = {dim.type for dim in res.dims.flat}
    assert AxisType.LATITUDE in dims
    assert AxisType.LONGITUDE in dims
    assert res.latitude.size == loc["latitude"].size
    assert res.longitude.size == loc["longitude"].size


def test_interpolate_regular_to_regular_point(era5_netcdf):
    field = Field.from_xarray(era5_netcdf["d2m"].to_dataset(), ncvar="d2m")
    loc = {
        "latitude": np.linspace(35, 48, num=20),
        "longitude": np.linspace(2, 20, num=20),
    }
    domain = GeodeticPoints(**loc)
    res = field.interpolate(domain=domain, method="linear")
    assert res.domain.crs == domain.crs
    assert res.domain.type == domain.type
    assert res.latitude.dims.size == 1
    assert res.latitude.dims[0].name == "points"
    assert res.latitude.size == loc["latitude"].size
    assert res.longitude.dims.size == 1
    assert res.longitude.dims[0].name == "points"
    assert res.longitude.size == loc["longitude"].size


def test_interpolate_rotated_pole_to_regular_gridded(era5_rotated_netcdf_wso):
    wso = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    loc = {
        "latitude": np.linspace(35, 48, num=20),
        "longitude": np.linspace(2, 20, num=20),
    }
    domain = GeodeticGrid(**loc)
    domain.type = DomainType.GRIDDED
    res = wso.interpolate(domain=domain, method="linear")
    assert res.domain.crs == domain.crs
    assert res.domain.type == domain.type
    dims = {dim.type for dim in res.dims.flat}
    assert AxisType.LATITUDE in dims
    assert AxisType.LONGITUDE in dims
    assert res.latitude.size == loc["latitude"].size
    assert res.longitude.size == loc["longitude"].size


def test_interpolate_rotated_pole_to_regular_point(era5_rotated_netcdf_wso):
    wso = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    loc = {
        "latitude": np.linspace(35, 48, num=20),
        "longitude": np.linspace(2, 20, num=20),
    }
    domain = GeodeticPoints(**loc)
    res = wso.interpolate(domain=domain, method="linear")
    assert res.domain.crs == domain.crs
    assert res.domain.type == domain.type
    assert res.latitude.dims.size == 1
    assert res.latitude.dims[0].name == "points"
    assert res.latitude.size == loc["latitude"].size
    assert res.longitude.dims.size == 1
    assert res.longitude.dims[0].name == "points"
    assert res.longitude.size == loc["longitude"].size


def test_resample_without_original_bounds(era5_globe_netcdf):
    tp = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    tp_r = tp.resample(operator="maximum", frequency="W")
    assert tp.values.max() == tp_r.values.max()
    diff = 7 * 24 * 60 * 60 * 1_000_000_000
    diff_ = np.diff(tp_r.time.values)
    assert np.all(diff_ == np.full_like(diff_, fill_value=diff))
    assert tp_r.time.bounds is not None
    assert tp_r.time.bounds["time_bounds"].shape[0] == tp_r.time.shape[0]


def test_resample_with_original_bounds(era5_rotated_netcdf_wso):
    wso = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    wso_r = wso.resample(operator="sum", frequency="12H")
    diff = 12 * 60 * 60 * 1_000_000_000
    diff_ = np.diff(wso_r.time.values)
    assert np.all(diff_ == np.full_like(diff_, fill_value=diff))
    assert wso_r.time.bounds is not None
    assert wso_r.time.bounds["time_bounds"].shape[0] == wso_r.time.shape[0]


def test_adding_time_bounds(era5_netcdf):
    field = Field.from_xarray(era5_netcdf["d2m"].to_dataset(), ncvar="d2m")
    assert field.time.bounds is None
    time_bounds = np.empty(shape=(field.time.size, 2), dtype=field.time.dtype)
    time_resolution = field.time.values[1] - field.time.values[0]
    time_bounds[0, 0] = field.time.values[0] - time_resolution
    time_bounds[1:, 0] = field.time.values[:-1]
    time_bounds[:, 1] = field.time.values
    field.time.bounds = time_bounds
    assert isinstance(field.time.bounds, dict)
    assert "time_bounds" in field.time.bounds
    assert field.time.bounds["time_bounds"].shape == field.time.shape + (2,)
    assert "bounds" in field.time.encoding
    assert field.time.encoding["bounds"] == "time_bounds"


def test_regridding_regular_to_regular_conservative(era5_netcdf):
    field_in = Field.from_xarray(era5_netcdf, ncvar="d2m")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="conservative",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    assert field_out.coords.keys() == {"time", "latitude", "longitude"}
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_regular_to_regular_bilinear(era5_netcdf):
    field_in = Field.from_xarray(era5_netcdf, ncvar="d2m")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="bilinear",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    assert field_out.coords.keys() == {"time", "latitude", "longitude"}
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_regular_to_regular_nearest_s2d(era5_netcdf):
    field_in = Field.from_xarray(era5_netcdf, ncvar="d2m")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="nearest_s2d",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    assert field_out.coords.keys() == {"time", "latitude", "longitude"}
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_regular_to_regular_nearest_d2s(era5_netcdf):
    field_in = Field.from_xarray(era5_netcdf, ncvar="d2m")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="nearest_d2s",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    assert field_out.coords.keys() == {"time", "latitude", "longitude"}
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_regular_to_regular_patch(era5_netcdf):
    field_in = Field.from_xarray(era5_netcdf, ncvar="d2m")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="patch",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    assert field_out.coords.keys() == {"time", "latitude", "longitude"}
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_rotated_pole_to_regular_bilinear(era5_rotated_netcdf_wso):
    field_in = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="bilinear",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    dims = field_in.coords.keys() - {"grid_latitude", "grid_longitude"}
    assert field_out.coords.keys() == dims
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_rotated_pole_to_regular_nearest_s2d(
    era5_rotated_netcdf_wso,
):
    field_in = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="nearest_s2d",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    dims = field_in.coords.keys() - {"grid_latitude", "grid_longitude"}
    assert field_out.coords.keys() == dims
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_rotated_pole_to_regular_nearest_d2s(
    era5_rotated_netcdf_wso,
):
    field_in = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="nearest_d2s",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    dims = field_in.coords.keys() - {"grid_latitude", "grid_longitude"}
    assert field_out.coords.keys() == dims
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_regridding_rotated_pole_to_regular_patch(era5_rotated_netcdf_wso):
    field_in = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    lat_in, lon_in = field_in.latitude, field_in.longitude

    lat_step = lon_step = 0.1
    lat = np.arange(lat_in.min(), lat_in.max(), lat_step)
    lon = np.arange(lon_in.min(), lon_in.max(), lon_step)
    target = Domain(
        coords=[
            Coordinate(
                data=lat,
                axis=Axis(name="latitude", is_dim=True),
                dims=("latitude",),
            ),
            Coordinate(
                data=lon,
                axis=Axis(name="longitude", is_dim=True),
                dims=("longitude",),
            ),
        ],
        crs=GeogCS(6371229),
        domaintype=DomainType.GRIDDED,
    )
    field_out = field_in.regrid(
        target=target,
        method="patch",
        weights_path=None,
        reuse_weights=True,
    )

    assert field_out.domain.crs == target.crs
    dims = field_in.coords.keys() - {"grid_latitude", "grid_longitude"}
    assert field_out.coords.keys() == dims
    assert field_in.time.shape == field_out.time.shape
    assert np.all(field_in.time.values == field_out.time.values)
    assert target.latitude.shape == field_out.latitude.shape
    assert np.allclose(target.latitude.values, field_out.latitude.values)
    assert target.longitude.shape == field_out.longitude.shape
    assert np.allclose(target.longitude.values, field_out.longitude.values)
    assert np.allclose(
        np.nanquantile(field_in.values, [0, 0.25, 0.5, 0.75, 1]),
        np.nanquantile(field_out.values, [0, 0.25, 0.5, 0.75, 1]),
        atol=1,
    )


def test_sel_by_time_combo_only_day(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    field = field.sel(time={"day": 10})
    assert len(field.time) == 24


def test_resample_if_gap_in_time_axis(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    field = field.sel(time={"day": [10, 15], "hour": [10, 16]})
    val1 = (
        field.to_xarray(False)
        .tp.isel(time=field.time.to_xarray(False).time.dt.day.values == 10)
        .sum(dim=field.time.name)
    )
    val2 = (
        field.to_xarray(False)
        .tp.isel(time=field.time.to_xarray(False).time.dt.day.values == 15)
        .sum(dim=field.time.name)
    )
    result = field.resample(frequency="1D", operator="sum")
    assert len(result.time) == 2
    assert np.allclose(result.to_xarray(False).tp.values[0, ...], val1)
    assert np.allclose(result.to_xarray(False).tp.values[1, ...], val2)


def test_auxiliary_coords_after_resampling(era5_rotated_netcdf_tmin2m):
    field = Field.from_xarray(era5_rotated_netcdf_tmin2m, ncvar="TMIN_2M")
    assert "latitude" in field.domain.coords
    assert field.domain.coords["latitude"].type is CoordinateType.DEPENDENT
    assert "longitude" in field.domain.coords
    assert field.domain.coords["longitude"].type is CoordinateType.DEPENDENT
    field = field.resample(frequency="1D", operator="min")
    assert "latitude" in field.domain.coords
    assert field.domain.coords["latitude"].type is CoordinateType.DEPENDENT
    assert "longitude" in field.domain.coords
    assert field.domain.coords["longitude"].type is CoordinateType.DEPENDENT


@pytest.mark.skip("Currently domaintype is set only in `locations` method")
def test_field_domain_type_regular_lat_lon(
    era5_netcdf,
):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    assert field.domain.type is DomainType.GRIDDED


def test_keeping_field_domain_type_in_to_xarray_and_from_xarray(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    field = field.locations(latitude=10, longitude=20)
    assert field.domain.type is DomainType.POINTS
    field = field.sel(time={"day": 10})
    assert field.domain.type is DomainType.POINTS


def test_field_serialization_success_with_domtype_attr(era5_netcdf):
    field = Field.from_xarray(era5_netcdf, ncvar="tp")
    field = field.locations(latitude=10, longitude=20)
    assert field.domain.type is DomainType.POINTS
    field.to_netcdf(RES_PATH)
    clear_test_res()


def test_extract_polygons_regular(era5_globe_netcdf, it_shape_100_km):
    tp = era5_globe_netcdf["tp"]
    field = Field.from_xarray(era5_globe_netcdf, ncvar="tp")
    coord_names = field.coords.keys()
    field, mask = field.extract_polygons(
        geometry=it_shape_100_km, crop=True, return_mask=True
    )
    lon_lb = it_shape_100_km.bounds["minx"].min()
    lon_ub = it_shape_100_km.bounds["maxx"].max()
    lat_lb = it_shape_100_km.bounds["miny"].min()
    lat_ub = it_shape_100_km.bounds["maxy"].max()
    assert mask.values.dtype == np.bool_
    assert mask.coords.keys() == {"latitude", "longitude"}
    assert np.allclose(mask.coords["latitude"], tp.coords["latitude"])
    assert np.allclose(mask.coords["longitude"], tp.coords["longitude"])
    assert field.coords.keys() == coord_names
    assert field.latitude.values.min() >= lat_lb
    assert field.latitude.values.max() <= lat_ub
    assert field.longitude.values.min() >= lon_lb
    assert field.longitude.values.max() <= lon_ub
    assert np.nanmin(field.values) >= np.nanmin(tp)
    assert np.nanmin(field.values) <= np.nanmax(tp)


def test_extract_polygons_rotated_pole(
    era5_rotated_netcdf_wso, it_shape_100_km
):
    wso = era5_rotated_netcdf_wso["W_SO"]
    field = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    field, mask = field.extract_polygons(
        geometry=it_shape_100_km, crop=True, return_mask=True
    )
    lon_lb = it_shape_100_km.bounds["minx"].min()
    lon_ub = it_shape_100_km.bounds["maxx"].max()
    lat_lb = it_shape_100_km.bounds["miny"].min()
    lat_ub = it_shape_100_km.bounds["maxy"].max()
    assert mask.values.dtype == np.bool_
    assert mask.coords.keys() == {"latitude", "longitude"}
    assert field.latitude.values.min() >= lat_lb
    assert field.latitude.values.max() <= lat_ub
    assert field.longitude.values.min() >= lon_lb
    assert field.longitude.values.max() <= lon_ub
    assert np.nanmin(field.values) >= np.nanmin(wso)
    assert np.nanmin(field.values) <= np.nanmax(wso)


def test_using_geo_domtype_attribute_in_serialization_field(
    era5_rotated_netcdf_wso,
):
    clear_test_res()
    field = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    field.domain.type = DomainType("points")
    field_dset = field.to_xarray(False)[
        "lwe_thickness_of_moisture_content_of_soil_layer"
    ]
    assert "__geo_domtype" in field_dset.attrs
    assert isinstance(field_dset.attrs["__geo_domtype"], DomainType)
    with pytest.raises(TypeError, match=r"Invalid value for attr*"):
        field_dset.to_netcdf(RES_PATH)

    field_dset = field.to_xarray(True)["W_SO"]
    assert "__geo_domtype" not in field_dset.attrs
    field_dset.to_netcdf(RES_PATH)
    clear_test_res()


def test_keep_encoding_after_to_regular(era5_rotated_netcdf_wso):
    field = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    reg_field = field.to_regular()
    compare_dicts(
        field.encoding,
        reg_field.encoding,
        exclude_d1=["grid_mapping", "coordinates", "_FillValue"],
        exclude_d2=["grid_mapping", "coordinates", "_FillValue"],
    )


def test_keep_coordinate_encoding_with_scalars_after_to_regular(
    era5_rotated_netcdf_wso,
):
    era5_rotated_netcdf_wso = era5_rotated_netcdf_wso.assign_coords({"sc": 10})
    era5_rotated_netcdf_wso["sc"].attrs["standard_name"] = "scalar_coord"
    era5_rotated_netcdf_wso["W_SO"].encoding["coordinates"] = "lat lon sc"
    field = Field.from_xarray(era5_rotated_netcdf_wso, ncvar="W_SO")
    reg_field = field.to_regular()
    xr_ds = reg_field.to_xarray()
    assert xr_ds["W_SO"].encoding["coordinates"] == "sc"
