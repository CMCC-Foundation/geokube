import numpy as np
import pandas as pd
import pytest

import geokube.core.coord_system as crs
import geokube.utils.exceptions as ex
from geokube.core.axis import Axis
from geokube.core.bounds import Bounds1D, BoundsND
from geokube.core.coordinate import Coordinate, CoordinateType
from geokube.core.domain import Domain, DomainType
from geokube.core.unit import Unit
from geokube.core.variable import Variable
from tests import compare_dicts
from tests.fixtures import *


def test_from_xarray_rotated_pole_wso(era5_rotated_netcdf):
    domain = Domain.from_xarray(era5_rotated_netcdf, ncvar="W_SO")
    # TODO: domaintype is currently not set
    # assert domain.type is DomainType.GRIDDED
    assert domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    assert "time" in domain
    assert "latitude" in domain
    assert "longitude" in domain
    assert "depth" in domain
    assert "grid_latitude" in domain
    assert "grid_longitude" in domain
    assert domain.vertical.has_bounds
    assert isinstance(domain.vertical.bounds["soil1_bnds"], Bounds1D)

    res = domain.to_xarray(encoding=False)
    assert isinstance(res, xr.core.coordinates.DatasetCoordinates)
    assert "time" in res
    assert "latitude" in res
    assert "longitude" in res
    assert "depth" in res
    assert "grid_latitude" in res
    assert "grid_longitude" in res


def test_from_xarray_rotated_pole_tmin2m(era5_rotated_netcdf):
    domain = Domain.from_xarray(era5_rotated_netcdf, ncvar="TMIN_2M")
    # TODO: domaintype is currently not set
    # assert domain.type is DomainType.GRIDDED
    assert domain.crs == crs.RotatedGeogCS(
        grid_north_pole_latitude=47, grid_north_pole_longitude=-168
    )
    assert "time" in domain
    assert "latitude" in domain
    assert "longitude" in domain
    assert "height" in domain
    assert "grid_latitude" in domain
    assert "grid_longitude" in domain

    res = domain.to_xarray(encoding=True)
    assert isinstance(res, xr.core.coordinates.DatasetCoordinates)
    assert "time" in res
    assert "lat" in res
    assert "lon" in res
    assert "height_2m" in res
    assert "rlat" in res
    assert "rlon" in res


def test_from_xarray_curvilinear_grid(nemo_ocean_16):
    domain = Domain.from_xarray(nemo_ocean_16, ncvar="vt")
    assert "time" in domain
    assert "latitude" in domain
    assert "longitude" in domain
    assert "depthv" in domain  # domain.depthv.attrs['name'] is `depthv`
    assert domain["depthv"].units == Unit("m")
    assert "x" not in domain
    assert "y" not in domain
    assert domain.time.has_bounds
    assert isinstance(domain.time.bounds["time_centered_bounds"], Bounds1D)
    assert domain.vertical.has_bounds
    assert isinstance(domain.vertical.bounds["depthv_bounds"], Bounds1D)
    assert domain.latitude.has_bounds
    assert isinstance(domain.latitude.bounds["bounds_lat"], BoundsND)
    assert domain.longitude.has_bounds
    assert isinstance(domain.longitude.bounds["bounds_lon"], BoundsND)


def test_from_xarray_regular_latlon(era5_globe_netcdf):
    domain = Domain.from_xarray(era5_globe_netcdf, ncvar="tp")
    res = domain.to_xarray()
    assert "latitude" in domain
    assert "latitude" in res
    assert "units" in res["latitude"].attrs
    assert res["latitude"].attrs == era5_globe_netcdf["latitude"].attrs
    compare_dicts(
        res["latitude"].encoding,
        era5_globe_netcdf["latitude"].encoding,
        exclude_d1="name",
    )
    assert res["latitude"].encoding["name"] == "latitude"
    assert "longitude" in domain
    assert "longitude" in res
    assert "units" in res["longitude"].attrs
    compare_dicts(
        res["longitude"].encoding,
        era5_globe_netcdf["longitude"].encoding,
        exclude_d1="name",
    )
    assert res["longitude"].attrs == era5_globe_netcdf["longitude"].attrs
    assert res["longitude"].encoding["name"] == "longitude"
    assert "time" in domain
    assert "time" in res
    assert "units" in res["time"].encoding
    assert "calendar" in res["time"].encoding
    assert res["time"].attrs == era5_globe_netcdf["time"].attrs
    compare_dicts(
        res["time"].encoding, era5_globe_netcdf["time"].encoding, exclude_d1="name"
    )
    assert res["time"].encoding["name"] == "time"
    assert domain.crs == crs.RegularLatLon()
    assert "crs" in res
    assert res["crs"].attrs == {
        "semi_major_axis": 6371229.0,
        "grid_mapping_name": "latitude_longitude",
    }
