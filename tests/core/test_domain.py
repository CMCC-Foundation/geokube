import numpy as np
import pandas as pd
import pytest

import geokube.core.coord_system as crs
from geokube.core.axis import Axis, AxisType
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
    assert isinstance(domain.time.bounds["time_counter_bounds"], Bounds1D)
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
    compare_dicts(
        res["latitude"].attrs,
        era5_globe_netcdf["latitude"].attrs,
        exclude_d1="standard_name",
    )
    compare_dicts(
        res["latitude"].encoding,
        era5_globe_netcdf["latitude"].encoding,
        exclude_d1=["name", "_FillValue"],
        exclude_d2="_FillValue",
    )
    assert res["latitude"].encoding["_FillValue"] is None
    assert res["latitude"].encoding["name"] == "latitude"
    assert "longitude" in domain
    assert "longitude" in res
    assert "units" in res["longitude"].attrs
    compare_dicts(
        res["longitude"].encoding,
        era5_globe_netcdf["longitude"].encoding,
        exclude_d1=["name", "_FillValue"],
        exclude_d2="_FillValue",
    )
    assert res["latitude"].encoding["_FillValue"] is None
    compare_dicts(
        res["longitude"].attrs,
        era5_globe_netcdf["longitude"].attrs,
        exclude_d1="standard_name",
    )
    assert res["longitude"].encoding["name"] == "longitude"
    assert "time" in domain
    assert "time" in res
    assert "units" in res["time"].encoding
    assert "calendar" in res["time"].encoding
    compare_dicts(
        res["time"].attrs,
        era5_globe_netcdf["time"].attrs,
        exclude_d1="standard_name",
    )
    compare_dicts(
        res["time"].encoding,
        era5_globe_netcdf["time"].encoding,
        exclude_d1=["name", "_FillValue"],
    )
    assert res["time"].encoding["_FillValue"] is None
    assert res["time"].encoding["name"] == "time"
    assert domain.crs == crs.RegularLatLon()
    assert "crs" in res
    assert res["crs"].attrs == {
        "semi_major_axis": 6371229.0,
        "grid_mapping_name": "latitude_longitude",
    }


def test_regular_lat_lon_domain_from_coords_dict():
    coords = {
        "lat": np.arange(1, 10),
        "lon": np.arange(6, 50),
        "height_2m": np.arange(4, 14),
        "time": pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
    }
    dims = ("time", "lat", "lon", "height_2m")
    domain = Domain._make_domain_from_coords_dict_dims_and_crs(coords, dims)
    assert domain.crs == crs.RegularLatLon()
    assert Axis("time") in domain
    assert AxisType.TIME in domain
    assert domain[AxisType.TIME].is_dim
    assert Axis("lat") in domain
    assert AxisType.LATITUDE in domain
    assert domain[Axis("latitude")].is_dim
    assert Axis("lon") in domain
    assert AxisType.LONGITUDE in domain
    assert domain["lon"].is_dim
    assert Axis("vertical") in domain
    assert AxisType.VERTICAL in domain
    assert domain[Axis("height_2m")].is_dim

    coords = {
        AxisType.LONGITUDE: np.arange(1, 10),
        AxisType.LATITUDE: np.arange(6, 50),
        "height_2m": np.arange(4, 14),
        "time": pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
    }
    dims = (AxisType.TIME, Axis("lat"), "lon", "height_2m")
    domain = Domain._make_domain_from_coords_dict_dims_and_crs(coords, dims)
    assert domain.crs == crs.RegularLatLon()
    assert Axis("time") in domain
    assert AxisType.TIME in domain
    assert Axis("lat") in domain
    assert AxisType.LATITUDE in domain
    assert Axis("lon") in domain
    assert AxisType.LONGITUDE in domain
    assert Axis("vertical") in domain
    assert AxisType.VERTICAL in domain


def test_curvilinear_lat_lon_domain_from_coords_dict():
    dims = ("time", "x", "y")
    coords = {
        "lat": ((Axis("x"), Axis("y")), np.array([[0, 1], [1, 2]])),
        "lon": ((Axis("x"), Axis("y")), np.array([[0, 1], [1, 2]])),
        "time": pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
    }
    domain = Domain._make_domain_from_coords_dict_dims_and_crs(
        coords, dims, crs=crs.CurvilinearGrid()
    )
    assert domain.crs == crs.CurvilinearGrid()
    assert Axis("lat") in domain
    assert AxisType.LATITUDE in domain
    assert not domain[AxisType.LATITUDE].is_dim
    assert Axis("lon") in domain
    assert AxisType.LONGITUDE in domain
    assert not domain[AxisType.LONGITUDE].is_dim
    assert Axis("x") not in domain
    assert AxisType.X not in domain
    assert Axis("y") not in domain
    assert AxisType.Y not in domain

    dims = (Axis("time"), AxisType.X, AxisType.Y)
    coords = {
        "lat": ((AxisType.X, AxisType.Y), np.array([[0, 1], [1, 2]])),
        "lon": (("x", "y"), np.array([[0, 1], [1, 2]])),
        "time": pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
    }
    domain = Domain._make_domain_from_coords_dict_dims_and_crs(
        coords, dims, crs=crs.CurvilinearGrid()
    )
    assert domain.crs == crs.CurvilinearGrid()
    assert Axis("lat") in domain
    assert AxisType.LATITUDE in domain
    assert not domain["lat"].is_dim
    assert Axis("lon") in domain
    assert AxisType.LONGITUDE in domain
    assert not domain["lon"].is_dim
    assert Axis("x") not in domain
    assert AxisType.X not in domain
    assert Axis("y") not in domain
    assert AxisType.Y not in domain


def test_rotated_pole_lat_lon_domain_from_coords_dict():
    dims = ("time", AxisType.X, AxisType.Y)
    coords = {
        "lat": ((Axis("rlat"), Axis("rlon")), np.array([[0, 1], [1, 2]])),
        "lon": ((Axis("rlat"), Axis("rlon")), np.array([[0, 1], [1, 2]])),
        "rlat": np.array([6, 7]),
        "rlon": np.array([9, 8]),
        "time": pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
    }
    domain = Domain._make_domain_from_coords_dict_dims_and_crs(
        coords, dims, crs=crs.RotatedGeogCS(12345, 5678)
    )
    assert isinstance(domain.crs, crs.RotatedGeogCS)
    assert Axis("lat") in domain
    assert AxisType.LATITUDE in domain
    assert not domain[AxisType.LATITUDE].is_dim
    assert Axis("lon") in domain
    assert AxisType.LONGITUDE in domain
    assert not domain[AxisType.LONGITUDE].is_dim
    assert Axis("rlat") in domain
    assert AxisType.X in domain
    assert domain[AxisType.X].is_dim
    assert Axis("y") in domain
    assert AxisType.Y in domain
    assert domain[AxisType.Y].is_dim


def test_regular_lat_lon_domain_with_bounds_and_properties_from_dict():
    time_bounds = np.stack(
        (
            pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
            pd.date_range("02-01-2001", "11-01-2001", freq="1D"),
        )
    ).T
    coords = {
        "lat": np.arange(1, 10),
        "lon": np.arange(6, 50),
        "height_2m": np.arange(4, 14),
        "time": (
            ("time",),
            pd.date_range("01-01-2001", "10-01-2001", freq="1D"),
            time_bounds,
            None,
            {"some_prop": "prop_val"},
        ),
    }
    dims = ("time", "lat", "lon", "height_2m")
    domain = Domain._make_domain_from_coords_dict_dims_and_crs(coords, dims)
    assert "time" in domain
    assert domain[AxisType.TIME].has_bounds
    assert "some_prop" in domain[Axis("Time")].properties
    assert domain[Axis("Time")].properties["some_prop"] == "prop_val"


def test_skip_redundand_coord_from_coordinates_string_in_encoding(
    era5_globe_netcdf,
):
    era5_globe_netcdf["tp"].encoding["coordinates"] = "time latitude"
    with pytest.warns(
        UserWarning,
        match=r"Coordinate (time|latitude) was already defined as dimension!",
    ):
        Domain.from_xarray(era5_globe_netcdf, ncvar="tp")


def test_skip_redundand_coord_from_coordinates_string_in_attrs(
    era5_globe_netcdf,
):
    era5_globe_netcdf["tp"].attrs["coordinates"] = "time latitude"
    with pytest.warns(
        UserWarning,
        match=r"Coordinate (time|latitude) was already defined as dimension!",
    ):
        Domain.from_xarray(era5_globe_netcdf, ncvar="tp")


def test_skip_coordinate_from_coordinates_string_if_not_present_in_dataset(
    era5_globe_netcdf,
):
    era5_globe_netcdf["tp"].attrs["coordinates"] = "time not_exsiting"
    with pytest.warns(
        UserWarning,
        match=r"Coordinate not_exsiting does not exist in the dataset!",
    ):
        Domain.from_xarray(era5_globe_netcdf, ncvar="tp")


def test_using_geo_domtype_attribute_for_domain_type(era5_globe_netcdf):
    era5_globe_netcdf["tp"].attrs["__geo_domtype"] = "gridded"
    domain = Domain.from_xarray(era5_globe_netcdf, ncvar="tp")
    assert domain.type is DomainType.GRIDDED


def test_to_dict_store_proper_keys(era5_globe_netcdf):
    details = Domain.from_xarray(era5_globe_netcdf, ncvar="tp").to_dict()
    assert isinstance(details, dict)
    assert "crs" in details
    assert "coordinates" in details


def test_to_dict_store_reg_crs_with_names_and_attributes(era5_globe_netcdf):
    details = Domain.from_xarray(era5_globe_netcdf, ncvar="tp").to_dict()
    crs = details["crs"]
    assert isinstance(crs, dict)
    assert crs["name"] == "latitude_longitude"
    assert "semi_major_axis" in crs
    assert crs["semi_major_axis"] == 6371229.0
    assert "semi_minor_axis" in crs
    assert crs["semi_minor_axis"] == 6371229.0
    assert "inverse_flattening" in crs
    assert crs["inverse_flattening"] == 0.0
    assert "longitude_of_prime_meridian" in crs
    assert crs["longitude_of_prime_meridian"] == 0.0


def test_to_dict_store_rotated_crs_with_names_and_attributes(
    era5_rotated_netcdf,
):
    details = Domain.from_xarray(era5_rotated_netcdf, ncvar="W_SO").to_dict()
    crs = details["crs"]
    assert isinstance(crs, dict)
    assert crs["name"] == "rotated_latitude_longitude"
    assert "grid_north_pole_latitude" in crs
    assert crs["grid_north_pole_latitude"] == 47.0
    assert "grid_north_pole_longitude" in crs
    assert crs["grid_north_pole_longitude"] == -168.0
    assert "north_pole_grid_longitude" in crs
    assert crs["north_pole_grid_longitude"] == 0
    assert "ellipsoid" in crs
    assert crs["ellipsoid"] is None


def test_to_dict_store_coords_reg_crs(era5_globe_netcdf):
    details = Domain.from_xarray(era5_globe_netcdf, ncvar="tp").to_dict()
    coords = details["coordinates"]
    assert len(coords) == 3
    assert isinstance(coords, dict)
    assert set(coords.keys()) == {"time", "latitude", "longitude"}
    for k in coords.values():
        assert "values" in k
        assert isinstance(k["values"], list)


def test_to_dict_store_coords_rot_crs(era5_rotated_netcdf):
    details = Domain.from_xarray(era5_rotated_netcdf, ncvar="W_SO").to_dict()
    coords = details["coordinates"]
    assert len(coords) == 6
    assert isinstance(coords, dict)
    assert set(coords.keys()) == {
        "depth",
        "time",
        "grid_latitude",
        "grid_longitude",
        "latitude",
        "longitude",
    }
    for k in coords.values():
        assert "values" in k
        assert isinstance(k["values"], list)
