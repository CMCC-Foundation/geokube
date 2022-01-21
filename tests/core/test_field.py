import numpy as np
import xarray as xr
from geokube.backend.netcdf import open_datacube, open_dataset
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.datacube import DataCube
from geokube.core.dimension import Dimension
from geokube.core.domain import Domain
from geokube.core.variable import Variable
import pytest

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, AxisType
from geokube.core.enums import LongitudeConvention, MethodType
from geokube.core.field import Field
from geokube.utils import util_methods
from tests.fixtures import *
from tests import RES_PATH, clear_test_res


def test_from_xarray_dataarray(era5_globe_netcdf):
    with pytest.raises(ex.HCubeTypeError):
        _ = Field.from_xarray_dataarray(era5_globe_netcdf)

    field = Field.from_xarray_dataarray(era5_globe_netcdf["tp"])
    assert np.all(
        field.domain[AxisType.LATITUDE].data == era5_globe_netcdf["latitude"].values
    )
    assert np.all(
        field.domain[AxisType.LONGITUDE].data == era5_globe_netcdf["longitude"].values
    )
    assert np.all(field.domain[AxisType.TIME].data == era5_globe_netcdf["time"].values)


def test_from_xarray_dataarray_2(era5_rotated_netcdf_wso):
    with pytest.raises(ex.HCubeTypeError):
        _ = Field.from_xarray_dataarray(era5_rotated_netcdf_wso)

    field = Field.from_xarray_dataarray(era5_rotated_netcdf_wso["W_SO"])
    assert np.all(
        field.domain[AxisType.X].data == era5_rotated_netcdf_wso["rlon"].values
    )
    assert np.all(
        field.domain[AxisType.Y].data == era5_rotated_netcdf_wso["rlat"].values
    )
    assert np.all(
        field.domain[AxisType.TIME].data == era5_rotated_netcdf_wso["time"].values
    )
    assert field.domain[AxisType.TIME].bounds is None
    assert field.domain[AxisType.VERTICAL].bounds is None
    assert field.cell_methods.method == MethodType.MEAN
    assert field.cell_methods.axis == "soil1"
    assert str(field.cell_methods) == "soil1: mean"


def test_from_xarray_dataset(era5_globe_netcdf):

    with pytest.raises(ex.HCubeTypeError):
        _ = Field.from_xarray_dataset(era5_globe_netcdf["tp"], "a")

    field = Field.from_xarray_dataset(era5_globe_netcdf, field_name="tp")
    assert np.all(
        field.domain[AxisType.LATITUDE].data == era5_globe_netcdf["latitude"].values
    )
    assert np.all(
        field.domain[AxisType.LONGITUDE].data == era5_globe_netcdf["longitude"].values
    )
    assert np.all(field.domain[AxisType.TIME].data == era5_globe_netcdf["time"].values)


def test_from_xarray_dataset_2(era5_rotated_netcdf_wso):
    field = Field.from_xarray_dataset(era5_rotated_netcdf_wso, field_name="W_SO")
    assert np.all(
        field.domain[AxisType.X].data == era5_rotated_netcdf_wso["rlon"].values
    )
    assert np.all(
        field.domain[AxisType.Y].data == era5_rotated_netcdf_wso["rlat"].values
    )
    assert np.all(
        field.domain[AxisType.TIME].data == era5_rotated_netcdf_wso["time"].values
    )
    assert field.domain[AxisType.TIME].bounds is not None
    assert field.domain[AxisType.VERTICAL].bounds is not None
    assert field.cell_methods.method == MethodType.MEAN
    assert field.cell_methods.axis == "soil1"
    assert str(field.cell_methods) == "soil1: mean"


def test_from_xarray_dataset_3(nemo_ocean_16):
    field = Field.from_xarray_dataset(nemo_ocean_16, field_name="vt")
    assert np.all(field.domain[AxisType.X].data == nemo_ocean_16["x"].values)
    assert np.all(field.domain[AxisType.Y].data == nemo_ocean_16["y"].values)
    assert np.all(
        field.domain[AxisType.TIME].data == nemo_ocean_16["time_counter"].values
    )
    assert field.domain[AxisType.TIME].bounds is not None
    assert field.domain[AxisType.VERTICAL].bounds is not None
    assert np.all(
        field.domain[AxisType.LONGITUDE].bounds.data
        == nemo_ocean_16["bounds_lon"].values
    )
    assert np.all(
        field.domain[AxisType.LATITUDE].bounds.data
        == nemo_ocean_16["bounds_lat"].values
    )


def test_from_xarray_field_id(nemo_ocean_16):
    field = Field.from_xarray_dataset(
        nemo_ocean_16, field_name="vt", field_id="{online_operation}:{interval_write}"
    )
    assert field.name == "average:1 month"

    field = Field.from_xarray_dataarray(
        nemo_ocean_16["vt"], field_id="{interval_operation}:{interval_write}"
    )
    assert field.name == "200 s:1 month"


def test_from_xarray_dataset_with_ancillary_vars():
    # TODO:
    pass


def test_to_xarray_1(nemo_ocean_16):
    res = Field.from_xarray_dataset(nemo_ocean_16, field_name="vt").to_xarray()
    assert res.dims == nemo_ocean_16.dims
    for c in nemo_ocean_16.coords.keys():
        assert np.all(res[c] == nemo_ocean_16[c])
        for attk in nemo_ocean_16[c].attrs.keys():
            if attk == "units":
                continue  # in encoding!
            assert res[c].attrs[attk] == nemo_ocean_16[c].attrs[attk]

    res.to_netcdf(RES_PATH)
    ds = xr.open_dataset(RES_PATH, decode_coords="all")
    # x,y are dimensions without coordinates in original file
    # crs is created by GeoKube as object keeping info about cooridnate reference system
    assert set(ds.coords.keys()) - set(nemo_ocean_16.coords.keys()) == {"x", "y", "crs"}
    clear_test_res()


def test_to_xarray_2(era5_rotated_netcdf_wso):
    res = Field.from_xarray_dataset(
        era5_rotated_netcdf_wso, field_name="W_SO"
    ).to_xarray()
    res.to_netcdf(RES_PATH)
    ds = xr.open_dataset(RES_PATH, decode_coords="all")
    assert set(ds.coords.keys()) - set(era5_rotated_netcdf_wso.coords.keys()) == {"crs"}
    assert set(era5_rotated_netcdf_wso.coords.keys()) - set(ds.coords.keys()) == {
        "rotated_pole"
    }
    for d in ds.dims:
        assert d in era5_rotated_netcdf_wso.dims
        assert np.all(ds[d].values == era5_rotated_netcdf_wso[d].values)

    for d in era5_rotated_netcdf_wso.dims:
        assert d in ds.dims

    da = ds["W_SO"]
    dat = era5_rotated_netcdf_wso["W_SO"]

    assert set(da.coords.keys()) - set(dat.coords.keys()) == {"crs"}
    assert set(dat.coords.keys()) - set(da.coords.keys()) == {"rotated_pole"}
    for d in da.dims:
        assert d in dat.dims
        assert np.all(da[d].values == dat[d].values)

    for d in dat.dims:
        assert d in da.dims

    assert dat.attrs == da.attrs
    assert dat.encoding.pop("grid_mapping") == "rotated_pole"
    assert da.encoding.pop("grid_mapping") == "crs"
    assert da.encoding.pop("source") != dat.encoding.pop("source")
    clear_test_res()


def test_sel(era5_globe_netcdf, era5_netcdf, nemo_ocean_16):
    ind1 = {
        "latitude": slice(44, 36),
        "longitude": slice(10, 17),
        "time": {"hour": [10, 15]},
    }
    f = Field.from_xarray_dataset(era5_netcdf, field_name="tp")
    res = f.sel(ind1)
    assert np.all(res.domain[AxisType.LATITUDE].data >= 36)
    assert np.all(res.domain[AxisType.LATITUDE].data <= 44)
    assert np.all(res.domain[AxisType.LONGITUDE].data >= 10)
    assert np.all(res.domain[AxisType.LONGITUDE].data <= 17)
    da = res.domain[AxisType.TIME].to_xarray_dataarray()
    assert np.all((da.dt.hour == 10) | (da.dt.hour == 15))

    ind2 = {
        "latitude": slice(90, 78),
        "longitude": slice(-20, 20),
        "time": {"day": 10, "hour": 22},
    }
    f = Field.from_xarray_dataset(era5_globe_netcdf, field_name="tp")
    res = f.sel(ind2, roll_if_needed=False)
    assert np.all(res.domain[AxisType.LATITUDE].data >= 78)
    assert np.all(res.domain[AxisType.LATITUDE].data <= 90)
    assert np.all(res.domain[AxisType.LONGITUDE].data <= 20)
    assert np.all(res.domain[AxisType.LONGITUDE].data >= 0)
    assert (
        res.domain[AxisType.LONGITUDE].convention is LongitudeConvention.POSITIVE_WEST
    )
    da = res.domain[AxisType.TIME].to_xarray_dataarray()
    assert np.all((da.dt.day == 10))
    assert np.all((da.dt.hour == 15))

    res = f.sel(ind2, roll_if_needed=True)
    assert np.all(res.domain[AxisType.LATITUDE].data >= 78)
    assert np.all(res.domain[AxisType.LATITUDE].data <= 90)
    assert np.all(res.domain[AxisType.LONGITUDE].data <= 20)
    assert np.all(res.domain[AxisType.LONGITUDE].data >= -20)
    assert (
        res.domain[AxisType.LONGITUDE].convention is LongitudeConvention.NEGATIVE_WEST
    )
    da = res.domain[AxisType.TIME].to_xarray_dataarray()
    assert np.all((da.dt.day == 10))
    assert np.all((da.dt.hour == 15))


def test_geobbox(era5_globe_netcdf):
    f = Field.from_xarray_dataset(era5_globe_netcdf, field_name="tp")
    res = f.geobbox(north=90, south=70, west=60, east=90, roll_if_needed=True)
    assert np.all(res.domain[AxisType.LATITUDE].values <= 90)
    assert np.all(res.domain["latitude"].values >= 70)
    assert np.all(res.domain[AxisType.LONGITUDE].values <= 90)
    assert np.all(res.domain["longitude"].values >= 60)


def test_geobbox_2(era5_globe_netcdf):
    f = Field.from_xarray_dataset(era5_globe_netcdf, field_name="tp")
    res = f.geobbox(north=90, south=70, west=-20, east=20, roll_if_needed=True)
    assert np.all(res.domain[AxisType.LATITUDE].values <= 90)
    assert np.all(res.domain["latitude"].values >= 70)
    assert np.all(res.domain[AxisType.LONGITUDE].values <= 20)
    assert np.all(res.domain["longitude"].values >= -20)

    res = f.geobbox(north=90, south=70, west=-20, east=20, roll_if_needed=False)
    assert np.all(res.domain[AxisType.LATITUDE].values <= 90)
    assert np.all(res.domain["latitude"].values >= 70)
    assert np.all(res.domain[AxisType.LONGITUDE].values <= 20)
    assert np.all(res.domain["longitude"].values >= 0)


def test_geobbox_3(era5_rotated_netcdf_wso):
    f = Field.from_xarray_dataset(era5_rotated_netcdf_wso, field_name="W_SO")
    res = f.geobbox(
        north=39, south=41, west=16, east=19, roll_if_needed=False
    )  # rollin supported only for independent variables
    # TODO: verify somehow indrect selection
    pass


def test_locations(era5_globe_netcdf):
    f = Field.from_xarray_dataset(era5_globe_netcdf, field_name="tp")
    res = f.locations(latitude=[40, 46], longitude=[10, 10]).to_xarray()
    assert np.all((res.latitude.values == 40) | (res.latitude.values == 46))
    assert np.all(res.longitude.values == 10)


def test_locations_2(era5_rotated_netcdf_wso):
    f = Field.from_xarray_dataset(era5_rotated_netcdf_wso, field_name="W_SO")
    res = f.locations(latitude=[38, 37], longitude=[15.5, 17.6]).to_xarray()
    assert len(res.points) == 2
    min_idx1 = np.unravel_index(
        np.argmin(
            (era5_rotated_netcdf_wso.lat.values - 38) ** 2
            + (era5_rotated_netcdf_wso.lon.values - 15.5) ** 2
        ),
        shape=era5_rotated_netcdf_wso.lat.shape,
    )
    min_idx2 = np.unravel_index(
        np.argmin(
            (era5_rotated_netcdf_wso.lat.values - 37) ** 2
            + (era5_rotated_netcdf_wso.lon.values - 17.6) ** 2
        ),
        shape=era5_rotated_netcdf_wso.lat.shape,
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lat.values[min_idx1] - res.points.lat[0]).values
        < 1e-5
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lat.values[min_idx2] - res.points.lat[1]).values
        < 1e-5
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lon.values[min_idx1] - res.points.lon[0]).values
        < 1e-5
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lon.values[min_idx2] - res.points.lon[1]).values
        < 1e-5
    )

    res = f.locations(latitude=[38, 42], longitude=20.1).to_xarray()
    assert len(res.points) == 2
    min_idx1 = np.unravel_index(
        np.argmin(
            (era5_rotated_netcdf_wso.lat.values - 38) ** 2
            + (era5_rotated_netcdf_wso.lon.values - 20.1) ** 2
        ),
        shape=era5_rotated_netcdf_wso.lat.shape,
    )
    min_idx2 = np.unravel_index(
        np.argmin(
            (era5_rotated_netcdf_wso.lat.values - 42) ** 2
            + (era5_rotated_netcdf_wso.lon.values - 20.1) ** 2
        ),
        shape=era5_rotated_netcdf_wso.lat.shape,
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lat.values[min_idx1] - res.points.lat[0]).values
        < 1e-5
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lat.values[min_idx2] - res.points.lat[1]).values
        < 1e-5
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lon.values[min_idx1] - res.points.lon[0]).values
        < 1e-5
    )
    assert (
        np.abs(era5_rotated_netcdf_wso.lon.values[min_idx2] - res.points.lon[1]).values
        < 1e-5
    )


def test_locations_3(nemo_ocean_16):
    f = Field.from_xarray_dataset(nemo_ocean_16, field_name="vt")
    res = f.locations(latitude=[-27, -17], longitude=-115).to_xarray()
    assert len(res.points) == 2
    min_idx1 = np.unravel_index(
        np.argmin(
            (nemo_ocean_16.nav_lat.values + 27) ** 2
            + (nemo_ocean_16.nav_lon.values + 115) ** 2
        ),
        shape=nemo_ocean_16.nav_lat.shape,
    )
    min_idx2 = np.unravel_index(
        np.argmin(
            (nemo_ocean_16.nav_lat.values + 17) ** 2
            + (nemo_ocean_16.nav_lon.values + 115) ** 2
        ),
        shape=nemo_ocean_16.nav_lat.shape,
    )
    assert (
        np.abs(nemo_ocean_16.nav_lat.values[min_idx1] - res.points.nav_lat[0]).values
        < 1e-5
    )
    assert (
        np.abs(nemo_ocean_16.nav_lat.values[min_idx2] - res.points.nav_lat[1]).values
        < 1e-5
    )
    assert (
        np.abs(nemo_ocean_16.nav_lon.values[min_idx1] - res.points.nav_lon[0]).values
        < 1e-5
    )
    assert (
        np.abs(nemo_ocean_16.nav_lon.values[min_idx2] - res.points.nav_lon[1]).values
        < 1e-5
    )


def test_locations_4(era5_globe_netcdf):
    f = Field.from_xarray_dataset(era5_globe_netcdf, field_name="tp")
    res = f.locations(latitude=40, longitude=10).to_xarray()
    assert "latitude" in res
    assert "longitude" in res
    assert "time" in res


def test_regrid(nemo_ocean_16):
    f = Field.from_xarray_dataset(nemo_ocean_16, field_name="vt")
    lat_step, lon_step = 0.1, 0.1
    coord = f.domain.coordinate
    lat, lon = coord("latitude").values, coord("longitude").values

    lat_axis = Axis(atype=AxisType.LATITUDE)
    lat_dim = Dimension(name="latitude", axis=lat_axis)
    lon_axis = Axis(atype=AxisType.LONGITUDE)
    lon_dim = Dimension(name="longitude", axis=lon_axis)

    lat_coord = Coordinate(
        variable=Variable(
            name="lat",
            data=np.arange(lat.min(), lat.max() + 0.5 * lat_step, lat_step),
            dims=lat_dim,
        ),
        axis=lat_axis,
    )
    lon_coord = Coordinate(
        variable=Variable(
            name="lon",
            data=np.arange(lon.min(), lon.max() + 0.5 * lon_step, lon_step),
            dims=lon_dim,
        ),
        axis=lon_axis,
    )

    target_domain = Domain(
        coordinates=[lat_coord, lon_coord],
        crs=RegularLatLon(),
    )

    res = f.regrid(target_domain)
    assert isinstance(res.domain.crs, RegularLatLon)
    rxr = res.to_xarray()
    for a in ["nav_lat", "nav_lon", "time_counter", "time_centered", "depthv"]:
        assert a in res.domain._coords
    for a in [
        "nav_lat",
        "nav_lon",
        "time_counter",
        "time_counter_bounds",
        "time_centered",
        "time_centered_bounds",
        "depthv",
        "depthv_bounds",
    ]:
        assert a in rxr.coords

    f = Field.from_xarray_dataset(
        nemo_ocean_16, field_name="vt", field_id="{online_operation}"
    )
    res = f.regrid(target_domain)
    assert isinstance(res.domain.crs, RegularLatLon)
    rxr = res.to_xarray()
    assert "average" == f.name
    assert "average" == res.name
    assert "average" in rxr


def test_resample(era5_netcdf, era5_rotated_netcdf_wso):
    f = Field.from_xarray_dataset(era5_netcdf, field_name="tp")
    with pytest.raises(ex.HCubeValueError):
        _ = f.resample(operator="max", frequency="1D").to_xarray()

    res = f.resample(operator="maximum", frequency="1D").to_xarray()
    assert np.all(res.time.dt.hour == 0)
    assert np.all(res.time.dt.day <= 30)
    assert np.all(res.time.dt.day >= 1)

    res = f.resample(operator="maximum", frequency="1M").to_xarray()
    assert len(res.time) == 1

    f = Field.from_xarray_dataset(era5_rotated_netcdf_wso, field_name="W_SO")
    res = f.resample(operator="sum", frequency="1D").to_xarray()
    assert len(res.time) == 3
    assert np.all(
        era5_rotated_netcdf_wso["W_SO"].isel(time=[0]).sum("time").values
        == res["W_SO"].isel(time=0).values
    )
    assert np.all(
        era5_rotated_netcdf_wso["W_SO"].isel(time=slice(1, 25)).sum("time").values
        == res["W_SO"].isel(time=1).values
    )
    assert np.all(
        era5_rotated_netcdf_wso["W_SO"].isel(time=slice(25, None)).sum("time").values
        == res["W_SO"].isel(time=2).values
    )


def test_plot():
    # TODO
    pass
