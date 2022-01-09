import cartopy.crs as ccrs
import numpy as np
from geokube.backend.netcdf import open_datacube
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.dimension import Dimension
from geokube.core.domain import Domain
from geokube.core.variable import Variable
import pytest

from geokube.core.axis import Axis, AxisType
from geokube.core.datacube import DataCube
from tests.fixtures import *


def test_from_xarray_1(era5_netcdf, era5_globe_netcdf):
    dc = DataCube.from_xarray(era5_netcdf)
    assert len(dc) == 2
    res = dc.geobbox(north=90, south=70, west=60, east=90, roll_if_needed=True)
    for f in res.values():
        assert np.all(f.domain[AxisType.LATITUDE].values <= 90)
        assert np.all(f.domain["latitude"].values >= 70)
        assert np.all(f.domain[AxisType.LONGITUDE].values <= 90)
        assert np.all(f.domain["longitude"].values >= 60)

    dc = DataCube.from_xarray(era5_globe_netcdf)
    assert len(dc) == 1
    res = dc.geobbox(north=90, south=70, west=-20, east=20, roll_if_needed=True)
    for f in res.values():
        assert np.all(f.domain[AxisType.LATITUDE].values <= 90)
        assert np.all(f.domain["latitude"].values >= 70)
        assert np.all(f.domain[AxisType.LONGITUDE].values <= 20)
        assert np.all(f.domain["longitude"].values >= -20)


def test_from_xarray_2(era5_rotated_netcdf):
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    assert len(dc) == 2
    res = dc.geobbox(north=39, south=41, west=16, east=19, roll_if_needed=False)
    for f in res.values():
        assert np.all(f.domain[AxisType.LATITUDE].values <= 41 + 1)
        assert np.all(f.domain["latitude"].values >= 39 - 1)
        assert np.all(f.domain[AxisType.LONGITUDE].values <= 19 + 1)
        assert np.all(f.domain["longitude"].values >= 16 - 1)


def test_from_xarray_3(nemo_ocean_16):
    dc = DataCube.from_xarray(nemo_ocean_16)
    assert len(dc) == 1
    res = dc.locations(latitude=[-27, -17], longitude=-115)
    for f in res.values():
        fa = f.to_xarray()
        assert len(fa.points) == 2
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
            np.abs(nemo_ocean_16.nav_lat.values[min_idx1] - fa.points.nav_lat[0]).values
            < 1e-5
        )
        assert (
            np.abs(nemo_ocean_16.nav_lat.values[min_idx2] - fa.points.nav_lat[1]).values
            < 1e-5
        )
        assert (
            np.abs(nemo_ocean_16.nav_lon.values[min_idx1] - fa.points.nav_lon[0]).values
            < 1e-5
        )
        assert (
            np.abs(nemo_ocean_16.nav_lon.values[min_idx2] - fa.points.nav_lon[1]).values
            < 1e-5
        )


def test_from_xarray_4(era5_globe_netcdf):
    dc = DataCube.from_xarray(era5_globe_netcdf)
    assert len(dc) == 1
    res = dc.locations(latitude=-17, longitude=-115).to_xarray()
    assert "latitude" in res
    assert "longitude" in res


@pytest.mark.skip("Skipped due to using a number of files")
def test_large_file():
    import xarray as xr

    dset = xr.open_mfdataset(
        "/data/inputs/ERA5/single-levels/reanalysis/*/*.nc",
        parallel=True,
        decode_coords="all",
        chunks={"latitude": -1, "longitude": -1, "time": 50},
    )
    dc = DataCube.from_xarray(dset)
    res = dc.geobbox(north=90, south=85, west=85, east=90, roll_if_needed=True)
    for f in res.values():
        assert np.all(f.domain[AxisType.LATITUDE].values <= 90)
        assert np.all(f.domain["latitude"].values >= 70)
        assert np.all(f.domain[AxisType.LONGITUDE].values <= 90)
        assert np.all(f.domain["longitude"].values >= 60)


def test_resample(era5_rotated_netcdf):
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    res = dc.resample(operator="sum", frequency="1D")
    for f in res.values():
        fxr = f.to_xarray()
        assert len(fxr.time) == 3
        assert np.all(
            era5_rotated_netcdf[f.name].isel(time=[0]).sum("time").values
            == fxr[f.name].isel(time=0).values
        )
        assert np.all(
            era5_rotated_netcdf[f.name].isel(time=slice(1, 25)).sum("time").values
            == fxr[f.name].isel(time=1).values
        )
        assert np.all(
            era5_rotated_netcdf[f.name].isel(time=slice(25, None)).sum("time").values
            == fxr[f.name].isel(time=2).values
        )


def test_regrid(era5_rotated_netcdf):
    dc = DataCube.from_xarray(era5_rotated_netcdf)
    lat_step, lon_step = 0.1, 0.1
    coord = dc["W_SO"].domain.coordinate
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

    res = dc.regrid(target_domain)
    assert len(res) == 2
    assert "W_SO" in res
    assert "TMIN_2M" in res
    for f in res.values():
        assert isinstance(f.domain.crs, RegularLatLon)


def test_to_xarray(era5_rotated_netcdf):
    res = DataCube.from_xarray(era5_rotated_netcdf).to_xarray()
    for dv in era5_rotated_netcdf.data_vars.keys():
        assert dv in res
        m1 = np.isnan(res[dv].values) & np.isnan(era5_rotated_netcdf[dv].values)
        # NaN == NaN always returns False
        assert np.all((res[dv].values == era5_rotated_netcdf[dv].values) | m1)
    for c in era5_rotated_netcdf.coords.keys():
        if c == "rotated_pole":
            assert np.all(res["crs"].values == era5_rotated_netcdf[c].values)
            continue
        assert c in res
        assert np.all(res[c].values == era5_rotated_netcdf[c].values)
