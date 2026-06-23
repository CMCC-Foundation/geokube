import numpy as np

import geokube.core.coord_system as crs
from geokube.backend import open_datacube
from geokube.core.coord_system import GeogCS, RotatedGeogCS
from geokube.core.datacube import DataCube
from geokube.core.domain import Domain
from geokube.core.enums import RegridMethod

from tests.fixtures import *

DOWNSCALED = "tests/resources/era5_downscaled_over_italy.nc"


def _downscaled_subset():
    kube = open_datacube(DOWNSCALED, decode_coords="all")
    return kube.geobbox(north=42.5, south=41.5, west=12.0, east=13.0)


def test_datacube_to_regular_from_rotated_pole():
    sub = _downscaled_subset()
    assert isinstance(sub.domain.crs, RotatedGeogCS)

    reg = sub.to_regular()

    # to_regular turns the dependent 2-D lat/lon of the rotated grid into
    # independent 1-D latitude/longitude dimensions on a regular grid.
    assert not isinstance(reg.domain.crs, RotatedGeogCS)
    field = next(iter(reg.fields.values()))
    assert field.latitude.type.name == "INDEPENDENT"
    assert field.longitude.type.name == "INDEPENDENT"
    assert 0 not in field.shape


def test_datacube_regrid_bilinear(era5_globe_netcdf):
    dc = DataCube.from_xarray(era5_globe_netcdf)
    lat = np.arange(-10.0, 10.0, 2.0)
    lon = np.arange(-20.0, 20.0, 2.0)
    target = Domain(
        coords={"latitude": lat, "longitude": lon}, crs=GeogCS(6371229)
    )

    res = dc.regrid(target, method=RegridMethod.BILINEAR)

    field = next(iter(res.fields.values()))
    assert np.allclose(field.latitude.values, lat)
    assert np.allclose(field.longitude.values, lon)
    assert field.latitude.type.name == "INDEPENDENT"


def test_datacube_resample_mean(era5_netcdf):
    dc = DataCube.from_xarray(era5_netcdf)
    res = dc.resample(frequency="1D", operator="mean")

    assert res.time.size < dc.time.size
    field = next(iter(res.fields.values()))
    # resample now attaches time bounds spanning each resampled period.
    assert field.time.bounds is not None
    assert field.time.bounds["time_bounds"].shape[0] == field.time.shape[0]
