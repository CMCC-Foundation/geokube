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
    # TODO: __ddsapi_name missing for lat/lon/time, what then?
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
