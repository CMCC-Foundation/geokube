import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from geokube.backend.netcdf import open_datacube
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.domain import Domain
from geokube.core.variable import Variable
import pytest

from geokube.core.axis import Axis, Axis
from geokube.core.datacube import DataCube
from tests.fixtures import *
from tests import RES_PATH, clear_test_res
