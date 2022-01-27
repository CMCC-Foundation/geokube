import numpy as np
import xarray as xr
from geokube.backend.netcdf import open_datacube, open_dataset
from geokube.core.coord_system import RegularLatLon
from geokube.core.coordinate import Coordinate
from geokube.core.datacube import DataCube
from geokube.core.axis import Axis
from geokube.core.domain import Domain
from geokube.core.variable import Variable
import pytest

import geokube.utils.exceptions as ex
from geokube.core.axis import Axis, Axis
from geokube.core.enums import LongitudeConvention, MethodType
from geokube.core.field import Field
from geokube.utils import util_methods
from tests.fixtures import *
from tests import RES_PATH, clear_test_res
