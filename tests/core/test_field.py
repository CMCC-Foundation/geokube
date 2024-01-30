import pytest

import geokube.core.axis as axis
from geokube.core.field import PointsField, ProfilesField, GridField

from tests.fixtures.features import *

#
# Field Points Test
#

def test_create_points(f_points_no_ancillary):
    assert f_points_no_ancillary.dim_axes == (axis.time, axis.vertical, axis.latitude, axis.longitude)

#
# Field Profiles Test
#

def test_create_profiles(f_profiles_no_ancillary):
    assert f_profiles_no_ancillary.dim_axes == (axis.time, axis.vertical, axis.latitude, axis.longitude)


#
# Field Grid Test
#

def test_create_grid(f_grid_no_ancillary):
    assert f_grid_no_ancillary.dim_axes == (axis.time, axis.vertical, axis.latitude, axis.longitude)

