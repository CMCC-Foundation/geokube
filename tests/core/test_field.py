import pytest

import geokube.core.axis as axis
from geokube.core.field import PointsField, ProfilesField, GridField

from tests.fixtures.features import *

def test_create_pointsfield(f_points_no_ancillary):
    assert f_points_no_ancillary.dim_axes == (axis.vertical, axis.latitude, axis.longitude)